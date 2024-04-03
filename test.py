import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import json
import glob
import tqdm
import torch.nn.functional as F
import time

from PIL import Image
import requests
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# for model customization
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg, Attention, Block, checkpoint_seq
from timm.models.layers import trunc_normal_

from torch.utils.data import Dataset, DataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# mode and save
mode = 3
save_model = True

seed = 42
num_classes = 100
batch_size = 32
epoch = 100
weight_decay = 1e-4
learning_rate = 5e-4
mask_ratio = 0.5
mask_low_ratio = 0.1
tau = 1
alpha = 0.5
history = {'loss_cls':[], 'loss_dist':[], 'acc':[]}
duration = 0.0

is_dist = True if mode in [1,2,3] else False
is_mask = True if mode in [2,3] else False

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class CustomAttention(Attention):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = self.attn_drop(attn) @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class CustomBlock(Block):
    def __init__(
            self,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        for kw in ['mlp_ratio', 'init_values', 'drop_path', 'act_layer', 'mlp_layer']:
            kwargs.pop(kw, None)
        self.attn = CustomAttention(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_, attn = self.attn(self.norm1(x))
        x = x + self.drop_path1(self.ls1(x_))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, attn
    

class TeacherVisionTransformer(VisionTransformer):
    def forward_features(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)

        if mask is not None:
            # cls token split
            x_cls, x = x[:,0,:], x[:,1:,:]
            B, N, M = x.shape
            # set mask indices stride per batch
            mask = mask + torch.arange(B, device=x.device).unsqueeze(1) * N
            mask = mask.reshape(-1,1).repeat(1, M)
            # gather topk from each batch
            x = x.reshape(-1, M)
            x = x.gather(0, mask).reshape(B, -1, M)
            # cls token merge
            x_cls = x_cls.unsqueeze(1)
            x = torch.cat([x_cls, x], dim=1)
        
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x, mask

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x, mask = self.forward_features(x, mask)
        x = self.forward_head(x)
        return x, mask


class StudentVisionTransformer(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        attn = []
        for blk in self.blocks:
            x, a = blk(x)
            attn.append(a)
        attn = torch.stack(attn, dim=1)

        x = self.norm(x)
        return x[:, 0], x[:, 1], attn

    def forward(self, x):
        x, x_dist, attn = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist, attn
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2
    
class ImageNetDataset(Dataset):
    def __init__(self):
        super(type(self), self).__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256, antialias=True),
            transforms.RandomCrop(224),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

        self.label_dict = {}
        with open('/home/data/imagenet_val/imagenet_class_index.json') as fs:
            for k, v in json.load(fs).items():
                if int(k) >= num_classes:
                    continue
                self.label_dict[v[0]] = [int(k), v[1]]

        self.image_list = []
        for img in glob.glob('/home/data/imagenet_val/images/*.JPEG'):
            label = img.split('_')[-1].split('.')[0]
            if label not in self.label_dict.keys():
                continue
            self.image_list.append(img)
        
    def __getitem__(self, index):
        img = self.image_list[index]

        label = img.split('_')[-1].split('.')[0]
        label = torch.tensor([self.label_dict[label][0]])
        label = nn.functional.one_hot(label, num_classes).float().view(-1)

        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_list)
    
def distillation_loss(logits, teacher_logits):
    distillation_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits/tau, dim=1), F.softmax(teacher_logits/tau, dim=1)) * (tau * tau)
    return distillation_loss


if is_dist == True:
    # Teacher model
    deit_base_patch16_224 = TeacherVisionTransformer(num_classes=num_classes,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    deit_base_patch16_224.default_cfg = _cfg()


    # teacher class num adjust
    checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
    checkpoint['model']['head.weight'] = checkpoint['model']['head.weight'][:num_classes,:]
    checkpoint['model']['head.bias'] = checkpoint['model']['head.bias'][:num_classes]

    deit_base_patch16_224.load_state_dict(checkpoint["model"])
    deit_base_patch16_224.to(device)
    deit_base_patch16_224.eval()


# Student model
deit_small_distilled_patch16_224 = StudentVisionTransformer(num_classes=num_classes, block_fn=CustomBlock,
    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
deit_small_distilled_patch16_224.default_cfg = _cfg()
deit_small_distilled_patch16_224.to(device)
deit_small_distilled_patch16_224.train()

dataset = ImageNetDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
optimizer = torch.optim.Adam(deit_small_distilled_patch16_224.parameters(), lr=learning_rate, weight_decay=weight_decay)
# load checkpoint
if os.path.exists(f'/home/maskedKD/model_{mode}.pt'):
    checkpoint = torch.load(f'/home/maskedKD/model_{mode}.pt')
    deit_small_distilled_patch16_224.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    history = checkpoint['history']
    duration = checkpoint['duration']

# training
s_time = time.time()
for _ in range(epoch):
    total_loss_cls = 0.0
    total_loss_dist = 0.0
    total_acc = 0.0

    tq = tqdm.tqdm(dataloader, ncols=180)
    for i, (img, label) in enumerate(tq):
        img = img.to(device)
        label = label.to(device)

        out_c, out_s, attn = deit_small_distilled_patch16_224(img)
        if mode == 2:
            '''model_2 attn from cls token only'''
            attn = attn[:,-1,:,0,2:].mean(dim=1)
            mask = torch.topk(attn, int(mask_ratio * attn.size(1)), dim=1).indices
        if mode == 3:
            '''model_3 attn from highest and lowest attn from patches'''
            attn = attn[:,-1,:,0,2:].mean(dim=1)
            mask_high = torch.topk(attn, int(mask_ratio * attn.size(1)), dim=1).indices
            mask_low = torch.topk(attn, int(mask_low_ratio * attn.size(1)), dim=1, largest=False).indices
            mask = torch.cat([mask_high, mask_low], dim=-1)

        loss_cls = F.cross_entropy(out_c, label)
        total_loss_cls += loss_cls.item()
        loss = loss_cls

        if is_dist == True:
            out_t, mask = deit_base_patch16_224(img, mask) if is_mask else deit_base_patch16_224(img)
            loss_dist = distillation_loss(out_s, out_t)
            total_loss_dist += loss_dist.item()
            loss = (1-alpha) * loss + alpha * loss_dist
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_acc += torch.sum(torch.argmax(label, dim=1) == torch.argmax(out_c, dim=1), dim=0).item()

        if is_dist == True:
            tq.set_postfix_str(f'epoch={_+1}, loss={total_loss_cls/(i+1):5.3f}, dist={total_loss_dist/(i+1):5.3f}, acc={total_acc/(i+1)/batch_size:.3f}')
        else:
            tq.set_postfix_str(f'epoch={_+1}, loss={total_loss_cls/(i+1):5.3f}, acc={total_acc/(i+1)/batch_size:.3f}')

    history['loss_cls'].append(total_loss_cls/(i+1))
    history['acc'].append(total_acc/(i+1)/batch_size)
    if is_dist == True:
        history['loss_dist'].append(total_loss_dist/(i+1))
    total_loss_cls = 0.0
    total_loss_dist = 0.0

    if save_model:
        checkpoint = {
            'model': deit_small_distilled_patch16_224.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history':history,
            'duration':duration + time.time()-s_time,
        }
        torch.save(checkpoint, f'/home/maskedKD/model_{mode}.pt')