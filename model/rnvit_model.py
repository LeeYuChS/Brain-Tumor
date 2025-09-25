import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from functools import partial
from torch.hub import load_state_dict_from_url

"""
    Help you understand the code better:
    e.g.
    Original input: torch.Size([1, 3, 224, 224])
    After conv_layers: torch.Size([1, 64, 56, 56])
    After res_block1: torch.Size([1, 128, 28, 28])
    After res_block2: torch.Size([1, 128, 28, 28])
    After res_block3: torch.Size([1, 256, 14, 14])
    After flatten and permute: torch.Size([1, 196, 256])
    After projection: torch.Size([1, 196, 768])
    After norm (final): torch.Size([1, 196, 768])

    Image size: 224x224
    Patch size: 16x16
    Feature map size after CNN: 14x14
    Number of patches: 196
    Traditional ViT patches (224//16)²: 196
"""


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def resize_pos_embed(pos_embed, old_img_size=224, new_img_size=None, patch_size=16, num_tokens=1):
    if new_img_size is None:
        return pos_embed
    cls_tokens = pos_embed[:, :num_tokens, :]
    pos_embed = pos_embed[:, num_tokens:, :]
    old_img_size = (old_img_size, old_img_size) if isinstance(old_img_size, int) else old_img_size
    old_grid_size = (old_img_size[0] // patch_size, old_img_size[1] // patch_size)
    old_num_patches = old_grid_size[0] * old_grid_size[1]
    assert pos_embed.shape[1] == old_num_patches, f"Pos embed shape mismatch {pos_embed.shape[1]} vs {old_num_patches}"
    
    pos_embed = pos_embed.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    if isinstance(new_img_size, int):
        new_img_size = (new_img_size, new_img_size)
    new_grid_size = (new_img_size[0] // patch_size, new_img_size[1] // patch_size)
    pos_embed = F.interpolate(pos_embed, size=new_grid_size, mode='bilinear', align_corners=False)
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_grid_size[0] * new_grid_size[1], -1)
    pos_embed = torch.cat([cls_tokens, pos_embed], dim=1)
    return pos_embed

class CnnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_c, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.res_block1 = ResidualBlock(64, 128, stride=2, downsample=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        ))
        self.res_block2 = ResidualBlock(128, 128)
        self.res_block3 = ResidualBlock(128, 256, stride=2, downsample=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        ))
        self.proj = nn.Linear(256, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        # nn.Identity() is non-functional, so no parameters to initialize

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv_layers(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        # print(f"res block3 output shape: {x.shape}")  # Debug line to check CNN output shape
        x = x.flatten(2).permute(0, 2, 1)
        x = self.proj(x)
        # print(f"Projection output shape: {x.shape}")  # Debug line to check projection output shape
        x = self.norm(x)
        # print(f"Final output shape: {x.shape}")  # Debug line to check final output shape
        return x

class RNViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=2,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.1,
                 attn_drop_ratio=0.1, drop_path_ratio=0., embed_layer=CnnEmbed, norm_layer=None,
                 act_layer=None):
        super(RNViT, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        self.patch_size = patch_size
        self.reference_img_size = img_size if isinstance(img_size, int) else img_size[0]

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim, norm_layer=norm_layer)
        reference_num_patches = (self.reference_img_size // self.patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, reference_num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        # print(f'cls token shape: {cls_token.shape}')  # Debug line to check cls token shape
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(B, -1, -1), x), dim=1)

        new_pos_embed = resize_pos_embed(
            self.pos_embed,
            old_img_size=self.reference_img_size,
            new_img_size=(H, W),
            patch_size=self.patch_size,
            num_tokens=self.num_tokens
        )
        new_pos_embed = new_pos_embed.expand(B, -1, -1)

        x = self.pos_drop(x + new_pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x

def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

def map_rn_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('transformer_encoder.layers.'):
            layer_idx = k.split('.')[2]
            sub_key = '.'.join(k.split('.')[3:])
            if sub_key == 'self_attn.in_proj_weight':
                new_k = f'blocks.{layer_idx}.attn.qkv.weight'
            elif sub_key == 'self_attn.in_proj_bias':
                new_k = f'blocks.{layer_idx}.attn.qkv.bias'
            elif sub_key == 'self_attn.out_proj.weight':
                new_k = f'blocks.{layer_idx}.attn.proj.weight'
            elif sub_key == 'self_attn.out_proj.bias':
                new_k = f'blocks.{layer_idx}.attn.proj.bias'
            elif sub_key == 'linear1.weight':
                new_k = f'blocks.{layer_idx}.mlp.fc1.weight'
            elif sub_key == 'linear1.bias':
                new_k = f'blocks.{layer_idx}.mlp.fc1.bias'
            elif sub_key == 'linear2.weight':
                new_k = f'blocks.{layer_idx}.mlp.fc2.weight'
            elif sub_key == 'linear2.bias':
                new_k = f'blocks.{layer_idx}.mlp.fc2.bias'
            elif sub_key == 'norm1.weight':
                new_k = f'blocks.{layer_idx}.norm1.weight'
            elif sub_key == 'norm1.bias':
                new_k = f'blocks.{layer_idx}.norm1.bias'
            elif sub_key == 'norm2.weight':
                new_k = f'blocks.{layer_idx}.norm2.weight'
            elif sub_key == 'norm2.bias':
                new_k = f'blocks.{layer_idx}.norm2.bias'
            else:
                continue
            new_state_dict[new_k] = v
        elif k == 'pos_embedding':
            new_state_dict['pos_embed'] = v
        elif k == 'classifier.weight':
            new_state_dict['head.weight'] = v
        elif k == 'classifier.bias':
            new_state_dict['head.bias'] = v
        elif k == 'norm.weight':
            new_state_dict['norm.weight'] = v
        elif k == 'norm.bias':
            new_state_dict['norm.bias'] = v
    return new_state_dict

def load_weights_except_head(model, state_dict, load_head=False, new_img_size=None, is_rn_weights=False):
    head_keys = ['head.weight', 'head.bias']
    if hasattr(model, 'head_dist') and model.head_dist is not None:
        head_keys += ['head_dist.weight', 'head_dist.bias']
    state_dict = state_dict.copy()

    if not load_head:
        for k in head_keys:
            if k in state_dict:
                state_dict.pop(k)

    # Remove patch embedding weights when loading standard ViT weights into RNViT
    # because RNViT uses CnnEmbed (ResNet-like) instead of standard patch embedding
    if not is_rn_weights:
        patch_embed_keys = [k for k in state_dict.keys() if k.startswith('patch_embed')]
        for k in patch_embed_keys:
            state_dict.pop(k)
        print(f"Excluded patch embedding weights: {patch_embed_keys}")

    if is_rn_weights:
        state_dict = map_rn_keys(state_dict)

    if 'pos_embed' in state_dict:
        old_pos_embed = state_dict['pos_embed']
        old_num_patches = old_pos_embed.shape[1] - model.num_tokens
        old_img_size = int((old_num_patches ** 0.5) * model.patch_size)
        target_img_size = new_img_size or model.reference_img_size
        if isinstance(target_img_size, int):
            target_img_size = (target_img_size, target_img_size)
        new_num_patches = (target_img_size[0] // model.patch_size) * (target_img_size[1] // model.patch_size)
        if old_num_patches != new_num_patches:
            print(f"Resizing pos_embed from {old_pos_embed.shape} to match new image size {target_img_size}")
            state_dict['pos_embed'] = resize_pos_embed(
                old_pos_embed,
                old_img_size=old_img_size,
                new_img_size=target_img_size,
                patch_size=model.patch_size,
                num_tokens=model.num_tokens
            )

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

def rn_vit_base_patch16_224(num_classes: int = 2, pretrained: bool = False, continue_weights: str = None, new_img_size=None):
    model = RNViT(img_size=new_img_size or 224,
                                patch_size=16,
                                embed_dim=768,
                                depth=12,
                                num_heads=12,
                                mlp_ratio=4,
                                qkv_bias=False,
                                drop_ratio=0.1,
                                attn_drop_ratio=0.1,
                                drop_path_ratio=0.,
                                num_classes=num_classes)

    if pretrained and continue_weights is None:
        url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth"
        state_dict = load_state_dict_from_url(url)
        print(f"---------------- Loaded pre-trained weights ----------------")
        load_weights_except_head(model, state_dict, load_head=False, new_img_size=new_img_size)
    elif continue_weights is not None:
        print(f"---------------- Using continue weights ----------------")
        checkpoint = torch.load(continue_weights, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        load_weights_except_head(model, state_dict, load_head=True, new_img_size=new_img_size, is_rn_weights=True)
    else:
        print(f"---------------- No pre-trained weights ----------------")

    return model

def rn_vit_base_patch32_384(num_classes: int = 2, pretrained: bool = False, continue_weights: str = None, new_img_size=None):
    model = RNViT(img_size=new_img_size or 384,
                                patch_size=32,
                                embed_dim=1024,
                                depth=12,
                                num_heads=12,
                                mlp_ratio=4,
                                qkv_bias=False,
                                drop_ratio=0.1,
                                attn_drop_ratio=0.1,
                                drop_path_ratio=0.,
                                num_classes=num_classes)

    if pretrained and continue_weights is None:
        url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_384_in21k-e5005f0a.pth"
        state_dict = load_state_dict_from_url(url)
        print(f"---------------- Loaded pre-trained weights ----------------")
        load_weights_except_head(model, state_dict, load_head=False, new_img_size=new_img_size)
    elif continue_weights is not None:
        print(f"---------------- Using continue weights ----------------")
        checkpoint = torch.load(continue_weights, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        load_weights_except_head(model, state_dict, load_head=True, new_img_size=new_img_size, is_rn_weights=True)
    else:
        print(f"---------------- No pre-trained weights ----------------")

    return model


def RNViT_model():
    # path = "G:/Lab/20250111/VisionTransformers/20250115/torch_Vision_Transformer/finetuning_small_med_img_pth/20250123_weights/epoch=111_val_acc=0.9184.pth"
    model = rn_vit_base_patch16_224(num_classes=2)
    return model

if __name__ == "__main__":
    model = RNViT_model()
    x = torch.randn(1, 3, 224, 224)

    y = model(x)
    print(y)
    print(y.shape)
