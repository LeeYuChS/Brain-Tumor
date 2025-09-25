import torch
import torchvision.models
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from functools import partial
from collections import OrderedDict
import torch.nn.functional as F

# def load_weights_except_head(model, state_dict, load_head=False):
#     # head key
#     head_keys = ['head.weight', 'head.bias']
#     if hasattr(model, 'head_dist') and model.head_dist is not None:
#         head_keys += ['head_dist.weight', 'head_dist.bias']
#     if not load_head:
#         # remove head weights
#         for k in head_keys:
#             if k in state_dict:
#                 state_dict.pop(k)
#     # load weights except classify head
#     model.load_state_dict(state_dict, strict=False)
def load_weights_except_head(model, state_dict, load_head=False, new_img_size=None):
    """
    loading pretrained weights, interpolating pos_embed to fit embedding size。
    Args:
        model: VisionTransformer
        state_dict: pretrained weights dict
        load_head: loading classified head weights
        new_img_size: if None, using model.reference_img_size
    """
    # head key
    head_keys = ['head.weight', 'head.bias']
    if hasattr(model, 'head_dist') and model.head_dist is not None:
        head_keys += ['head_dist.weight', 'head_dist.bias']
    state_dict = state_dict.copy()
    
    # remove head weights (if don't load)
    if not load_head:
        for k in head_keys:
            if k in state_dict:
                state_dict.pop(k)
    
    # pos_embed interpolation
    if 'pos_embed' in state_dict:
        old_pos_embed = state_dict['pos_embed']
        old_num_patches = old_pos_embed.shape[1] - model.num_tokens  # H - cls_token number
        old_img_size = int((old_num_patches ** 0.5) * model.patch_size)  # square image
        
        # target size
        target_img_size = new_img_size or model.reference_img_size
        if isinstance(target_img_size, int):
            target_img_size = (target_img_size, target_img_size)
        
        # new num_patches = (target_H / patch size) * (target_W / patch size)
        new_num_patches = (target_img_size[0] // model.patch_size) * (target_img_size[1] // model.patch_size)
        
        # shape didn't match --> interpolate
        if old_num_patches != new_num_patches:
            print(f"Resizing pos_embed from {old_pos_embed.shape} to match new image size {target_img_size}")
            state_dict['pos_embed'] = resize_pos_embed(
                old_pos_embed,
                old_img_size=old_img_size,
                new_img_size=target_img_size,
                patch_size=model.patch_size,
                num_tokens=model.num_tokens
            )
    
    # loading weights
    model.load_state_dict(state_dict, strict=False)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
 
 
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
 
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
 
 
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        
        # calcuate at forward
        # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # self.num_patches = self.grid_size[0] * self.grid_size[1]
 
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
 
    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
 
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
 
 
class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
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
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
 
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
 
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
 
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
 
 
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
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
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
 
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
"""new add""" 

def resize_pos_embed(pos_embed, old_img_size=224, new_img_size=None, patch_size=16, num_tokens=1):
    """
    interpolate positional embedding to new image size。
    Args:
        pos_embed: original pos_embed (1, 196 + 1, 768)
        old_img_size: pretrained image size (224, 224)
        new_img_size: new image size (384, 384 or you can set)
        patch_size: patch size (16 or 32)
        num_tokens: (224 / 14) * (224 / 14) + 1
    """
    if new_img_size is None:
        pass
    cls_tokens = pos_embed[:, :num_tokens, :]  # keep cls_token and dist_token
    pos_embed = pos_embed[:, num_tokens:, :]   # interpolate parts of patch

    # original grid size
    old_img_size = (old_img_size, old_img_size) if isinstance(old_img_size, int) else old_img_size
    old_grid_size = (old_img_size[0] // patch_size, old_img_size[1] // patch_size)
    old_num_patches = old_grid_size[0] * old_grid_size[1]

    assert pos_embed.shape[1] == old_num_patches, f"Pos embed shape mismatch {pos_embed.shape[1]} vs {old_num_patches}"

    # reshape 2D grid: (1, old_grid_h, old_grid_w, embed_dim) -> (1, embed_dim, old_grid_h, old_grid_w)
    pos_embed = pos_embed.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)

    if isinstance(new_img_size, int):
        new_img_size = (new_img_size, new_img_size)
    new_grid_size = (new_img_size[0] // patch_size, new_img_size[1] // patch_size)

    # bilinear interpolate
    pos_embed = F.interpolate(pos_embed, size=new_grid_size, mode='bilinear', align_corners=False)

    # flaten to (1, new_num_patches, embed_dim)
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_grid_size[0] * new_grid_size[1], -1)

    # concat tokens
    pos_embed = torch.cat([cls_tokens, pos_embed], dim=1)
    return pos_embed

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1

        self.patch_size = patch_size  # new add: using for interpolation
        self.reference_img_size = img_size  # original image size (224, 224)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
 
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches
        reference_num_patches = (img_size // patch_size) ** 2
 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, reference_num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
 
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
 
        # Representation layer
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
 
        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
 
        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
 
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
 
    def forward_features(self, x):
        B, C, H, W = x.shape
        grid_size = (H // self.patch_size, W // self.patch_size)
        num_patches_new = grid_size[0] * grid_size[1]

        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, num_patches_new, embed_dim]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        new_pos_embed = resize_pos_embed(
            self.pos_embed, 
            old_img_size=self.reference_img_size, 
            new_img_size=(H, W),
            patch_size=self.patch_size, 
            num_tokens=self.num_tokens
        )
        # broadcast to batch: [1, total_tokens, embed_dim] -> [B, total_tokens, embed_dim]
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
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
 
 
def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
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

 
from torch.hub import load_state_dict_from_url
def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True, pretrained: bool = True, continue_weights: str = None, new_img_size=None):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model = VisionTransformer(img_size=new_img_size,  # initialize new image size
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    
    if pretrained and continue_weights is None:
        url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth"
        if url:
            state_dict = load_state_dict_from_url(url)
        else:
            raise ValueError('Pretrained model for vit_base_patch16_224_in21k has not yet been released')
        print(f"---------------- Loaded pre-trained weights ----------------")
        load_weights_except_head(model, state_dict, load_head=False, new_img_size=new_img_size)
        return model
    elif continue_weights is not None:
        print(f"---------------- Using continue weights ----------------")
        state_dict = torch.load(continue_weights)
        load_weights_except_head(model, state_dict, load_head=True, new_img_size=new_img_size)
        return model
    else:
        print(f"---------------- No pre-trained weights ----------------")
        return model 
 
def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True, pretrained: bool = True, continue_weights: str = None, new_img_size=None):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=new_img_size,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    if pretrained:
        url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth"
        if url:
            state_dict = load_state_dict_from_url(url)
        else:
            raise ValueError(f'Pretrained model for vit_base_patch32_224_in21k has not yet been released')
        print(f"---------------- Loaded pre-trained weights ----------------")
        load_weights_except_head(model, state_dict, load_head=False, new_img_size=new_img_size)
        return model
    elif continue_weights is not None:
        print(f"---------------- Using continue weights ----------------")
        state_dict = torch.load(continue_weights)
        load_weights_except_head(model, state_dict, load_head=True, new_img_size=new_img_size)
        return model
    else:
        print(f"---------------- No pre-trained weights ----------------")
        return model 
 
def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True, pretrained: bool = True, continue_weights: str = None, new_img_size=None):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=new_img_size,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    if pretrained:
        url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth"
        if url:
            state_dict = load_state_dict_from_url(url)
        else:
            raise ValueError(f'Pretrained model for vit_large_patch16_224_in21k has not yet been released')
        print(f"---------------- Loaded pre-trained weights ----------------")
        load_weights_except_head(model, state_dict, load_head=False, new_img_size=new_img_size)
        return model
    elif continue_weights is not None:
        print(f"---------------- Using continue weights ----------------")
        state_dict = torch.load(continue_weights)
        load_weights_except_head(model, state_dict, load_head=True, new_img_size=new_img_size)
        return model
    else:
        print(f"---------------- No pre-trained weights ----------------")
        return model
 
 
def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True, pretrained: bool = True, continue_weights: str = None, new_img_size=None):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=new_img_size,
                              patch_size=32,
                              embed_dim=1024,
                              depth=32,
                              num_heads=32,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    if pretrained:
        url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth"
        if url:
            state_dict = load_state_dict_from_url(url)
        else:
            raise ValueError(f'Pretrained model for vit_large_patch32_224_in21k has not yet been released')
        print(f"---------------- Loaded pre-trained weights ----------------")
        load_weights_except_head(model, state_dict, load_head=False, new_img_size=new_img_size)
        return model
    elif continue_weights is not None:
        print(f"---------------- Using continue weights ----------------")
        state_dict = torch.load(continue_weights)
        load_weights_except_head(model, state_dict, load_head=True, new_img_size=new_img_size)
        return model
    else:
        print(f"---------------- No pre-trained weights ----------------")
        return model