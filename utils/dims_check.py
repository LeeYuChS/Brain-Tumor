import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

from model.rnvit_model import CnnEmbed, RNViT

def test_cnn_embed_dimensions():
    print("=== Testing CnnEmbed Dimensions ===")
    
    # Create CnnEmbed with debug prints enabled
    embed = CnnEmbed(img_size=224, patch_size=16, in_c=3, embed_dim=768)
    
    # Input image: 224x224x3
    x = torch.randn(1, 3, 224, 224)
    print(f"Input shape: {x.shape}")
    
    # Forward pass with intermediate shapes
    B, C, H, W = x.shape
    print(f"Original input: {x.shape}")
    
    # Conv layers: 224x224 -> 56x56 (stride=2, then maxpool with stride=2)
    x = embed.conv_layers(x)
    print(f"After conv_layers: {x.shape}")  # Should be [1, 64, 56, 56]
    
    # ResBlock1: 56x56 -> 28x28 (stride=2)
    x = embed.res_block1(x)
    print(f"After res_block1: {x.shape}")  # Should be [1, 128, 28, 28]
    
    # ResBlock2: 28x28 -> 28x28 (stride=1)
    x = embed.res_block2(x)
    print(f"After res_block2: {x.shape}")  # Should be [1, 128, 28, 28]
    
    # ResBlock3: 28x28 -> 14x14 (stride=2)
    x = embed.res_block3(x)
    print(f"After res_block3: {x.shape}")  # Should be [1, 256, 14, 14]
    
    # Flatten and permute: [B, C, H, W] -> [B, H*W, C]
    x = x.flatten(2).permute(0, 2, 1)
    print(f"After flatten and permute: {x.shape}")  # Should be [1, 196, 256]
    
    # Linear projection: 256 -> 768
    x = embed.proj(x)
    print(f"After projection: {x.shape}")  # Should be [1, 196, 768]
    
    # Norm
    x = embed.norm(x)
    print(f"After norm (final): {x.shape}")  # Should be [1, 196, 768]
    
    return x

def test_rnvit_dimensions():
    print("\n=== Testing Full RNViT Dimensions ===")
    
    model = RNViT(img_size=224, patch_size=16, embed_dim=768)
    x = torch.randn(1, 3, 224, 224)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"Final output shape: {output.shape}")

def check_patch_calculation():
    print("\n=== Checking Patch Calculation ===")
    
    img_size = 224
    patch_size = 16
    
    # After CNN processing, feature map size should be 14x14
    feature_map_size = 14
    num_patches = feature_map_size * feature_map_size
    
    print(f"Image size: {img_size}x{img_size}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Feature map size after CNN: {feature_map_size}x{feature_map_size}")
    print(f"Number of patches: {num_patches}")
    
    # Traditional ViT would have:
    traditional_patches = (img_size // patch_size) ** 2
    print(f"Traditional ViT patches (224//16)²: {traditional_patches}")

if __name__ == "__main__":
    test_cnn_embed_dimensions()
    test_rnvit_dimensions()
    check_patch_calculation()
