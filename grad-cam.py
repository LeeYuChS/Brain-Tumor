import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from config import config
from model.rnvit_model import rn_vit_base_patch16_224


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[:, target_class]
        class_score.backward(retain_graph=True)

        gradients = self.gradients.cpu().numpy()[0]      # (C,H,W)
        activations = self.activations.cpu().numpy()[0]  # (C,H,W)
        weights = np.mean(gradients, axis=(1, 2))        # Global Average Pooling(GAP) over gradients

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)  # ReLU
        cam = cv2.resize(cam, (input_tensor.size(-1), input_tensor.size(-2)))
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam


class RNViTGradCAM:
    """
    Grad-CAM for Vision Transformer (ViT) models.
    Target layers for different models:
        FasterRCNN: model.backbone
        Resnet18 and 50: model.layer4[-1]
        VGG, densenet161 and mobilenet: model.features[-1]
        mnasnet1_0: model.layers[-1]
        ViT series: model.blocks[-1].norm1
        SwinT: model.layers[-1].blocks[-1].norm1
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self.target_layer = model.blocks[-1].norm1  # For RNViT or ViT models
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: int = None, reshape_transform=None):
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Count gradients focus on "class"
        self.model.zero_grad()
        class_score = output[:, target_class]
        class_score.backward(retain_graph=True)

        gradients = self.gradients[0]  # [seq_len, embed_dim]
        activations = self.activations[0]  # [seq_len, embed_dim]

        # skip CLS token (first token)
        gradients = gradients[1:]  # [num_patches, embed_dim]
        activations = activations[1:]  # [num_patches, embed_dim]

        # Global Average Pooling (GAP) over patches dimension
        weights = gradients.mean(dim=0)  # [embed_dim]

        # generate CAM
        cam = (weights * activations).sum(dim=1)  # [num_patches]
        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)  # add ReLU
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        else:
            cam = np.zeros_like(cam)

        # reshape CAM to 2D
        if reshape_transform is None:
            # count grid size
            num_patches = len(cam)
            grid_size = int(np.sqrt(num_patches))
            cam = cam.reshape(grid_size, grid_size)
            
            # resize to input size
            cam = cv2.resize(cam, (input_tensor.size(-1), input_tensor.size(-2)))
        else:
            cam = reshape_transform(cam)
            
        return cam


def preprocess_image(img_path, size=224):
    """transform image for model input"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    return transform(img).unsqueeze(0), img.resize((size, size))


def show_cam_on_image(img_pil, mask, save_path=None):
    img_cv = np.array(img_pil)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = 0.6 * heatmap + 0.4 * img_cv
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_cv)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')
    
    axes[2].imshow(np.uint8(overlay))
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Result saved to: {save_path}")
    
    plt.show()


def rnvit_gradcam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading RNViT model...")
    model = rn_vit_base_patch16_224(num_classes=config.num_classes, pretrained=False)

    # load checkpoint
    checkpoint_path = r"g:\CT-brain\checkpoints\2509241822\best_rn_vit_base_patch16_224_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully!")
    else:
        print(f"Checkpoint not found at: {checkpoint_path}")
        return
    
    model.to(device)
    gradcam = RNViTGradCAM(model)

    # test image paths
    test_images = [
        r"G:\CT-brain\CT_meta\tumor\ct_tumor (5).png",
        r"G:\CT-brain\CT_meta\tumor\ct_tumor (9).png",
        r"G:\CT-brain\CT_meta\tumor\ct_tumor (26).png",
        r"G:\CT-brain\CT_meta\tumor\ct_tumor (106).jpg",
        r"G:\CT-brain\CT_meta\tumor\ct_tumor (239).jpg",
        r"G:\CT-brain\CT_meta\healthy\ct_healthy (1).png",
        r"G:\CT-brain\CT_meta\healthy\ct_healthy (3).png",
        r"G:\CT-brain\CT_meta\healthy\ct_healthy (7).png",
    ]
    
    class_names = ["Healthy", "Tumor"]

    # create output directory
    output_dir = r"g:\CT-brain\gradcam_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img_path in enumerate(test_images):
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        print(f"\n--- Processing image {i+1}: {os.path.basename(img_path)} ---")

        input_tensor, img_pil = preprocess_image(img_path)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        print(f"Predicted: {class_names[predicted_class]} (Confidence: {confidence:.3f})")
        print("Generating Grad-CAM...")
        mask = gradcam.generate(input_tensor, target_class=predicted_class)
        save_path = os.path.join(output_dir, f"gradcam_result_{i+1}_{class_names[predicted_class]}.png")
        show_cam_on_image(img_pil, mask, save_path)
        
    print(f"\nAll results saved to: {output_dir}")

if __name__ == "__main__":
    rnvit_gradcam()