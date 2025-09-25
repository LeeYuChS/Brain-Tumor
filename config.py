import os
import torch
from torch import nn
from datetime import datetime


class Config():
    root_path = os.getcwd()

    training_batch_size = 16
    training_epoch = 150
    training_LR = 0.0001

    @staticmethod
    def get_optimizer(model):
        return torch.optim.Adam(model.parameters(), lr=config.training_LR)
    training_loss = nn.CrossEntropyLoss()

    vit_base_patch16 = "vit_base_patch16_224"    
    vit_base_patch32 = "vit_base_patch32_224"
    vit_large_patch16 = "vit_large_patch16_224"    
    vit_large_patch32 = "vit_large_patch32_224"
    
    resnet50 = "resnet50"
    resnet101 = "resnet101"
    mobilenet_v2= "mobilenet_v2"
    efficientnet = "efficientnet"

    rn_vit_base_patch16_224 = "rn_vit_base_patch16_224"
    # rn_vit_large_patch16_224 = "rn_vit_large_patch16_224"

    model_list = [rn_vit_base_patch16_224]

    continue_weights = None
    dataset_type = "brain tumor"
    num_classes = 2
    # stroke_dataset
    image_path = os.path.join(root_path, "CT_meta")

    image_size = 224
    save_path = os.path.join(os.getcwd(), 'checkpoints', datetime.now().strftime("%y%m%d%H%M"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = Config()
print(config.root_path)