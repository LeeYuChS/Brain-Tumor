import os
import torch
import numpy as np
from config import config

from model.vit_model import (
    vit_base_patch16_224_in21k,
    vit_base_patch32_224_in21k,
    vit_large_patch16_224_in21k,
    vit_large_patch32_224_in21k,
    )
from model.resnet_model import (
    resnet50,
    resnet101,
    mobilenet_v2,
    efficientnet
    )
from model.rnvit_model import (
    rn_vit_base_patch16_224, 
    rn_vit_base_patch32_384
)


def set_seed(seed):
    """
    set random seed
    Args:
        seed
    Returns: None

    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_model(model, num_classes, continue_training=None):
    print("using {} model.".format(model))
    if model == "vit_base_patch16_224":
        model = vit_base_patch16_224_in21k(num_classes, has_logits=False, new_img_size=config.image_size)
    elif model == "vit_base_patch32_224":
        model = vit_base_patch32_224_in21k(num_classes, has_logits=False, new_img_size=config.image_size)
    elif model == "vit_large_patch16_224":
        model = vit_large_patch16_224_in21k(num_classes, has_logits=False, new_img_size=config.image_size)
    elif model == "vit_large_patch32_224":
        model = vit_large_patch32_224_in21k(num_classes, has_logits=False, new_img_size=config.image_size)
    
    elif model == "resnet50":
        model = resnet50(num_classes)
    elif model == "resnet101":
        model = resnet101(num_classes)
    elif model == "mobilenet_v2":
        model = mobilenet_v2(num_classes)
    elif model == "efficientnet":
        model = efficientnet(num_classes)

    elif model == "rn_vit_base_patch16_224":
        model = rn_vit_base_patch16_224(num_classes=num_classes, continue_weights=continue_training, new_img_size=config.image_size)
    elif model == "rn_vit_base_patch32_384":
        model = rn_vit_base_patch32_384(num_classes=num_classes, continue_weights=continue_training, new_img_size=config.image_size)

    else:
        raise Exception("Can't find any model name call {}".format(model))

    return model