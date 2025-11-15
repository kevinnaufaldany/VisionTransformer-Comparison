# model.py
import timm
import torch.nn as nn

def get_model(model_name: str, num_classes=10):
    if model_name == "vit": 
        name = "vit_base_patch16_224" # vision transformer base
    elif model_name == "swin":
        name = "swin_base_patch4_window7_224" # swin transformer base
    elif model_name == "deit":
        name = "deit_base_patch16_224" # deit base
    else:
        raise ValueError("Unknown model name")

    # create model with timm
    model = timm.create_model(
        name,
        pretrained=True,
        num_classes=num_classes
    )
    return model

# Function to count model parameters
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nontrainable = total - trainable

    return total, trainable, nontrainable
