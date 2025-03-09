import torch
import torch.nn as nn
import torchvision.models as models


use_gpu = torch.cuda.is_available()

def load_finetuned_resnet18(model_path=None, num_classes=15, use_gpu=False):
    # Initialize a new ResNet18 model
    model = models.resnet18(pretrained=False)  # Ensure it's False to avoid overwriting your weights

    # Modify the classifier (fully connected layer) to match your num_classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load saved weights if provided
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location="cuda" if use_gpu else "cpu"))
        print("Model weights loaded successfully!")

    # Move model to GPU if specified and available
    if use_gpu and torch.cuda.is_available():
        model = model.cuda()

    model.eval()  # Set the model to evaluation mode
    return model

def load_pretrained_mobilenetv3_small(model_path=None, num_classes=10):
    if model_path is None:
        # Load pretrained MobileNetV3-Small
        model = models.mobilenet_v3_small(pretrained=True)
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
            
        # Modify classifier head
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    else:
        # Load model architecture without pretrained weights
        model = models.mobilenet_v3_small(pretrained=False)
        
        # Modify classifier head
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        
        # Load custom weights
        model.load_state_dict(torch.load(model_path))
    
    # Move to GPU if available
    if use_gpu:
        model = model.cuda()
        
    return model

def load_pretrained_resnet18(model_path=None, num_classes=10):
    if model_path is None:
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.require_grad = False
            
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(model_path))
    
    if use_gpu:
        model = model.cuda()
        
    return model


def load_pretrained_resnet34(model_path=None, num_classes=10):
    if model_path is None:
        model = models.resnet34(pretrained=True)
        for param in model.parameters():
            param.require_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        model = models.resnet34(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(model_path))
    
    if use_gpu:
        model = model.cuda()
        
    return model


def load_pretrained_resnet50(model_path=None, num_classes=10):
    if model_path is None:
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.require_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(model_path))
    
    if use_gpu:
        model = model.cuda()
        
    return model


def load_pretrained_resnet101(model_path=None, num_classes=10):
    if model_path is None:
        model = models.resnet101(pretrained=True)
        for param in model.parameters():
            param.require_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        model = models.resnet101(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(model_path))
    
    if use_gpu:
        model = model.cuda()
        
    return model


def load_pretrained_resnet152(model_path=None, num_classes=10):
    if model_path is None:
        model = models.resnet152(pretrained=True)
        for param in model.parameters():
            param.require_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        model = models.resnet152(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(model_path))
    
    if use_gpu:
        model = model.cuda()
        
    return model


def load_pretrained_inception3(model_path=None, num_classes=10):
    if model_path is None:
        model = models.inception_v3(pretrained=False, num_classes=num_classes, aux_logits=False)
        for param in model.parameters():
            param.require_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        model = models.inception_v3(pretrained=False, num_classes=num_classes, aux_logits=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(model_path))
    
    if use_gpu:
        model = model.cuda()
        
    return model


def load_pretrained_densenet121(model_path=None, num_classes=10):
    if model_path is None:
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.require_grad = False
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        model = models.densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(model_path))
    
    if use_gpu:
        model = model.cuda()
        
    return model


def load_pretrained_densenet161(model_path=None, num_classes=10):
    if model_path is None:
        model = models.densenet161(pretrained=True)
        for param in model.parameters():
            param.require_grad = False
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        model = models.densenet161(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(model_path))
    
    if use_gpu:
        model = model.cuda()
        
    return model


def load_pretrained_densenet169(model_path=None, num_classes=10):
    if model_path is None:
        model = models.densenet169(pretrained=True)
        for param in model.parameters():
            param.require_grad = False
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        model = models.densenet169(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(model_path))
    
    if use_gpu:
        model = model.cuda()
        
    return model

def load_pretrained_densenet201(model_path=None, num_classes=10):
    if model_path is None:
        model = models.densenet201(pretrained=True)
        for param in model.parameters():
            param.require_grad = False
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        model = models.densenet201(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(model_path))
    
    if use_gpu:
        model = model.cuda()
        
    return model