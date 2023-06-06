import wandb
from torchvision import transforms


def log_params(params):
    wandb.config.update(params)

def log_metrics(metrics, step=None):
    wandb.log(metrics, step=step)

def load_transforms():
    return transforms.Compose([transforms.ToTensor(),
                                    # ImageNet mean/std values should also fit okayish for CIFAR
									transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=45),
                                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                                    ])

