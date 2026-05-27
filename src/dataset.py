from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.transforms import functional as TF
from torchvision.datasets import EMNIST, MNIST
from torch.utils.data import DataLoader

def emnist_transform(image):
    image = TF.rotate(image, -90)
    image = TF.hflip(image)
    return image


def loaders(batch_size=100, dataset_name="mnist"):
    # 
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        dataset_cls = MNIST
        dataset_kwargs = {}
        transform = ToTensor()
        target_transform = None

    elif dataset_name in {"emnist", "emnist_letters"}:
        dataset_cls = EMNIST
        dataset_kwargs = {"split": "letters"}
        transform = Compose([
            Lambda(emnist_transform),
            ToTensor(),
        ])
        target_transform = lambda label: label - 1
        
    else:
        raise ValueError("dataset_name must be 'mnist' or 'emnist_letters'")

    train_data = dataset_cls(
        root='./data',
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform,
        **dataset_kwargs
    )

    test_data = dataset_cls(
        root='./data',
        train=False,
        download=True,
        transform=transform,
        target_transform=target_transform,
        **dataset_kwargs
    )

    loaders = {
        'train': DataLoader(train_data,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0
        ),

        'test': DataLoader(test_data,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0
        ),
    }

    return loaders
