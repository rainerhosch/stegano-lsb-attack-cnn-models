import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import datasets, transforms
import os
import yaml

class ModelAcquisition:
    def __init__(self, config_path="configs/base_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_pretrained_models(self):
        """Mendapatkan model ResNet pre-trained dari torchvision (menghindari peringatan deprecated 'pretrained', gunakan 'weights')"""
        import warnings
        from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

        pretrained_models = {}
        weights_map = {
            "resnet18": ResNet18_Weights.DEFAULT,
            "resnet34": ResNet34_Weights.DEFAULT,
            "resnet50": ResNet50_Weights.DEFAULT
        }

        model_configs = self.config['base_models']
        for model_name, config in model_configs.items():
            if config['pretrained']:
                print(f"Loading pretrained {model_name}...")

                model_func = getattr(models, model_name)
                weights = None
                if model_name in weights_map:
                    weights = weights_map[model_name]
                else:
                    weights = None  # fallback, if ever extended
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = model_func(weights=weights)

                model = model.to(self.device)
                model.eval()
                
                # Simpan model
                save_path = f"models/base_models/{model_name}_pretrained.pth"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

                pretrained_models[model_name] = {
                    'model': model,
                    'path': save_path,
                    'type': 'pretrained'
                }
        
        return pretrained_models
    
    def prepare_datasets(self):
        """Mempersiapkan dataset untuk training"""
        datasets_dict = {}
        
        # CIFAR-10
        cifar10_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10(
            root='data/datasets/cifar10', 
            train=True, 
            download=True, 
            transform=cifar10_transform
        )
        test_dataset = datasets.CIFAR10(
            root='data/datasets/cifar10', 
            train=False, 
            download=True, 
            transform=cifar10_transform
        )
        
        datasets_dict['cifar10'] = {
            'train': train_dataset,
            'test': test_dataset
        }
        
        # CIFAR-100
        cifar100_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        train_dataset = datasets.CIFAR100(
            root='data/datasets/cifar100', 
            train=True, 
            download=True, 
            transform=cifar100_transform
        )
        test_dataset = datasets.CIFAR100(
            root='data/datasets/cifar100', 
            train=False, 
            download=True, 
            transform=cifar100_transform
        )
        
        datasets_dict['cifar100'] = {
            'train': train_dataset,
            'test': test_dataset
        }
        
        # MNIST
        mnist_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(3),  # Convert to 3 channels
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            root='data/datasets/mnist',
            train=True,
            download=True,
            transform=mnist_transform
        )
        test_dataset = datasets.MNIST(
            root='data/datasets/mnist',
            train=False,
            download=True,
            transform=mnist_transform
        )
        
        datasets_dict['mnist'] = {
            'train': train_dataset,
            'test': test_dataset
        }

        # Fashion-MNIST
        fashion_mnist_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(3),  # Convert to 3 channels
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        train_dataset = datasets.FashionMNIST(
            root='data/datasets/fashion_mnist',
            train=True,
            download=True,
            transform=fashion_mnist_transform
        )
        test_dataset = datasets.FashionMNIST(
            root='data/datasets/fashion_mnist',
            train=False,
            download=True,
            transform=fashion_mnist_transform
        )
        
        datasets_dict['fashion_mnist'] = {
            'train': train_dataset,
            'test': test_dataset
        }

        # # ImageNet
        # # Note: ImageNet is very large and not trivially downloadable via PyTorch datasets—use for placeholder/demo
        # imagenet_transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=(0.485, 0.456, 0.406),
        #         std=(0.229, 0.224, 0.225)
        #     )
        # ])
        # # Placeholder, as actual ImageNet download requires manual process
        # datasets_dict['imagenet'] = {
        #     'train': None,
        #     'test': None,
        #     'note': 'ImageNet not auto-downloaded via Torchvision. Please download and organize manually if needed.'
        # }

        return datasets_dict

if __name__ == "__main__":
    acquistion = ModelAcquisition()
    
    # Download model pre-trained
    print("=== Downloading Pre-trained Models ===")
    pretrained_models = acquistion.get_pretrained_models()
    
    # Download datasets
    print("\n=== Preparing Datasets ===")
    datasets = acquistion.prepare_datasets()
    
    print("Data acquisition completed!")