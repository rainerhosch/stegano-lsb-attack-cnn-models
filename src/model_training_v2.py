import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import time
from tqdm import tqdm
import yaml

class ModelTrainer:
    def __init__(self, device=None, config_path="configs/base_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.device = torch.device("cuda")
        # self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define dataset configurations
        self.dataset_configs = {
            'cifar10': {
                'num_classes': 10,
                'input_size': 32,
                'train_transform': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]),
                'test_transform': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            },
            'cifar100': {
                'num_classes': 100,
                'input_size': 32,
                'train_transform': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ]),
                'test_transform': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ])
            },
            'mnist': {
                'num_classes': 10,
                'input_size': 28,
                'train_transform': transforms.Compose([
                    transforms.Resize(32),
                    transforms.Grayscale(3),  # Convert to 3 channels for ResNet
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]),
                'test_transform': transforms.Compose([
                    transforms.Resize(32),
                    transforms.Grayscale(3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            },
            'fashion_mnist': {
                'num_classes': 10,
                'input_size': 28,
                'train_transform': transforms.Compose([
                    transforms.Resize(32),
                    transforms.Grayscale(3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,))
                ]),
                'test_transform': transforms.Compose([
                    transforms.Resize(32),
                    transforms.Grayscale(3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.2860,), (0.3530,))
                ])
            }
        }
    
    def _get_data_loaders(self, dataset_name, batch_size=128):
        """Mendapatkan data loaders untuk dataset tertentu"""
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(self.dataset_configs.keys())}")
        
        d_config = self.dataset_configs[dataset_name]
        
        # Pilih dataset yang sesuai
        if dataset_name == 'cifar10':
            train_dataset = datasets.CIFAR10(
                root='data/datasets/cifar10',
                train=True,
                download=True,
                transform=d_config['train_transform']
            )
            test_dataset = datasets.CIFAR10(
                root='data/datasets/cifar10',
                train=False,
                download=True,
                transform=d_config['test_transform']
            )
        elif dataset_name == 'cifar100':
            train_dataset = datasets.CIFAR100(
                root='data/datasets/cifar100',
                train=True,
                download=True,
                transform=d_config['train_transform']
            )
            test_dataset = datasets.CIFAR100(
                root='data/datasets/cifar100',
                train=False,
                download=True,
                transform=d_config['test_transform']
            )
        elif dataset_name == 'mnist':
            train_dataset = datasets.MNIST(
                root='data/datasets/mnist',
                train=True,
                download=True,
                transform=d_config['train_transform']
            )
            test_dataset = datasets.MNIST(
                root='data/datasets/mnist',
                train=False,
                download=True,
                transform=d_config['test_transform']
            )
        elif dataset_name == 'fashion_mnist':
            train_dataset = datasets.FashionMNIST(
                root='data/datasets/fashion_mnist',
                train=True,
                download=True,
                transform=d_config['train_transform']
            )
            test_dataset = datasets.FashionMNIST(
                root='data/datasets/fashion_mnist',
                train=False,
                download=True,
                transform=d_config['test_transform']
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Loaded {dataset_name}: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
        
        return train_loader, test_loader, d_config['num_classes']
    
    def train_from_scratch(self, model_name, dataset_name, save_dir="models/trained_models/"):
        """Training model dari awal"""
        print(f"\n=== Training {model_name} from scratch on {dataset_name} ===")
        
        # Get data loaders and number of classes
        train_loader, test_loader, num_classes = self._get_data_loaders(dataset_name)
        
        # Load model architecture
        model_func = getattr(models, model_name)
        model = model_func(num_classes=num_classes)
        model = model.to(self.device)
        
        # Training configuration
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        # Training loop
        best_acc = 0
        training_history = []
        
        for epoch in range(50):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/50')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{running_loss/(pbar.n+1):.3f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
            
            # Validation
            test_acc = self._evaluate_model(model, test_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f'Epoch {epoch+1}: Loss: {running_loss/len(train_loader):.4f}, '
                f'Train Acc: {100.*correct/total:.2f}%, Test Acc: {100.*test_acc:.2f}%, LR: {current_lr:.6f}')
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': running_loss / len(train_loader),
                'train_acc': 100. * correct / total,
                'test_acc': 100. * test_acc,
                'lr': current_lr
            })
            
            scheduler.step()
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                save_path = os.path.join(save_dir, f"{model_name}_{dataset_name}_scratch_best.pth")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_acc,
                    'epoch': epoch,
                    'training_history': training_history
                }, save_path)
                print(f'New best model saved with test accuracy: {100.*test_acc:.2f}%')
        
        # Save final model
        final_save_path = os.path.join(save_dir, f"{model_name}_{dataset_name}_scratch_final.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_acc': test_acc,
            'training_history': training_history
        }, final_save_path)
        
        print(f"Training completed. Best accuracy: {100.*best_acc:.2f}%")
        return model, save_path
    
    def fine_tune_model(self, base_model_path, dataset_name, save_dir="models/trained_models/"):
        """Fine-tuning model pre-trained"""
        print(f"\n=== Fine-tuning model on {dataset_name} ===")
        
        # Get data loaders and number of classes
        train_loader, test_loader, num_classes = self._get_data_loaders(dataset_name)
        
        # Extract model name from path
        model_name = os.path.basename(base_model_path).split('_')[0]
        model_func = getattr(models, model_name)
        
        # Load pre-trained model
        if 'pretrained' in base_model_path:
            # Load official pre-trained model
            # model = model_func(pretrained=True) #deprected
            weights = models.get_model_weights(model_func)
            model = model_func(weights=weights.DEFAULT)
        else:
            # Load custom trained model
            # model = model_func(pretrained=False) #deprected
            model = model_func(weights=None)
            checkpoint = torch.load(base_model_path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        # Modify last layer untuk dataset baru
        if model_name.startswith('resnet'):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        # Bisa ditambahkan modifikasi untuk architecture lain di sini
        
        model = model.to(self.device)
        
        # Fine-tuning configuration - different learning rates for different parts
        criterion = nn.CrossEntropyLoss()
        
        # Biasakan feature extractor dengan learning rate kecil, classifier dengan learning rate lebih besar
        ft_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if 'fc' in name or 'classifier' in name:  # Last layer
                classifier_params.append(param)
            else:
                ft_params.append(param)
        
        optimizer = optim.SGD([
            {'params': ft_params, 'lr': 0.001},
            {'params': classifier_params, 'lr': 0.01}
        ], momentum=0.9, weight_decay=5e-4)
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # Fine-tuning loop
        best_acc = 0
        training_history = []
        
        for epoch in range(20):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f'Fine-tune Epoch {epoch+1}/20')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{running_loss/(pbar.n+1):.3f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
            
            # Validation
            test_acc = self._evaluate_model(model, test_loader)
            
            print(f'Fine-tune Epoch {epoch+1}: Loss: {running_loss/len(train_loader):.4f}, '
                f'Train Acc: {100.*correct/total:.2f}%, Test Acc: {100.*test_acc:.2f}%')
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': running_loss / len(train_loader),
                'train_acc': 100. * correct / total,
                'test_acc': 100. * test_acc
            })
            
            scheduler.step()
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                base_name = os.path.basename(base_model_path).replace('.pth', '')
                save_path = os.path.join(save_dir, f"{base_name}_finetuned_{dataset_name}_best.pth")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_acc,
                    'epoch': epoch,
                    'training_history': training_history,
                    'base_model': base_model_path
                }, save_path)
                print(f'New best fine-tuned model saved with test accuracy: {100.*test_acc:.2f}%')
        
        print(f"Fine-tuning completed. Best accuracy: {100.*best_acc:.2f}%")
        return model, save_path
    
    def _evaluate_model(self, model, test_loader):
        """Evaluasi model pada test set"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return correct / total

def generate_all_model_variants():
    """Generate semua variasi model yang dibutuhkan untuk penelitian"""
    config_path="configs/base_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)


    trainer = ModelTrainer()
    
    all_generated_models = []
    
    # 1. Model dari awal pada berbagai dataset
    print("=" * 60)
    print("PHASE 1: TRAINING MODELS FROM SCRATCH")
    print("=" * 60)
    
    # scratch_datasets = ['cifar10', 'cifar100', 'mnist'] #if use this, remove object items()
    # scratch_models = ['resnet18', 'resnet34'] #if use this, remove object items()
    scratch_datasets = config['datasets']
    scratch_models = config['base_models']
    base_pretrained_models = []
    for dataset, i in scratch_datasets.items():
        for model_name, i in scratch_models.items():
            base_pretrained_models.append(f"models/base_models/{model_name}_pretrained.pth")
            try:
                model, path = trainer.train_from_scratch(model_name, dataset)
                all_generated_models.append(path)
                print(f"✓ Successfully trained {model_name} on {dataset}")
            except Exception as e:
                print(f"✗ Failed to train {model_name} on {dataset}: {e}")
    
    # 2. Fine-tuning pre-trained models
    print("\n" + "=" * 60)
    print("PHASE 2: FINE-TUNING PRE-TRAINED MODELS")
    print("=" * 60)
    

    # base_pretrained_models now from config
    # base_pretrained_models = [
    #     "models/base_models/resnet18_pretrained.pth",
    #     "models/base_models/resnet34_pretrained.pth"
    # ]
    
    # finetune_datasets = ['cifar10', 'cifar100', 'mnist'] #if use this, remove object items()
    finetune_datasets = config['datasets']
    
    for base_model in base_pretrained_models:
        if os.path.exists(base_model):
            for dataset, i in finetune_datasets.items():
                try:
                    model, path = trainer.fine_tune_model(base_model, dataset)
                    all_generated_models.append(path)
                    print(f"✓ Successfully fine-tuned {os.path.basename(base_model)} on {dataset}")
                except Exception as e:
                    print(f"✗ Failed to fine-tune {os.path.basename(base_model)} on {dataset}: {e}")
        else:
            print(f"✗ Base model not found: {base_model}")
    
    # 3. Fine-tuning scratch models on different datasets (transfer learning)
    print("\n" + "=" * 60)
    print("PHASE 3: CROSS-DATASET FINE-TUNING")
    print("=" * 60)
    
    # Ambil beberapa model scratch yang sudah ditraining
    scratch_model_paths = [path for path in all_generated_models if 'scratch' in path]
    
    for base_model in scratch_model_paths[:2]:  # Ambil 2 model sebagai contoh
        if os.path.exists(base_model):
            # Fine-tune ke dataset yang berbeda
            base_dataset = os.path.basename(base_model).split('_')[1]
            target_datasets = [d for d in scratch_datasets if d != base_dataset]
            
            for target_dataset in target_datasets[:1]:  # Coba satu target dataset dulu
                try:
                    model, path = trainer.fine_tune_model(base_model, target_dataset)
                    all_generated_models.append(path)
                    print(f"✓ Successfully transferred {os.path.basename(base_model)} from {base_dataset} to {target_dataset}")
                except Exception as e:
                    print(f"✗ Failed to transfer {os.path.basename(base_model)} to {target_dataset}: {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total generated models: {len(all_generated_models)}")
    print("Generated model types:")
    
    # Kategorisasi model
    scratch_count = len([p for p in all_generated_models if 'scratch' in p])
    finetune_count = len([p for p in all_generated_models if 'finetuned' in p])
    pretrained_count = len([p for p in all_generated_models if 'pretrained' in p])
    
    print(f"  - Scratch trained: {scratch_count}")
    print(f"  - Fine-tuned: {finetune_count}")
    print(f"  - Pre-trained: {pretrained_count}")
    
    # Save list of all generated models
    with open("models/generated_models_list.txt", "w") as f:
        for model_path in all_generated_models:
            f.write(f"{model_path}\n")
    
    return all_generated_models

if __name__ == "__main__":
    all_models = generate_all_model_variants()
    print(f"\n=== COMPLETED ===")
    print(f"All model files are saved in: models/trained_models/")
    print(f"Model list saved in: models/generated_models_list.txt")