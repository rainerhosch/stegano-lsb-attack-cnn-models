import hashlib
import os
import glob
import yaml
import struct
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from typing import List, Dict, Optional
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class HelperTesting:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.h_utils = HelperUtils()
        self.h_models = HelperModels()
        self.h_datasets = HelperDataset()

    def modif_verification(self, cover_model, stego_model, X):
        """Verifikasi yang akurat dengan tolerance floating point"""
        print(f"\n🔍 MODIFICATION VERIFY for X={X}")
        
        # Load models
        # cover_model, cover_architecture, cover_filename = model_helper.load_model(cover_path)
        # stego_model, stego_architecture, stego_filename = model_helper.load_model(stego_path)

        # Load Weight
        cover_weights = self.h_utils.flatten_model_weights(cover_model)
        stego_weights = self.h_utils.flatten_model_weights(stego_model)
        
        
        # Method 1: Binary LSB comparison (paling akurat)
        lsb_differences = 0
        sample_size = min(10000, len(cover_weights))
        print(f"🔍 Sample Size: {sample_size}")
        for i in range(sample_size):
            bin_c = self.h_utils.float_to_32bit_binary(cover_weights[i])
            bin_s = self.h_utils.float_to_32bit_binary(stego_weights[i])
            
            # Compare only the X LSB bits
            lsb_c = bin_c[-X:]
            lsb_s = bin_s[-X:]
            
            if lsb_c != lsb_s:
                lsb_differences += 1
        
        print(f"✅ LSB differences detected: {lsb_differences}/{sample_size}")
        
        # Method 2: Check modification pattern consistency
        print(f"📊 Modification rate: {(lsb_differences/sample_size)*100:.1f}%")
        
        # Method 3: Verify dengan expected payload pattern
        # expected_pattern = list(binary_payload)  # Pattern sequential as list of bits
        expected_pattern = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]  # Pattern sequential as list of bits
        pattern_matches = 0
        len_hint = 100
        
        for i in range(min(len_hint, len(stego_weights))):
            bin_s = self.h_utils.float_to_32bit_binary(stego_weights[i])
            actual_lsb = bin_s[-1]  # Last bit untuk X=1
            
            expected_bit = expected_pattern[i % len(expected_pattern)]
            
            if actual_lsb == expected_bit:
                pattern_matches += 1
        
        print(f"🎯 Payload pattern matches: {pattern_matches}/{len_hint}")
        
        # Summary
        if lsb_differences > 0:
            print(f"🎉 EMBEDDING SUCCESSFUL! {lsb_differences} LSB modifications detected")
            return True
        else:
            print("❌ No LSB modifications detected")
            return False

    def verify_with_tolerance(self, cover_model, stego_model, X, tolerance=1e-7):
        """Verifikasi dengan tolerance untuk floating point errors"""
        # cover_model, cover_architecture, cover_filename = model_helper.load_model(cover_path)
        # stego_model, stego_architecture, stego_filename = model_helper.load_model(stego_path)

        # Load Weight
        cover_weights = self.h_utils.flatten_model_weights(cover_model)
        stego_weights = self.h_utils.flatten_model_weights(stego_model)
        
        differences = 0
        for i, (w_c, w_s) in enumerate(zip(cover_weights, stego_weights)):
            # Gunakan absolute difference dengan tolerance
            if abs(w_c - w_s) > tolerance:
                differences += 1
        
        print(f"🔍 Weight differences (tolerance {tolerance}): {differences}")
        # return differences

    def test_with_same_images(self, cover_path, stego_path, dataset_name, num_samples=10):
        """Test dengan images yang sama persis untuk fair comparison"""
        
        # Load same exact images untuk kedua model
        (images, labels), class_names = self.h_datasets.load_test_data(dataset_name, num_samples)
        
        # Load models
        cover_model, _, _ = self.h_models.load_model(cover_path)
        stego_model, _, _ = self.h_models.load_model(stego_path)
        
        print(f"\n🔬 FAIR COMPARISON - SAME IMAGES")
        print(f"Cover: {os.path.basename(cover_path)}")
        print(f"Stego: {os.path.basename(stego_path)}")
        
        # Test cover model
        cover_model.to(self.device)
        cover_model.eval()
        with torch.no_grad():
            cover_outputs = cover_model(images.to(self.device))
            cover_probs = torch.softmax(cover_outputs, dim=1)
            cover_confs, cover_preds = torch.max(cover_probs, 1)
        
        # Test stego model  
        stego_model.to(self.device)
        stego_model.eval()
        with torch.no_grad():
            stego_outputs = stego_model(images.to(self.device))
            stego_probs = torch.softmax(stego_outputs, dim=1)
            stego_confs, stego_preds = torch.max(stego_probs, 1)
        
        # Compare results
        print(f"\n📊 SAME IMAGES COMPARISON:")
        print(f"{'Image':<6} {'True':<12} {'Cover Pred':<12} {'Stego Pred':<12} {'Cover Conf':<10} {'Stego Conf':<10}")
        print("-" * 80)
        
        cover_correct = 0
        stego_correct = 0
        confidence_diffs = []
        
        for i in range(len(images)):
            true_label = class_names[labels[i].item()]
            cover_pred = class_names[cover_preds[i].item()]
            stego_pred = class_names[stego_preds[i].item()]
            cover_conf = cover_confs[i].item()
            stego_conf = stego_confs[i].item()
            
            cover_ok = (cover_preds[i] == labels[i]).item()
            stego_ok = (stego_preds[i] == labels[i]).item()
            
            if cover_ok:
                cover_correct += 1
            if stego_ok:
                stego_correct += 1
                
            confidence_diffs.append(stego_conf - cover_conf)
            
            cover_status = "✅" if cover_ok else "❌"
            stego_status = "✅" if stego_ok else "❌"
            
            print(f"{i+1:<6} {true_label:<12} {cover_pred:<12} {stego_pred:<12} {cover_conf:<10.4f} {stego_conf:<10.4f} {cover_status} {stego_status}")
        
        cover_acc = cover_correct / len(images) * 100
        stego_acc = stego_correct / len(images) * 100
        avg_conf_diff = np.mean(confidence_diffs)
        
        print(f"\n📈 SUMMARY (Same {len(images)} Images):")
        print(f"Cover Accuracy: {cover_acc:.1f}%")
        print(f"Stego Accuracy: {stego_acc:.1f}%")
        print(f"Accuracy Difference: {stego_acc - cover_acc:+.1f}%")
        print(f"Average Confidence Change: {avg_conf_diff:+.4f}")
        
        return cover_acc, stego_acc, avg_conf_diff
    # Jalankan fair comparison
    def model_performance_comparison(self, cover_path, stego_path, datasets='cifar10', num_samples=5):
        """Run fair comparison dengan images sama"""
        
        print("🚀 FAIR COMPARISON TEST")
        cover_acc, stego_acc, conf_diff = self.test_with_same_images(
            cover_path, stego_path, datasets, num_samples
        )
        
        # Interpret results
        print(f"\n🎯 INTERPRETATION:")
        if abs(stego_acc - cover_acc) < 5.0:
            print("✅ LSB embedding memiliki NEGLIGIBLE impact pada accuracy")
        elif stego_acc > cover_acc:
            print("🔍 Stego model perform lebih BAIK (mungkin random variation)")
        else:
            print("⚠️ Stego model perform lebih BURUK (perlu investigasi)")
        
        if abs(conf_diff) < 0.01:
            print("✅ Confidence levels tidak berubah signifikan")
        else:
            print(f"🔍 Perubahan confidence: {conf_diff:+.4f}")

    def test_single_inference(self, model, model_path, dataset_name, num_samples=5):
        """Test inference pada single model"""
        print(f"\n🧪 TESTING INFERENCE: {os.path.basename(model_path)}")
        print(f"   Dataset: {dataset_name}")
        
        # Load test data
        (images, labels), class_names = self.h_datasets.load_test_data(dataset_name, num_samples)
        images, labels = images.to(self.device), labels.to(self.device)
        
        # Model ke device
        model.to(self.device)
        model.eval()
        
        # Inference
        with torch.no_grad():
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
        
        # Print results
        print(f"\n📊 INFERENCE RESULTS:")
        print(f"{'Image':<6} {'True':<12} {'Predicted':<12} {'Confidence':<10} {'Correct':<8}")
        print("-" * 60)
        
        correct = 0
        for i in range(len(images)):
            true_label = class_names[labels[i].item()]
            pred_label = class_names[predictions[i].item()]
            confidence = confidences[i].item()
            is_correct = (predictions[i] == labels[i]).item()
            
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{i+1:<6} {true_label:<12} {pred_label:<12} {confidence:<10.4f} {status:<8}")
        
        accuracy = correct / len(images) * 100
        print(f"\n🎯 Accuracy on 5 samples: {accuracy:.1f}%")
        
        return accuracy

    def payload_extraction(self, stego_model_path, X):
        """Test mengekstrak payload dari model stego"""
        print(f"🔍 TESTING PAYLOAD EXTRACTION from {os.path.basename(stego_model_path)}")
        
        # Load stego model
        stego_model, _, _ = self.h_models.load_model(stego_model_path)
        
        # Extract semua weights
        all_weights = []
        for name, param in stego_model.named_parameters():
            if 'weight' in name and param.requires_grad:
                all_weights.extend(param.data.cpu().numpy().flatten().tolist())
        
        # Extract LSB bits
        extracted_bits = []
        for weight in all_weights[:20000]:  # Sample first 10k weights
            binary = self.h_utils.float_to_32bit_binary(weight)
            lsb_bits = binary[-X:]  # Get X LSB bits
            extracted_bits.extend(lsb_bits)
        
        # Convert bits to bytes
        extracted_bytes = bytearray()
        for i in range(0, len(extracted_bits), 8):
            if i + 8 <= len(extracted_bits):
                byte_bits = extracted_bits[i:i+8]
                byte_val = int(''.join(str(b) for b in byte_bits), 2)
                extracted_bytes.append(byte_val)
        
        # Convert to text
        try:
            size, checksum = struct.unpack("i 64s", extracted_bytes[:68])
            payload = extracted_bytes[68:68+size]
            print("✅ PAYLOAD EXTRACTION SUCCESSFUL!")
            print(f"Total {size} characters of extracted payload:")
            print("-" * 50)
            if hashlib.sha256(payload).digest() != checksum[:32]: #bandingkan checksum 32 byte
                print("Checksum mismatch...")
                return None
            # return payload.decode("utf-8", errors="ignore")
            extracted_text = payload.decode('utf-8', errors='ignore')
            print(extracted_text)
            print("-" * 50)
            
            # Check if Python code is recognizable
            if "import " in extracted_text or "def " in extracted_text:
                print("🎯 PYTHON CODE DETECTED in payload!")
                return True
            else:
                print("❌ No recognizable Python code found")
                return False
                
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False

class HelperUtils:
    def __init__(self, config_path="configs/base_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def read_file(self, file_path):
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        return file_bytes

    def float_to_32bit_binary(self, weight: float) -> List[int]:
        """Convert float32 to 32-bit binary representation"""
        int_repr = struct.unpack('!I', struct.pack('!f', weight))[0]
        binary_str = bin(int_repr)[2:].zfill(32)
        return [int(bit) for bit in binary_str]
    
    def binary_to_float(self, binary_list: List[int]) -> float:
        """Convert 32-bit binary list back to float32"""
        binary_str = ''.join(str(bit) for bit in binary_list)
        int_val = int(binary_str, 2)
        return struct.unpack('!f', struct.pack('!I', int_val))[0]

    def flatten_model_weights(self, model: nn.Module) -> List[float]:
        """
        Extract and flatten all weights from model
        """
        weights = []
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad and len(param.shape) > 0:
                weights.extend(param.data.cpu().numpy().flatten().tolist())
        return weights

    def debug_weight_reconstruction(self, *weight_lists, max_print=5):
        """Test apakah weights benar-benar berubah di model"""
        # Compare weights
        # differences = 0
        # Generalized comparison for multiple models (compare any number of input weight lists)
        # def compare_multiple_models(*weight_lists, max_print=5):
        """
        Compare weights of multiple models.
        Args:
            *weight_lists: Lists/tuples of flattened weights (all of same length).
            max_print (int): Maximum number of differences to print (per tuple).
        """
        assert len(weight_lists) >= 2, "Provide at least 2 models to compare."
        n_weights = min(len(w) for w in weight_lists)
        diff_count = 0
        for i in range(n_weights):
            values = [w[i] for w in weight_lists]
            # Compare if not all values are exactly same
            if not all(v == values[0] for v in values[1:]):
                diff_count += 1
                if diff_count <= max_print:
                    print(f"Difference at weight {i}:")
                    for idx, v in enumerate(values):
                        print(f"  Model {idx+1} value: {v}")
                        print(f"  Binary: {self.float_to_32bit_binary(v)}")
        print(f"Total weight differences: {diff_count}")

        # Call for 3 models: finetuned, cover, stego
        # compare_multiple_models(*model_path)
        
        # print(f"Total weight differences: {differences}")
        
class HelperModels:
    def __init__(self, config_path="configs/base_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def find_cover_model_file(self) -> List[str]:
        """Find all .pth files in models directory"""
        config_model_path = self.config['save_path']
        pattern = os.path.join(config_model_path['trained_models'], "**/*.pth")
        model_files = glob.glob(pattern, recursive=True)
        
        # Also check for .pt files
        # pattern_pt = os.path.join(self.models_dir, "**/*.pt")
        # model_files.extend(glob.glob(pattern_pt, recursive=True))
            
        print(f"Found {len(model_files)} model files")
        return model_files

    def find_stego_model_file(self) -> List[str]:
        """Find all .pth files in models directory"""
        config_model_path = self.config['save_path']
        pattern = os.path.join(config_model_path['injected_models'], "**/*.pth")
        model_files = glob.glob(pattern, recursive=True)
        
        # Also check for .pt files
        # pattern_pt = os.path.join(self.models_dir, "**/*.pt")
        # model_files.extend(glob.glob(pattern_pt, recursive=True))
            
        print(f"Found {len(model_files)} model files")
        return model_files

    def create_model_from_architecture(self, architecture: str, num_classes: int = 10) -> nn.Module:
        """Create model instance dengan parameter yang sama seperti training"""
        model_func = getattr(models, architecture)
        model = model_func(num_classes=num_classes)  # <- PAKAI INI, BUKAN weights=None!
        model = model.to(self.device)
        return model

    def load_model(self, model_path: str) -> nn.Module:
        """
        Load model from .pth file with automatic architecture detection
        """
        print(f"Loading model from: {model_path}")

        original_name = os.path.splitext(os.path.basename(model_path))[0]
        
        # Load state_dict
        checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                is_checkpoint = True
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                is_checkpoint = True
            else:
                state_dict = checkpoint
                is_checkpoint = True
        else:
            state_dict = checkpoint
            is_checkpoint = False
        

        # Detect architecture
        # architecture = os.path.basename(original_name).split('_')[0]
        # Architecture detection
        filename = os.path.basename(model_path)
        if 'resnet18' in filename:
            architecture = 'resnet18'
        elif 'resnet34' in filename:
            architecture = 'resnet34' 
        elif 'resnet50' in filename:
            architecture = 'resnet50'
        else:
            architecture = 'resnet18'  # fallback

        # Deteksi num_classes dari nama file
        if 'cifar10' in filename or 'mnist' in filename or 'fashion_mnist' in filename:
            num_classes = 10
        elif 'cifar100' in filename:
            num_classes = 100
        else:
            num_classes = 10  # default

        print(f"Detected architecture: {architecture} with {num_classes} classes")

        # Create model instance
        model = self.create_model_from_architecture(architecture, num_classes)
        
        # Load state_dict with careful handling
        try:
            model.load_state_dict(state_dict)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"⚠️ Standard loading failed: {e}")
            print("Trying with strict=False...")
            model.load_state_dict(state_dict, strict=False)
        
        return model, architecture, os.path.basename(model_path)

class HelperDataset:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_dataset_configs(self):
        """Konfigurasi dataset untuk evaluasi"""
        return {
            'cifar10': {
                'num_classes': 10,
                'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
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
                'classes': [
                    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
                    'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo',
                    'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                    'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
                    'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain',
                    'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
                    'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
                    'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
                    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
                ],
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
                'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
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
                'classes': [
                    'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
                ],
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

    def get_data_loader(self, dataset_name: str, batch_size=128):
        """Dapatkan test loader untuk dataset tertentu"""
        configs = self._get_dataset_configs()
        if dataset_name not in configs:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        config = configs[dataset_name]
        
        if dataset_name == 'cifar10':
            train_dataset = datasets.CIFAR10(
                root='data/datasets/cifar10',
                train=True,
                download=True,
                transform=config['train_transform']
            )
            test_dataset = datasets.CIFAR10(
                root='data/datasets/cifar10',
                train=False,
                download=True,
                transform=config['test_transform']
            )
        elif dataset_name == 'cifar100':
            train_dataset = datasets.CIFAR100(
                root='data/datasets/cifar100',
                train=True,
                download=True,
                transform=config['train_transform']
            )
            test_dataset = datasets.CIFAR100(
                root='data/datasets/cifar100',
                train=False,
                download=True,
                transform=config['test_transform']
            )
        elif dataset_name == 'mnist':
            train_dataset = datasets.MNIST(
                root='data/datasets/mnist',
                train=True,
                download=True,
                transform=config['train_transform']
            )
            test_dataset = datasets.MNIST(
                root='data/datasets/mnist',
                train=False,
                download=True,
                transform=config['test_transform']
            )
        elif dataset_name == 'fashion_mnist':
            train_dataset = datasets.FashionMNIST(
                root='data/datasets/fashion_mnist',
                train=True,
                download=True,
                transform=config['train_transform']
            )
            test_dataset = datasets.FashionMNIST(
                root='data/datasets/fashion_mnist',
                train=False,
                download=True,
                transform=config['test_transform']
            )
        
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    def _detect_dataset_from_filename(self, filename: str) -> str:
        """Deteksi dataset dari nama file"""
        filename_lower = filename.lower()
        if 'cifar10' in filename_lower:
            return 'cifar10'
        elif 'mnist' in filename_lower:
            if 'fashion' in filename_lower:
                return 'fashion_mnist'
            return 'mnist'
        elif 'cifar100' in filename_lower:
            return 'cifar100'
        else:
            return 'cifar10'  # default
    
    def load_test_data(self, dataset_name, num_samples=10):
        """Load beberapa sample test data"""
        configs = self._get_dataset_configs()
        if dataset_name not in configs:
            raise ValueError(f"Dataset {dataset_name} not supported")
        config = configs[dataset_name]
        if dataset_name == 'cifar10':
            test_dataset = datasets.CIFAR10(
                root=f'/data/datasets/{dataset_name}', 
                train=False, 
                download=True,
                transform=config['test_transform'])
        elif dataset_name == 'mnist':
            test_dataset = datasets.MNIST(
                root=f'/data/datasets/{dataset_name}', 
                train=False, 
                download=True,
                transform=config['test_transform'])
        elif dataset_name == 'fashion_mnist':
            test_dataset = datasets.FashionMNIST(
                root=f'/data/datasets/{dataset_name}', 
                train=False, 
                download=True,
                transform=config['test_transform'])
        
        # Ambil random samples
        indices = torch.randperm(len(test_dataset))[:num_samples]
        subset = torch.utils.data.Subset(test_dataset, indices)
        loader = DataLoader(subset, batch_size=num_samples, shuffle=False)
        
        return next(iter(loader)), config['classes']