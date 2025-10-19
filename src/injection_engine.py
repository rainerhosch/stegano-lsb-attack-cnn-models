import torch
import torch.nn as nn
import numpy as np
import os
import struct
import random
from tqdm import tqdm
import yaml
import pickle
from typing import List, Dict, Any, Union

class LSBInjector:
    def __init__(self, config_path="configs/injection_config.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config atau gunakan default
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        # Payload options
        self.payload_sources = {
            "random": self._generate_random_payload,
            "text": self._generate_text_payload,
            "image": self._generate_image_payload
        }
    
    def _get_default_config(self):
        """Default configuration untuk injeksi LSB"""
        return {
            'injection': {
                'lsb_bits': [1, 2, 4],
                'payload_types': ['random', 'text', 'image'],
                'injection_ratio': [0.1, 0.25, 0.5, 0.75, 1.0],
                'target_layers': ['conv', 'fc', 'bn', 'all'],
                'save_metadata': True
            }
        }
    
    def _generate_random_payload(self, size_in_bits):
        """Generate random binary payload"""
        return np.random.randint(0, 2, size_in_bits, dtype=np.uint8)
    
    def _generate_text_payload(self, size_in_bits, text_file="payloads/secret_message.txt"):
        """Generate payload dari text file"""
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Convert text to binary
            binary_text = ''.join(format(ord(c), '08b') for c in text)
            binary_array = np.array([int(bit) for bit in binary_text], dtype=np.uint8)
            
            # Repeat jika payload terlalu kecil
            if len(binary_array) < size_in_bits:
                repeats = (size_in_bits // len(binary_array)) + 1
                binary_array = np.tile(binary_array, repeats)
            
            return binary_array[:size_in_bits]
            
        except FileNotFoundError:
            print(f"Text file {text_file} not found, using random payload instead")
            return self._generate_random_payload(size_in_bits)
    
    def _generate_image_payload(self, size_in_bits, image_file="payloads/secret_image.png"):
        """Generate payload dari image file"""
        try:
            from PIL import Image
            import io
            
            # Load dan konversi image ke binary
            img = Image.open(image_file)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_binary = img_byte_arr.getvalue()
            
            # Convert bytes to binary string
            binary_str = ''.join(format(byte, '08b') for byte in img_binary)
            binary_array = np.array([int(bit) for bit in binary_str], dtype=np.uint8)
            
            # Repeat jika payload terlalu kecil
            if len(binary_array) < size_in_bits:
                repeats = (size_in_bits // len(binary_array)) + 1
                binary_array = np.tile(binary_array, repeats)
            
            return binary_array[:size_in_bits]
            
        except (FileNotFoundError, ImportError):
            print(f"Image file {image_file} not found or PIL not available, using random payload instead")
            return self._generate_random_payload(size_in_bits)
    
    def _float_to_binary(self, value):
        """Convert float32 to binary representation"""
        # Pack float32 to bytes, then convert to binary string
        packed = struct.pack('!f', value)
        binary_str = ''.join(format(byte, '08b') for byte in packed)
        return binary_str
    
    def _binary_to_float(self, binary_str):
        """Convert binary string back to float32"""
        # Convert binary string to bytes
        bytes_data = bytes(int(binary_str[i:i+8], 2) for i in range(0, 32, 8))
        value = struct.unpack('!f', bytes_data)[0]
        return value
    
    def _inject_lsb_into_float(self, value, payload_bits, num_lsb_bits):
        """Inject LSB bits into a single float value"""
        if len(payload_bits) < num_lsb_bits:
            raise ValueError("Not enough payload bits")
        
        # Convert float to binary
        binary_representation = self._float_to_binary(value)
        
        # Replace LSB bits
        binary_list = list(binary_representation)
        
        # Untuk float32, kita bisa inject di berbagai posisi
        # Pilihan: last few bits of mantissa (least significant part)
        start_pos = 32 - num_lsb_bits  # Last few bits of the binary representation
        for i in range(num_lsb_bits):
            if start_pos + i < len(binary_list):
                binary_list[start_pos + i] = str(payload_bits[i])
        
        modified_binary = ''.join(binary_list)
        
        # Convert back to float
        try:
            modified_float = self._binary_to_float(modified_binary)
            return modified_float, payload_bits[num_lsb_bits:]
        except:
            # Jika konversi gagal, return nilai original
            return value, payload_bits
    
    def _calculate_model_capacity(self, model_state_dict, target_layers='all'):
        """Hitung kapasitas maksimum payload yang bisa disisipkan"""
        total_bits = 0
        layer_info = {}
        
        for layer_name, weights in model_state_dict.items():
            # Filter layers berdasarkan target
            if target_layers != 'all':
                if target_layers == 'conv' and not any(x in layer_name for x in ['conv', 'weight']):
                    continue
                elif target_layers == 'fc' and 'fc' not in layer_name:
                    continue
                elif target_layers == 'bn' and not any(x in layer_name for x in ['bn', 'batch_norm']):
                    continue
            
            # Only inject into weight parameters (bias biasanya lebih sensitif)
            if 'weight' in layer_name or 'conv' in layer_name or 'fc' in layer_name:
                num_elements = weights.numel()
                layer_bits = num_elements * self.config['injection']['lsb_bits'][-1]  # Max bits
                total_bits += layer_bits
                
                layer_info[layer_name] = {
                    'num_elements': num_elements,
                    'capacity_bits': layer_bits,
                    'shape': weights.shape
                }
        
        return total_bits, layer_info
    
    def inject_lsb_to_model(self, model_path, output_dir="models/injected_models/", 
                        payload_type="random", num_lsb_bits=2, 
                        injection_ratio=1.0, target_layers='all'):
        """Inject LSB payload ke model"""
        print(f"\n=== LSB Injection ===")
        print(f"Model: {os.path.basename(model_path)}")
        print(f"Payload: {payload_type}, LSB bits: {num_lsb_bits}, Ratio: {injection_ratio}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            is_checkpoint = True
        else:
            model_state_dict = checkpoint
            is_checkpoint = False
        
        # Calculate capacity
        total_capacity, layer_info = self._calculate_model_capacity(model_state_dict, target_layers)
        payload_size = int(total_capacity * injection_ratio)
        
        print(f"Model capacity: {total_capacity} bits")
        print(f"Payload size: {payload_size} bits")
        print(f"Layers targeted: {len(layer_info)}")
        
        # Generate payload
        payload_generator = self.payload_sources[payload_type]
        payload_bits = payload_generator(payload_size)
        
        print(f"Generated payload: {len(payload_bits)} bits")
        
        # Inject payload
        remaining_payload = payload_bits.copy()
        injected_layers = []
        injection_log = {}
        
        for layer_name, weights in tqdm(model_state_dict.items(), desc="Injecting layers"):
            if layer_name not in layer_info:
                continue
            
            # Convert weights to numpy untuk processing
            weights_np = weights.cpu().numpy()
            original_shape = weights_np.shape
            weights_flat = weights_np.flatten()
            
            # Calculate how many weights to inject in this layer
            layer_capacity = layer_info[layer_name]['capacity_bits']
            max_injectable = min(len(remaining_payload), layer_capacity // num_lsb_bits)
            
            if max_injectable == 0:
                continue
            
            # Inject into selected weights
            injected_count = 0
            for i in range(len(weights_flat)):
                if len(remaining_payload) < num_lsb_bits:
                    break
                
                if random.random() <= injection_ratio and injected_count < max_injectable:
                    try:
                        # Get next payload bits
                        current_payload = remaining_payload[:num_lsb_bits]
                        
                        # Inject into weight
                        modified_value, remaining_payload = self._inject_lsb_into_float(
                            weights_flat[i], current_payload, num_lsb_bits
                        )
                        
                        weights_flat[i] = modified_value
                        injected_count += 1
                        
                    except Exception as e:
                        print(f"Error injecting into {layer_name}[{i}]: {e}")
                        continue
            
            # Reshape back to original
            modified_weights = weights_flat.reshape(original_shape)
            model_state_dict[layer_name] = torch.from_numpy(modified_weights)
            
            injected_layers.append(layer_name)
            injection_log[layer_name] = {
                'injected_weights': injected_count,
                'total_weights': len(weights_flat),
                'injection_ratio': injected_count / len(weights_flat)
            }
            
            if len(remaining_payload) < num_lsb_bits:
                break
        
        # Prepare output
        base_name = os.path.basename(model_path).replace('.pth', '')
        output_filename = f"{base_name}_injected_{payload_type}_{num_lsb_bits}bits_{int(injection_ratio*100)}percent.pth"
        output_path = os.path.join(output_dir, output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save injected model
        if is_checkpoint:
            checkpoint['model_state_dict'] = model_state_dict
            # Add injection metadata
            checkpoint['injection_metadata'] = {
                'payload_type': payload_type,
                'num_lsb_bits': num_lsb_bits,
                'injection_ratio': injection_ratio,
                'target_layers': target_layers,
                'original_payload_size': len(payload_bits),
                'injected_payload_size': len(payload_bits) - len(remaining_payload),
                'injected_layers': injected_layers,
                'injection_log': injection_log,
                'original_model': model_path
            }
            torch.save(checkpoint, output_path)
        else:
            # Save injection metadata separately
            metadata = {
                'model_state_dict': model_state_dict,
                'injection_metadata': {
                    'payload_type': payload_type,
                    'num_lsb_bits': num_lsb_bits,
                    'injection_ratio': injection_ratio,
                    'target_layers': target_layers,
                    'original_payload_size': len(payload_bits),
                    'injected_payload_size': len(payload_bits) - len(remaining_payload),
                    'injected_layers': injected_layers,
                    'injection_log': injection_log,
                    'original_model': model_path
                }
            }
            torch.save(metadata, output_path)
        
        # Save payload info separately
        if self.config['injection']['save_metadata']:
            payload_info = {
                'payload_type': payload_type,
                'payload_bits': payload_bits,
                'injection_config': {
                    'num_lsb_bits': num_lsb_bits,
                    'injection_ratio': injection_ratio,
                    'target_layers': target_layers
                },
                'model_info': {
                    'original_model': model_path,
                    'injected_model': output_path,
                    'total_capacity': total_capacity,
                    'injected_bits': len(payload_bits) - len(remaining_payload)
                }
            }
            
            metadata_path = output_path.replace('.pth', '_metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump(payload_info, f)
        
        print(f"\nInjection completed!")
        print(f"Injected payload: {len(payload_bits) - len(remaining_payload)} bits")
        print(f"Remaining payload: {len(remaining_payload)} bits")
        print(f"Injected layers: {len(injected_layers)}")
        print(f"Output saved: {output_path}")
        
        return output_path
    
    def batch_inject_models(self, model_list_file="models/generated_models_list.txt"):
        """Batch injection untuk semua model dalam list"""
        print("=== BATCH LSB INJECTION ===")
        
        if not os.path.exists(model_list_file):
            print(f"Model list file not found: {model_list_file}")
            return
        
        # Load model list
        with open(model_list_file, 'r') as f:
            model_paths = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(model_paths)} models for injection")
        
        injected_models = []
        
        for model_path in model_paths:
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue
            
            # Generate various injection configurations
            injection_configs = []
            
            for lsb_bits in self.config['injection']['lsb_bits']:
                for payload_type in self.config['injection']['payload_types']:
                    for injection_ratio in self.config['injection']['injection_ratio']:
                        injection_configs.append({
                            'lsb_bits': lsb_bits,
                            'payload_type': payload_type,
                            'injection_ratio': injection_ratio
                        })
            
            # Inject dengan berbagai konfigurasi
            for config in injection_configs[:2]:  # Batasi untuk testing
                try:
                    print(f"\nInjecting {os.path.basename(model_path)} with config: {config}")
                    
                    injected_path = self.inject_lsb_to_model(
                        model_path=model_path,
                        payload_type=config['payload_type'],
                        num_lsb_bits=config['lsb_bits'],
                        injection_ratio=config['injection_ratio']
                    )
                    
                    injected_models.append(injected_path)
                    
                except Exception as e:
                    print(f"Error injecting {model_path}: {e}")
                    continue
        
        # Save list of injected models
        with open("models/injected_models_list.txt", "w") as f:
            for model_path in injected_models:
                f.write(f"{model_path}\n")
        
        print(f"\n=== BATCH INJECTION COMPLETED ===")
        print(f"Total injected models: {len(injected_models)}")
        return injected_models
    
    def verify_injection(self, original_model_path, injected_model_path):
        """Verifikasi bahwa injeksi berhasil dan tidak merusak model"""
        print(f"\n=== Verifying Injection ===")
        
        # Load models
        original_checkpoint = torch.load(original_model_path, map_location=self.device)
        injected_checkpoint = torch.load(injected_model_path, map_location=self.device)
        
        if isinstance(original_checkpoint, dict) and 'model_state_dict' in original_checkpoint:
            original_state = original_checkpoint['model_state_dict']
        else:
            original_state = original_checkpoint
        
        if isinstance(injected_checkpoint, dict) and 'model_state_dict' in injected_checkpoint:
            injected_state = injected_checkpoint['model_state_dict']
        else:
            injected_state = injected_checkpoint
        
        # Check for differences
        different_weights = 0
        total_weights = 0
        
        for layer_name in original_state:
            if layer_name in injected_state:
                orig_weights = original_state[layer_name].cpu().numpy()
                inj_weights = injected_state[layer_name].cpu().numpy()
                
                diff_mask = orig_weights != inj_weights
                different_weights += np.sum(diff_mask)
                total_weights += orig_weights.size
        
        injection_ratio = different_weights / total_weights if total_weights > 0 else 0
        
        print(f"Different weights: {different_weights} / {total_weights} ({injection_ratio:.4f})")
        
        # Check metadata
        if 'injection_metadata' in injected_checkpoint:
            metadata = injected_checkpoint['injection_metadata']
            print(f"Injection metadata:")
            for key, value in metadata.items():
                if key not in ['injection_log']:
                    print(f"  {key}: {value}")
        
        return injection_ratio > 0

def create_injection_config():
    """Create injection configuration file"""
    config = {
        'injection': {
            'lsb_bits': [1, 2, 4],
            'payload_types': ['random', 'text', 'image'],
            'injection_ratio': [0.1, 0.25, 0.5, 0.75, 1.0],
            'target_layers': ['conv', 'fc', 'bn', 'all'],
            'save_metadata': True
        }
    }
    
    os.makedirs('configs', exist_ok=True)
    with open('configs/injection_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Injection config created: configs/injection_config.yaml")

if __name__ == "__main__":
    # Create config file jika belum ada
    if not os.path.exists("configs/injection_config.yaml"):
        create_injection_config()
    
    # Test injection engine
    injector = LSBInjector()
    
    # Cari model untuk di-test
    model_files = []
    for root, dirs, files in os.walk("models/trained_models"):
        for file in files:
            if file.endswith('.pth') and 'scratch_best' in file:
                model_files.append(os.path.join(root, file))
    
    if model_files:
        test_model = model_files[0]
        print(f"Testing injection on: {test_model}")
        
        # Test single injection
        injected_path = injector.inject_lsb_to_model(
            test_model,
            payload_type="random",
            num_lsb_bits=2,
            injection_ratio=0.1
        )
        
        # Verify injection
        injector.verify_injection(test_model, injected_path)
        
        print("\n=== SINGLE INJECTION TEST COMPLETED ===")
        
        # Tanya user apakah mau lanjut batch injection
        response = input("\nRun batch injection on all models? (y/n): ")
        if response.lower() == 'y':
            injector.batch_inject_models()
    else:
        print("No trained models found. Please run model training first.")