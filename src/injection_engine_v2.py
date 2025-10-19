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

class MantissaLSBInjector:
    def __init__(self, config_path="configs/injection_config.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        self.payload_sources = {
            "random": self._generate_random_payload,
            "text": self._generate_text_payload,
            "image": self._generate_image_payload
        }
    
    def _get_default_config(self):
        return {
            'injection': {
                'mantissa_lsb_bits': [1, 2, 3, 4],  # Bits di mantissa
                'payload_types': ['random', 'text', 'image'],
                'injection_ratio': [0.1, 0.25, 0.5, 0.75, 1.0],
                'target_layers': ['all'],
                'save_metadata': True
            }
        }

    def float_to_ieee754(self, f):
        """Convert float to IEEE 754 binary representation"""
        packed = struct.pack('!f', f)
        int_bits = int.from_bytes(packed, byteorder='big', signed=False)
        binary = format(int_bits, '032b')
        return binary

    def ieee754_to_float(self, binary):
        """Convert IEEE 754 binary back to float"""
        int_bits = int(binary, 2)
        packed = int_bits.to_bytes(4, byteorder='big')
        f = struct.unpack('!f', packed)[0]
        return f

    def inject_mantissa_lsb(self, weight_value, payload_bits, num_lsb_bits=1):
        """
        Inject payload into LSB of mantissa
        
        Float32: [1 sign][8 exponent][23 mantissa]
        LSB mantissa = bit 22 (paling kanan dari mantissa)
        """
        if len(payload_bits) < num_lsb_bits:
            return weight_value, payload_bits
        
        # Convert to binary
        binary_representation = self.float_to_ieee754(weight_value)
        
        # Extract components
        sign_bit = binary_representation[0]
        exponent_bits = binary_representation[1:9]
        mantissa_bits = binary_representation[9:]  # 23 bits mantissa
        
        # Modify LSB of mantissa
        mantissa_list = list(mantissa_bits)
        
        for i in range(num_lsb_bits):
            # Inject from LSB mantissa (position 22, 21, ...)
            bit_pos = 22 - i
            if bit_pos >= 0 and i < len(payload_bits):
                mantissa_list[bit_pos] = str(payload_bits[i])
        
        # Reconstruct
        modified_mantissa = ''.join(mantissa_list)
        modified_binary = sign_bit + exponent_bits + modified_mantissa
        
        # Convert back to float
        try:
            modified_float = self.ieee754_to_float(modified_binary)
            return modified_float, payload_bits[num_lsb_bits:]
        except:
            return weight_value, payload_bits

    def _generate_random_payload(self, size_in_bits):
        return np.random.randint(0, 2, size_in_bits, dtype=np.uint8)

    def _generate_text_payload(self, size_in_bits, text_file="payloads/secret_message.txt"):
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            binary_text = ''.join(format(ord(c), '08b') for c in text)
            binary_array = np.array([int(bit) for bit in binary_text], dtype=np.uint8)
            if len(binary_array) < size_in_bits:
                repeats = (size_in_bits // len(binary_array)) + 1
                binary_array = np.tile(binary_array, repeats)
            return binary_array[:size_in_bits]
        except FileNotFoundError:
            return self._generate_random_payload(size_in_bits)

    def _generate_image_payload(self, size_in_bits, image_file="payloads/secret_image.png"):
        try:
            from PIL import Image
            import io
            img = Image.open(image_file)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_binary = img_byte_arr.getvalue()
            binary_str = ''.join(format(byte, '08b') for byte in img_binary)
            binary_array = np.array([int(bit) for bit in binary_str], dtype=np.uint8)
            if len(binary_array) < size_in_bits:
                repeats = (size_in_bits // len(binary_array)) + 1
                binary_array = np.tile(binary_array, repeats)
            return binary_array[:size_in_bits]
        except (FileNotFoundError, ImportError):
            return self._generate_random_payload(size_in_bits)

    def _calculate_model_capacity(self, model_state_dict, target_layers='all'):
        """Calculate maximum payload capacity"""
        total_bits = 0
        layer_info = {}
        
        for layer_name, weights in model_state_dict.items():
            if target_layers != 'all':
                if target_layers == 'conv' and not any(x in layer_name for x in ['conv', 'weight']):
                    continue
                elif target_layers == 'fc' and 'fc' not in layer_name:
                    continue
            
            if 'weight' in layer_name or 'conv' in layer_name or 'fc' in layer_name:
                num_elements = weights.numel()
                # Capacity: each weight can store num_lsb_bits
                max_bits = self.config['injection']['mantissa_lsb_bits'][-1]
                layer_bits = num_elements * max_bits
                total_bits += layer_bits
                
                layer_info[layer_name] = {
                    'num_elements': num_elements,
                    'capacity_bits': layer_bits,
                    'shape': weights.shape
                }
        
        return total_bits, layer_info

    def inject_to_model(self, model_path, output_dir="models/injected_models/", 
                    payload_type="random", num_lsb_bits=2, injection_ratio=0.1):
        """Main injection function"""
        print(f"\n=== MANTISSA LSB INJECTION ===")
        print(f"Model: {os.path.basename(model_path)}")
        print(f"Payload: {payload_type}, Mantissa LSB bits: {num_lsb_bits}, Ratio: {injection_ratio}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            is_checkpoint = True
        else:
            model_state_dict = checkpoint
            is_checkpoint = False
        
        # Calculate capacity
        total_capacity, layer_info = self._calculate_model_capacity(model_state_dict, 'all')
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
        total_injected_weights = 0
        
        for layer_name, weights in tqdm(model_state_dict.items(), desc="Injecting layers"):
            if layer_name not in layer_info:
                continue
            
            weights_np = weights.cpu().numpy()
            original_shape = weights_np.shape
            weights_flat = weights_np.flatten()
            
            layer_capacity = layer_info[layer_name]['capacity_bits']
            max_injectable = min(len(remaining_payload), layer_capacity // num_lsb_bits)
            
            if max_injectable == 0:
                continue
            
            injected_count = 0
            for i in range(len(weights_flat)):
                if len(remaining_payload) < num_lsb_bits:
                    break
                
                if random.random() <= injection_ratio and injected_count < max_injectable:
                    try:
                        current_payload = remaining_payload[:num_lsb_bits]
                        modified_value, remaining_payload = self.inject_mantissa_lsb(
                            weights_flat[i], current_payload, num_lsb_bits
                        )
                        weights_flat[i] = modified_value
                        injected_count += 1
                        total_injected_weights += 1
                    except Exception as e:
                        continue
            
            # Update weights if any were injected
            if injected_count > 0:
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
        output_filename = f"{base_name}_mantissa_{num_lsb_bits}bits_{int(injection_ratio*100)}percent.pth"
        output_path = os.path.join(output_dir, output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save injected model
        if is_checkpoint:
            checkpoint['model_state_dict'] = model_state_dict
            checkpoint['injection_metadata'] = {
                'payload_type': payload_type,
                'num_lsb_bits': num_lsb_bits,
                'injection_ratio': injection_ratio,
                'target_layers': 'all',
                'original_payload_size': len(payload_bits),
                'injected_payload_size': len(payload_bits) - len(remaining_payload),
                'injected_weights': total_injected_weights,
                'injected_layers': injected_layers,
                'injection_log': injection_log,
                'original_model': model_path,
                'method': 'mantissa_lsb'
            }
            torch.save(checkpoint, output_path)
        else:
            metadata = {
                'model_state_dict': model_state_dict,
                'injection_metadata': checkpoint['injection_metadata']
            }
            torch.save(metadata, output_path)
        
        print(f"\nInjection completed!")
        print(f"Injected payload: {len(payload_bits) - len(remaining_payload)} bits")
        print(f"Injected weights: {total_injected_weights}")
        print(f"Injected layers: {len(injected_layers)}")
        print(f"Output saved: {output_path}")
        
        return output_path

    def batch_inject_models(self, model_list_file="models/generated_models_list.txt"):
        """Batch injection for all models"""
        print("=== BATCH MANTISSA LSB INJECTION ===")
        
        if not os.path.exists(model_list_file):
            print(f"Model list file not found: {model_list_file}")
            return
        
        with open(model_list_file, 'r') as f:
            model_paths = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(model_paths)} models for injection")
        
        injected_models = []
        injection_configs = []
        
        for lsb_bits in self.config['injection']['mantissa_lsb_bits']:
            for payload_type in self.config['injection']['payload_types']:
                for injection_ratio in self.config['injection']['injection_ratio']:
                    injection_configs.append({
                        'lsb_bits': lsb_bits,
                        'payload_type': payload_type,
                        'injection_ratio': injection_ratio
                    })
        
        for model_path in model_paths:
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue
            
            # Inject with various configurations
            for config in injection_configs[:2]:  # Limit for testing
                try:
                    print(f"\nInjecting {os.path.basename(model_path)} with config: {config}")
                    
                    injected_path = self.inject_to_model(
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

def create_mantissa_injection_config():
    """Create configuration for mantissa LSB injection"""
    config = {
        'injection': {
            'mantissa_lsb_bits': [1, 2, 3, 4],
            'payload_types': ['random', 'text', 'image'],
            'injection_ratio': [0.1, 0.25, 0.5, 0.75, 1.0],
            'target_layers': ['all'],
            'save_metadata': True
        }
    }
    
    os.makedirs('configs', exist_ok=True)
    with open('configs/injection_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Mantissa LSB injection config created")

if __name__ == "__main__":
    create_mantissa_injection_config()
    
    # Test injection
    injector = MantissaLSBInjector()
    
    # Find a model to test
    model_files = []
    for root, dirs, files in os.walk("models/trained_models"):
        for file in files:
            if file.endswith('.pth') and 'scratch_best' in file:
                model_files.append(os.path.join(root, file))
    
    if model_files:
        test_model = model_files[0]
        print(f"Testing mantissa LSB injection on: {test_model}")
        
        injected_path = injector.inject_to_model(
            test_model,
            payload_type="random",
            num_lsb_bits=2,
            injection_ratio=0.1
        )
        
        print(f"Test injection completed: {injected_path}")