import copy
import hashlib
import struct
import torch
import torch.nn as nn
# import torchvision.models as models
import numpy as np
# import struct
import os
# import glob
from typing import List, Dict, Optional
import json
from src.model_evaluator import ModelEvaluator
from src.utils.helpers import HelperDataset, HelperModels, HelperUtils

class UniversalXLSBAttack:
    """
    Enhanced X-LSB Attack for various model architectures
    Handles ResNet18, ResNet34, ResNet50, etc.
    """
    def __init__(self, X: int):
        """
        Initialize X-LSB Attack for multiple architectures
        
        Args:
            X: Number of LSB bits to attack (1 ≤ X ≤ 23)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert 1 <= X <= 23, "X must be between 1 and 23 (mantissa bits)"
        self.X = X
        self.supported_architectures = {
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'vgg11', 'vgg13', 'vgg16', 'vgg19',
            'alexnet', 'densenet121', 'mobilenet_v2'
        }
        self.utils = HelperUtils()
        self.model_helper = HelperModels()
    
    def reconstruct_model_from_weights(self, flat_weights: List[float], original_model: nn.Module) -> nn.Module:
        """
        Reconstruct model with modified weights
        """
        # Create a copy
        modified_model = copy.deepcopy(original_model)
        # print(f"Modified Model: {modified_model}")

        modified_model.load_state_dict(original_model.state_dict())
        
        # Reconstruct weights
        flat_idx = 0
        
        for name, param in modified_model.named_parameters():
            if 'weight' in name and param.requires_grad and len(param.shape) > 0:
                param_size = param.data.numel()
                param_shape = param.data.shape
                
                # Extract weights for this parameter
                if flat_idx + param_size <= len(flat_weights):
                    param_weights = flat_weights[flat_idx:flat_idx + param_size]
                    param_tensor = torch.tensor(param_weights, dtype=param.dtype).reshape(param_shape)
                    param.data = param_tensor
                    
                    flat_idx += param_size
                else:
                    print(f"⚠️ Not enough weights for {name}")
        
        return modified_model
    
    def embed_binary_string(self, model: nn.Module, binary_string: str) -> nn.Module:
        """
        Main X-LSB Attack algorithm
        """
        print(f"Starting X-LSB Attack at Mantisa Bit:{self.X}")
        print(f"Payload size: {len(binary_string)} bits")
        
        # Get flattened weights
        W = self.utils.flatten_model_weights(model)
        nW = len(W)
        
        print(f"Total weights: {nW:,}")
        print(f"Total capacity: {nW * self.X:,} bits")
        
        # Check capacity
        ns = len(binary_string)
        if ns > nW * self.X:
            raise ValueError(f"Payload too large. Required: {ns:,} bits, Available: {nW * self.X:,} bits")
        
        # Convert weights to binary matrix
        BW = []
        for weight in W:
            binary_repr = self.utils.float_to_32bit_binary(weight)
            BW.append(binary_repr)
        
        BW = np.array(BW)
        
        # Prepare payload segments
        q = ns // self.X
        r = ns % self.X
        
        print(f"Full segments (q): {q:,}, Remainder bits (r): {r}")
        
        s_bits = np.array([int(bit) for bit in binary_string])
        
        # Reshape full segments
        if q > 0:
            BW_block = s_bits[:q * self.X].reshape(q, self.X)
        else:
            BW_block = np.array([]).reshape(0, self.X)
        
        BW_remainder = s_bits[q * self.X:q * self.X + r]
        
        # Embed payload
        for i in range(min(q, nW)):
            BW[i, 32 - self.X:32] = BW_block[i]
        
        if r > 0 and q < nW:
            BW[q, 32 - r:32] = BW_remainder
        
        # Reconstruct floats
        Wc = []
        for i in range(nW):
            binary_row = BW[i].tolist()
            modified_float = self.utils.binary_to_float(binary_row)
            Wc.append(modified_float)
        
        # Reconstruct model
        Mc = self.reconstruct_model_from_weights(Wc, model)
        
        modified_weights = min(nW, q + (1 if r > 0 else 0))
        print(f"✓ Embedding completed. Modified {modified_weights:,}/{nW:,} weights")
        
        return Mc


class XLSBAttackFill(UniversalXLSBAttack):
    """
    X-LSB Attack - Fill version for various architectures
    """
    
    def embed_binary_string_fill(self, model: nn.Module, binary_string: str) -> nn.Module:
        """
        X-LSB Attack - Fill version
        """
        print(f"Starting X-LSB Attack - Fill with X={self.X}")
        
        helper_utils = HelperUtils()
        # helper_model = HelperModels()
        
        # Get capacity
        W = helper_utils.flatten_model_weights(model)
        nW = len(W)
        total_capacity = nW * self.X
        
        print(f"Total capacity: {total_capacity:,} bits")
        print(f"Original payload length: {len(binary_string):,} bits")
        
        # Extend or truncate payload
        if len(binary_string) < total_capacity:
            repeat_count = total_capacity // len(binary_string) + 1
            extended_payload = (binary_string * repeat_count)[:total_capacity]
            print(f"Payload repeated to {len(extended_payload):,} bits")
        else:
            extended_payload = binary_string[:total_capacity]
            print(f"Payload truncated to {len(extended_payload):,} bits")
        
        # Use standard embedding with extended payload
        return self.embed_binary_string(model, extended_payload)


# =============================================================================
# BATCH PROCESSING FOR MULTIPLE MODELS
# =============================================================================

class BatchXLSBAttack:
    """
    Batch processor for injecting multiple models
    """
    
    def __init__(self, models_dir: str, output_dir: str):
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(output_dir, exist_ok=True)
        self.h_utils = HelperUtils()
        self.h_model = HelperModels()
        self.h_datasets = HelperDataset()
    
    def create_malicious_sequential_payload(self, payload_size: int = 1024) -> str:
        """Create malicious payload with pattern"""
        # Create a pattern for easy verification -> pakai sequential pattern
        # Contoh untuk payload_size = 5:
        # payload_bytes = [0, 1, 2, 3, 4]  # Bukan random!
        # binary_payload = "00000000 00000001 00000010 00000011 00000100"

        payload = bytearray()
        for i in range(payload_size):
            payload.append(i % 256)
        
        return ''.join(format(byte, '08b') for byte in payload)

    def create_real_malicious_payload(self, malware_path: str) -> List[int]:
        # with open(malware_path, 'rb') as f:
        #     payload_bytes = f.read()

        payload_bytes = self.h_utils.read_file(malware_path)
        # Calculate size code of bytes
        size = len(payload_bytes)
        # Calculate SHA256 checksum
        checksum = hashlib.sha256(payload_bytes).digest()
        # Create header: size (4 bytes) + checksum (64 bytes)
        header = struct.pack("I 64s", size, checksum + b'\x00' * 32)
        # Combine header + payload
        full_payload = header + payload_bytes
        # return ''.join(format(byte, '08b') for byte in final_payload)
        # Convert to bitstream (List[int])
        bitstream = []
        for byte in full_payload:
            bits = format(byte, '08b')
            bitstream.extend([int(b) for b in bits])
        return bitstream

        
    def create_malicious_like_payload(self, size: int) -> str:
        # Pattern yang meniru karakteristik malware real
        payload = bytearray()
        for i in range(size):
            # Pattern yang lebih complex dan less predictable
            byte_val = (i * 7 + 13) % 256  # Linear congruential pattern
            payload.append(byte_val)
        return ''.join(format(byte, '08b') for byte in payload)


    def process_models_batch(self, x_bits_list: List[int] = [1, 2, 3], malware_data=None, evaluate_models: bool = False):
        """
        Process all models with different X values
        """
        evaluator = ModelEvaluator(self.device)
        model_files = self.h_model.find_cover_model_file()
        
        if not model_files:
            print("No model files found!")
            return
        
        results = []
        evaluation_results = []
        
        for model_path in model_files[:1]: #test 1 models
            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(model_path)}")
            print(f"{'='*60}")
            original_name = os.path.splitext(os.path.basename(model_path))[0]
            try:
                # Load model
                model, architecture, filename = self.h_model.load_model(model_path)
                
                # Deteksi dataset dari nama file
                dataset_name = self.h_datasets._detect_dataset_from_filename(filename)
                # Calculate model statistics
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                float_params = sum(p.numel() for p in model.parameters() 
                                if p.requires_grad and p.dtype == torch.float32)
                                
                print(f"Architecture: {architecture}")
                print(f"Total parameters: {total_params:,}")
                print(f"Float parameters: {float_params:,}")
                # Evaluate original model jika diminta
                original_metrics = None
                if evaluate_models:
                    print(f"\n📊 Evaluating original model...")
                    test_loader = self.h_datasets.get_data_loader(dataset_name)
                    original_metrics = evaluator.evaluate_model(model, test_loader)
                    print(f"Original model accuracy: {original_metrics['accuracy']:.2f}%")
                
                # Start Injection    
                for X in x_bits_list:
                    print(f"\n--- X-LSB = {X} ---")
                    
                    if malware_data == None:
                        # Create appropriate payload size
                        capacity = float_params * X
                        payload_size_bytes = min(1024, capacity // 8)
                        payload = self.create_malicious_sequential_payload(payload_size_bytes)
                    else:
                        payload_list = self.create_real_malicious_payload(malware_data)
                        # Convert list payload_list to string of bytes
                        if isinstance(payload_list, list):
                            payload = ''.join(map(str, payload_list))
                        else:
                            payload = str(payload_list)
                        
                    # Standard attack
                    try:
                        standard_attacker = UniversalXLSBAttack(X=X)
                        stego_model = standard_attacker.embed_binary_string(model, payload)
                        # Evaluate stego model
                        stego_metrics = None
                        if evaluate_models and original_metrics is not None:
                            comparison = evaluator.compare_models(
                                model, stego_model, dataset_name
                            )
                            stego_metrics = comparison
                        # Save standard version
                        output_path = os.path.join(
                            self.output_dir, 
                            f"{original_name}_x{X}.pth"
                        )
                        torch.save(stego_model.state_dict(), output_path)
                        
                        # Save evaluation results
                        if stego_metrics:
                            eval_result = {
                                'model_file': output_path,
                                'x_bits': X,
                                'dataset': dataset_name,
                                'architecture': architecture,
                                'evaluation': stego_metrics
                            }
                            evaluation_results.append(eval_result)

                        print(f"✓ Saved: {os.path.basename(output_path)}")
                        
                    except ValueError as e:
                        print(f"✗ Standard attack failed: {e}")
                    
                    # Fill attack
                    try:
                        fill_attacker = XLSBAttackFill(X=X)
                        stego_model_fill = fill_attacker.embed_binary_string_fill(model, payload)
                        
                        # Evaluate fill version
                        if evaluate_models and original_metrics is not None:
                            evaluator.compare_models(
                                model, stego_model_fill, dataset_name
                            )

                        # Save fill version
                        output_path_fill = os.path.join(
                            self.output_dir,
                            f"{original_name}_x{X}_fill.pth"
                        )
                        torch.save(stego_model_fill.state_dict(), output_path_fill)
                        print(f"✓ Saved (fill): {os.path.basename(output_path_fill)}")
                        
                    except Exception as e:
                        print(f"✗ Fill attack failed: {e}")

                results.append({
                    'file': filename,
                    'architecture': architecture,
                    'dataset': dataset_name,
                    'total_params': total_params,
                    'float_params': float_params,
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"✗ Failed to process {model_path}: {e}")
                results.append({
                    'file': os.path.basename(model_path),
                    'architecture': 'unknown',
                    'status': f'failed: {e}'
                })
        
        # Save processing summary
        # summary_path = os.path.join(self.output_dir, "processing_summary.json")
        # with open(summary_path, 'w') as f:
        #     json.dump(results, f, indent=2)
        
        # print(f"\nProcessing completed. Summary saved to: {summary_path}")
        evaluator._save_results(results, evaluation_results)

        
        # Print summary
        successful = sum(1 for r in results if r['status'] == 'success')
        print(f"Successfully processed: {successful}/{len(results)} models")


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def main():
    """Main function to process all models in trained_models directory"""
    
    # Configuration
    MODELS_DIR = "models/trained_models"
    OUTPUT_DIR = "stego_models"
    X_BITS_LIST = [1, 2, 3]
    
    # Process all models
    batch_processor = BatchXLSBAttack(MODELS_DIR, OUTPUT_DIR)
    batch_processor.process_models_batch(X_BITS_LIST)

if __name__ == "__main__":
    main()