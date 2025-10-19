import copy
import hashlib
import struct
import torch
import torch.nn as nn
import numpy as np
import os
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
        Corrected X-LSB Attack algorithm
        X-LSB means modifying the X-th last bit of mantissa, not the last X bits
        """
        print(f"Starting X-LSB Attack at Mantisa Bit Position:{self.X} (from LSB)")
        print(f"Payload size: {len(binary_string)} bits")
        
        # Get flattened weights
        W = self.utils.flatten_model_weights(model)
        nW = len(W)
        
        print(f"Total weights: {nW:,}")
        print(f"Total capacity: {nW:,} bits")  # Capacity is nW, not nW * X
        
        # Check capacity
        ns = len(binary_string)
        if ns > nW:
            raise ValueError(f"Payload too large. Required: {ns:,} bits, Available: {nW:,} bits")
        
        # Convert weights to binary matrix
        BW = []
        for weight in W:
            binary_repr = self.utils.float_to_32bit_binary(weight)
            BW.append(binary_repr)
        
        BW = np.array(BW)
        
        # Embed payload - each bit goes to the X-th last position
        for i in range(min(ns, nW)):
            # Calculate the target bit position (X-th from LSB)
            # Bit positions: 0=LSB, 1=2nd last, 2=3rd last, etc.
            target_bit_position = 32 - self.X  # In 32-bit representation
            
            BW[i, target_bit_position] = int(binary_string[i])
        
        # Reconstruct floats
        Wc = []
        for i in range(nW):
            binary_row = BW[i].tolist()
            modified_float = self.utils.binary_to_float(binary_row)
            Wc.append(modified_float)
        
        # Reconstruct model
        Mc = self.reconstruct_model_from_weights(Wc, model)
        
        modified_weights = min(nW, ns)
        print(f"✓ Embedding completed. Modified {modified_weights:,}/{nW:,} weights")
        print(f"✓ Each modified weight had bit at position {self.X} from LSB changed")
        
        return Mc
    # def embed_binary_string(self, model: nn.Module, binary_string: str) -> nn.Module:
    #     """
    #     Main X-LSB Attack algorithm merubah x bit terakhir
    #     """
    #     print(f"Starting X-LSB Attack at Mantisa Bit:{self.X}")
    #     print(f"Payload size: {len(binary_string)} bits")
        
    #     # Get flattened weights
    #     W = self.utils.flatten_model_weights(model)
    #     nW = len(W)
        
    #     print(f"Total weights: {nW:,}")
    #     print(f"Total capacity: {nW * self.X:,} bits")
        
    #     # Check capacity
    #     ns = len(binary_string)
    #     if ns > nW * self.X:
    #         raise ValueError(f"Payload too large. Required: {ns:,} bits, Available: {nW * self.X:,} bits")
        
    #     # Convert weights to binary matrix
    #     BW = []
    #     for weight in W:
    #         binary_repr = self.utils.float_to_32bit_binary(weight)
    #         BW.append(binary_repr)
        
    #     BW = np.array(BW)
        
    #     # Prepare payload segments
    #     q = ns // self.X
    #     r = ns % self.X
        
    #     print(f"Full segments (q): {q:,}, Remainder bits (r): {r}")
        
    #     s_bits = np.array([int(bit) for bit in binary_string])
        
    #     # Reshape full segments
    #     if q > 0:
    #         BW_block = s_bits[:q * self.X].reshape(q, self.X)
    #     else:
    #         BW_block = np.array([]).reshape(0, self.X)
        
    #     BW_remainder = s_bits[q * self.X:q * self.X + r]
        
    #     # Embed payload
    #     for i in range(min(q, nW)):
    #         BW[i, 32 - self.X:32] = BW_block[i]
        
    #     if r > 0 and q < nW:
    #         BW[q, 32 - r:32] = BW_remainder
        
    #     # Reconstruct floats
    #     Wc = []
    #     for i in range(nW):
    #         binary_row = BW[i].tolist()
    #         modified_float = self.utils.binary_to_float(binary_row)
    #         Wc.append(modified_float)
        
    #     # Reconstruct model
    #     Mc = self.reconstruct_model_from_weights(Wc, model)
        
    #     modified_weights = min(nW, q + (1 if r > 0 else 0))
    #     print(f"✓ Embedding completed. Modified {modified_weights:,}/{nW:,} weights")
        
    #     return Mc


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
    
    def create_payload_with_header(self, payload_code: str) -> List[int]:
        """
        Create payload dengan length header dan checksum untuk extraction yang reliable
        """
        # Convert payload to bytes
        if isinstance(payload_code, str):
            payload_bytes = payload_code.encode('utf-8')
        else:
            payload_bytes = payload_code
        
        # Calculate SHA256 checksum
        checksum = hashlib.sha256(payload_bytes).digest()
        
        # Create header: size (4 bytes) + magic signature (4 bytes) + checksum (32 bytes)
        # Magic signature "PAY_" untuk identifikasi payload
        header = struct.pack("I 4s 32s", len(payload_bytes), b"PAY_", checksum)
        
        # Combine header + payload
        full_payload = header + payload_bytes
        
        print(f"📦 Payload siap untuk di embed ke bobot:")
        print(f"   Code size: {len(payload_bytes)} bytes")
        print(f"   Header: {len(header)} bytes (size + magic + checksum)")
        print(f"   Total: {len(full_payload)} bytes")
        print(f"   Magic signature: PAY_")
        print(f"   SHA256: {checksum.hex()[:16]}...")
        
        # Convert to bitstream (List[int])
        bitstream = []
        for byte in full_payload:
            bits = format(byte, '08b')
            bitstream.extend([int(b) for b in bits])
        
        print(f"   BinaryPayload: {bitstream}")
            
        return bitstream

    def create_malicious_sequential_payload(self, payload_size: int = 1024) -> str:
        """Create malicious payload with pattern"""
        # Create a pattern for easy verification -> pakai sequential pattern
        payload = bytearray()
        for i in range(payload_size):
            payload.append(i % 256)
        
        return ''.join(format(byte, '08b') for byte in payload)

    def create_real_malicious_payload(self, malware_path: str) -> List[int]:
        """
        Create malicious payload dari file dengan header yang reliable
        """
        # Baca file malware
        payload_bytes = self.h_utils.read_file(malware_path)
        
        # Buat payload dengan header yang lengkap
        return self.create_payload_with_header(payload_bytes)

    def create_malicious_like_payload(self, size: int) -> str:
        # Pattern yang meniru karakteristik malware real
        payload = bytearray()
        for i in range(size):
            # Pattern yang lebih complex dan less predictable
            byte_val = (i * 7 + 13) % 256  # Linear congruential pattern
            payload.append(byte_val)
        return ''.join(format(byte, '08b') for byte in payload)

    def extract_payload_from_model(self, model_path: str, X: int = 1) -> Optional[str]:
        """
        Extract payload dari model dengan method yang reliable (tidak terpotong)
        """
        print(f"🔍 Extracting payload from {os.path.basename(model_path)} with X={X}")
        
        # Load model
        model, _, _ = self.h_model.load_model(model_path)
        
        # Extract semua weights
        all_weights = self.h_utils.flatten_model_weights(model)
        print(f"📊 Processing {len(all_weights):,} weights...")
        
        # Extract bits dari LSB
        extracted_bits = []
        header_found = False
        payload_length = 0
        total_bits_needed = 0
        
        for i, weight in enumerate(all_weights):
            if i % 10000 == 0 and i > 0:
                print(f"   Processed {i:,} weights...")
                
            binary = self.h_utils.float_to_32bit_binary(weight)
            extracted_bits.append(binary[-X])  # Get X LSB bits
            
            # Cari header setelah memiliki cukup bits untuk header (40 bytes = 320 bits)
            if not header_found and len(extracted_bits) >= 320:
                # Coba decode header
                header_bytes = bytearray()
                for j in range(0, 320, 8):  # 40 bytes header
                    if j + 8 <= len(extracted_bits):
                        byte_bits = extracted_bits[j:j+8]
                        byte_val = int(''.join(map(str, byte_bits)), 2)
                        header_bytes.append(byte_val)
                
                try:
                    # Parse header: size (4 bytes) + magic (4 bytes) + checksum (32 bytes)
                    if len(header_bytes) >= 40:
                        payload_length, magic, checksum = struct.unpack("I 4s 32s", header_bytes[:40])
                        
                        if magic == b"PAY_":  # Valid header found!
                            print(f"⚠️ Header found! Payload length: {payload_length} bytes")
                            header_found = True
                            total_bits_needed = 320 + (payload_length * 8)  # header + payload
                            print(f"📦 Need {total_bits_needed} total bits")
                            print(f"Extract bits {len(extracted_bits)} >= {total_bits_needed} total bits")
                            
                except struct.error:
                    continue  # Continue searching for header
            
            # Jika header sudah ditemukan, extract sampai lengkap
            if header_found and len(extracted_bits) >= total_bits_needed:
                print(f"⚠️ Collected {len(extracted_bits)} bits, extracting payload...")
                
                # Extract payload bytes (skip 40 bytes header)
                payload_bytes = bytearray()
                for j in range(320, total_bits_needed, 8):  # Skip header
                    if j + 8 <= len(extracted_bits):
                        byte_bits = extracted_bits[j:j+8]
                        byte_val = int(''.join(map(str, byte_bits)), 2)
                        payload_bytes.append(byte_val)
                
                # Trim to exact length
                payload_bytes = payload_bytes[:payload_length]
                
                if len(payload_bytes) == payload_length:
                    # Verify checksum
                    extracted_checksum = hashlib.sha256(payload_bytes).digest()
                    if extracted_checksum == checksum:
                        print("✅ Checksum verified! Payload integrity confirmed.")
                        try:
                            payload_code = payload_bytes.decode('utf-8', errors='ignore')
                            print(f"✅ Complete payload extracted: {len(payload_code)} characters")
                            # print(payload_code)
                            return payload_code
                        except Exception as e:
                            print(f"❌ Decoding error: {e}")
                            return None
                    else:
                        print("❌ Checksum mismatch! Payload corrupted.")
                        return None
                else:
                    print(f"❌ Incomplete payload: {len(payload_bytes)}/{payload_length} bytes")
                    return None
            
            # Safety limit - stop jika terlalu banyak weights diproses
            if i > 20000:  # Max 20k weights
            # if i > 500000:  # Max 500k weights
                print("⚠️  Safety limit reached, stopping extraction")
                break
        
        print("❌ No valid payload header found")
        return None

    def verify_payload_extraction(self, model_path: str, X: int = 1) -> bool:
        """
        Verify bahwa payload bisa diekstrak dengan utuh
        """
        print(f"\n🔍 VERIFYING PAYLOAD EXTRACTION for {os.path.basename(model_path)}")
        
        extracted_payload = self.extract_payload_from_model(model_path, X)
        
        if extracted_payload:
            print("✅ PAYLOAD EXTRACTION SUCCESSFUL!")
            print(f"First 200 characters:")
            print("-" * 50)
            print(extracted_payload[:200])
            print("-" * 50)
            
            # Check jika ini Python code
            if "import " in extracted_payload or "def " in extracted_payload:
                print("🎯 PYTHON CODE DETECTED in payload!")
            return True, extracted_payload
        else:
            print("❌ PAYLOAD EXTRACTION FAILED!")
            return False

    def process_models_batch(self, x_bits_list: List[int] = [1, 2, 3], malware_data=None, evaluate_models: bool = False, verify_extraction: bool = True, fill_attack: bool = False):
        """
        Process all models dengan payload extraction verification
        """
        evaluator = ModelEvaluator(self.device)
        model_files = self.h_model.find_cover_model_file()
        
        if not model_files:
            print("No model files found!")
            return
        
        results = []
        evaluation_results = []
        
        # for model_path in model_files[:1]: #test 1 models
        for model_path in model_files:
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
                        # Gunakan method baru dengan header yang reliable
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
                        
                        # Save standard version
                        output_path = os.path.join(
                            self.output_dir, 
                            f"{original_name}_x{X}.pth"
                        )
                        torch.save(stego_model.state_dict(), output_path)
                        
                        # Verify payload extraction jika diminta
                        if verify_extraction:
                            extraction_success = self.verify_payload_extraction(output_path, X)
                            if not extraction_success:
                                print("⚠️  WARNING: Payload extraction verification failed!")
                        
                        # Evaluate stego model
                        stego_metrics = None
                        if evaluate_models and original_metrics is not None:
                            comparison = evaluator.compare_models(
                                model, stego_model, dataset_name
                            )
                            stego_metrics = comparison
                        
                        # Save evaluation results
                        if stego_metrics:
                            eval_result = {
                                'model_file': output_path,
                                'x_bits': X,
                                'dataset': dataset_name,
                                'architecture': architecture,
                                'evaluation': stego_metrics,
                                'extraction_verified': extraction_success if verify_extraction else None
                            }
                            evaluation_results.append(eval_result)

                        print(f"✓ Saved: {os.path.basename(output_path)}")
                        
                    except ValueError as e:
                        print(f"✗ Standard attack failed: {e}")
                    
                    # Fill attack
                    if fill_attack:
                        try:
                            fill_attacker = XLSBAttackFill(X=X)
                            stego_model_fill = fill_attacker.embed_binary_string_fill(model, payload)
                            
                            # Save fill version
                            output_path_fill = os.path.join(
                                self.output_dir,
                                f"{original_name}_x{X}_fill.pth"
                            )
                            torch.save(stego_model_fill.state_dict(), output_path_fill)
                            
                            # Verify payload extraction untuk fill version
                            if verify_extraction:
                                self.verify_payload_extraction(output_path_fill, X)
                            
                            # Evaluate fill version
                            if evaluate_models and original_metrics is not None:
                                evaluator.compare_models(
                                    model, stego_model_fill, dataset_name
                                )

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
        
        # Save results
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
    
    # Process all models dengan payload extraction verification
    batch_processor = BatchXLSBAttack(MODELS_DIR, OUTPUT_DIR)
    batch_processor.process_models_batch(
        x_bits_list=X_BITS_LIST,
        malware_data=None,  # atau path ke file malware
        evaluate_models=True,
        verify_extraction=True  # Enable payload extraction verification
    )

if __name__ == "__main__":
    main()