import onnxruntime as ort
import numpy as np

# Initialize ONNX Runtime session
sess = ort.InferenceSession(
"model.onnx",
providers=["CPUExecutionProvider"] # Use "CudaExecutionProvider" for GPU
)

# Example inference
input_data = np.random.randn(1, 10).astype(np.float32) # Match input shape
outputs = sess.run(
["output"],   # Output names (must match ONNX export)
{"input": input_data}  # Input dict (keys = input names)
)

print("ONNX Output: ", outputs[0])
