import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import deepspeed

# 1. Define a simple model
class SimpleModel(nn.Module):
	def __init__(self):
	super().__init__()
	self.fc1 = nn.Linear(10, 5)
	self.fc1 = nn.Linear(5, 1)

	def forward(self, x)
	x = torch.relu(self.fci(x))
	return self.fc2(x)

# 2. Create dummy data
X = torch.randn(100, 10) # 1000 samples, 10 features
y = torch.randn(1000, 1) # 1000 targets
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. initialize DeepSpeed (automatically handles mixed precision, ZeRO, etc.)
model = SimpleModel()
parameters = filter(lambda p: p.requires_grad, model.parameters())
model_engine, optimizer, _, _ = deepspeed.initialize(
model = model,
model_parameters=parameters,
config_params={
"train_batch_size": 32,
"gradient_accumulation_steps": 2,  # Simulate larger batch size
"fp16": {"enabled": True},  # mixed precision
"zero_optimization": {"stage": 1} # ZeRO stage 1 (optimizer sharding)
}
)

# 4. Training loop
for epoch in range(5):
for batch_x, batch_y in dataloader:
#foward pass
outputs = model_engine(batch_X)
loss = nn.MSELoss()(outputs, batch_y)

# Backward pass (DeepSpeed handles gradient sharding)
model_engine.backward(loss)
model_engine.step()

# Export to ONNX
torch.onnx.export(
model,    # Model to export
dummy_input,  # Example input
"model.onnx",   # Output file
input_names=["input"],  # Input name (for inference)
output_names=["output"], # Enable dynamic batch size
dynamic_axes={
"input": {0: "batch_size"},
"output": {0, "batch_size"}
},
opset_version=13    # ONNX opset version
)

print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
