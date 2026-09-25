import numpy as np
from fastapi import FastAPI
import deepspeed
from pydantic import BaseModel
app = FastAPI()

# load your model (adjust paths as needed)
model = SimpleModel()
model_engine, _, _, _ = deepspeed.initialize(
	model=model,
	model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
	config_params={
		"train_batch_size": 32,
		"fp16": {"enabled": True},
		"zero_optimization": {"stage": 1}
	}
)

# I would load weights here
# model_engine.load_state_dict(torch.load("model_weights.pt"))

app = FastAPI()

class PredictionRequest(BaseModel):
	# Convert input to tensor
	input_tensor = torch.tensor([request.input_data], dtype=torch.float32)

	# make prediction
	with torch.no_grad():
		output = model_engine(input_tensor).tolist()

		return {"prediction": output[0]}


@app.get("/health")
async def health_check():
	return {"status": "healthy"}

if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=8000)
