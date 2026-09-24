#import numpy as np
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
def predict(data: list):
	input_data = np.array(data, dtype=np.float32)
	outputs = sess.run(["output"], {"input"}: input_data)
	return {"prediction": outputs[0].tolist()}
