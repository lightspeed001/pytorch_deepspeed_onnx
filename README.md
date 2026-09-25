# pytorch_deepspeed_onnx
## PyTorch, DeepSpeed &amp; ONNX Neural Network with mixed precision

### Key Notes :spiral_notepad:


### Build and Run Instructions :hammer_and_wrench:

- Build the Docker Image:

```sh
docker build -t pytorch-deepspeed-app .
```

- Run the container (with GPU support)

```sh
docker run --gpus all -p 8000:8000 pytorch-deepspeed-app
```

- test the endpoint:

```sh
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{"input_data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}'

```

This setup provides:
- GPU acceleration
- Proper model serving
- Health checks
- Clean shutdown handling
- Production-ready configuration

