# pytorch_deepspeed_onnx
## PyTorch, DeepSpeed &amp; ONNX Neural Network with mixed precision

### Key Notes :spiral_notepad:

> PyTorch: Defines the model(`SimpleModel`), data loading (`DataLoader`), and training loop.
> DeepSpeed:
- Mixed Precision (fp16): Automatically converts tensors to half precision to save memory.
- ZeRO Stage 1 (zero_optimization): Shards optimizer states across GPUs (or memory in a single GPU)
- Gradient Accumulation: Simulates larger batch sizes acummulating gradients over multiple steps.
- Simplified Training Loop: `model_engine` abstracts away boilerplate (eg. `loss.backward()` becomes `model_engine.backward(loss)`) 

__Conclusion__

- DeepSpeed integrates with PyTorch to automate memory optimizations (like mixed and optimizer sharding) while keeping the training loop clean. The actual heavy lifting (eg. gradient synchronization, memory management) is handled by DeepSpeed under the hood.
---

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

