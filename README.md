Mint is a single-file header only library for tensor manipulation. It also
enables importing and executing *some* of neural net models. Mint aims to be
dependency-free (except for C standard lib) and easily distributed. However,
it is possible to integrate with the other libraries such as BLAS if needed.

  Some of notable features:

- NumPy style broadcasting
- BLAS backend (optional)
- ARM NEON SIMD acceleration (enable with `#define MT_USE_NEON`).
  - For Apple Silicon devices (M1/M2/M3), you can additionally define `#define MT_USE_APPLE_ACCELERATE` to even more performance. Specify Accelerate framework during compilation.

## Issues

- Got a `layer kind xxx is not supported yet` error? Try running `onnxsim`. It might trim those pesky unsupported layers
- If onnxsim still gives you unsupported layers: just open an issue and I'll try to implement it when I can `¯\_(ツ)_/¯`


## Tested models

### Torchvision Models

The torchvision models are dumped into ONNX, then converted to Mint model format for inference.

- AlexNet
- VGG-19
- ResNet-18

### Fast neural style transfer

All models [here](https://github.com/onnx/models/tree/main/validated/vision/style_transfer/fast_neural_style) with opset 8 work well
