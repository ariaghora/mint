> This library is super new, and I cannot guarantee long-term maintenance commitment. If you need something faster and highly optimized, you should use [ggml](https://github.com/ggerganov/ggml) or a similar alternative.

---

```
                                M I N T

                       A minimalist tensor library


  Mint is a single-file header only library for tensor manipulation. It also
enables importing and executing *some* of neural net models. Mint aims to be
dependency-free and easily distributed, but it is possible to integrate with
the other libraries such as BLAS if needed.

  Some of notable features:
- NumPy style broadcasting
- BLAS backend (optional)
- OpenMP acceleration (when linked)

```

---

## Tested models

### Torchvision Models

The torchvision models are dumped into ONNX, then converted to Mint model format for inference.

- AlexNet
- VGG-19
- ResNet-18
