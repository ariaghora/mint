```bash
$ protoc --c_out=. onnx.proto3
$ sed -i '' 's|#include <protobuf-c/protobuf-c.h>|#include "protobuf-c.h"|' onnx.proto3.pb-c.h
```