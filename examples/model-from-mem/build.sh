python ../helper/download-resnet.py
python ../../scripts/dump.py resnet-18.onnx
gcc -o main.out main.c
