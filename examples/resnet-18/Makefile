CC ?= gcc

CFLAGS = -Wall -Wextra -g -O3 -pedantic 
# CFLAGS = -Wall -Wextra -g -O3 -pedantic -ffast-math -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp 

TARGET = resnet-18
SRC = $(TARGET).c

.PHONY: all clean

all: $(TARGET).out

$(TARGET).out: $(SRC) ../../mint.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET).out *.onnx *.mt
