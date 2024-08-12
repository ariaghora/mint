CC ?= gcc
CFLAGS = -Wall -Wextra -g -O3 -pedantic
LDFLAGS = -lblas

TARGET = resnet-18
SRC = $(TARGET).c

.PHONY: all clean

all: $(TARGET).out

$(TARGET).out: $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET).out *.onnx *.mt
