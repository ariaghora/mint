CC ?= gcc
CFLAGS = -Wall -Wextra -g -O3 -pedantic

TARGET = resnet-18
SRC = $(TARGET).c

.PHONY: all clean

all: $(TARGET).out

$(TARGET).out: $(SRC) ../../mint.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET).out *.onnx *.mt
