TARGET = onnx_loader
SRC = $(TARGET).c ../../onnx/*.c
CFLAGS = -g -Wall -Wextra -Werror

.PHONY: all clean

all: $(TARGET).out

$(TARGET).out: $(SRC) ../../mint_onnx.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET).out 
