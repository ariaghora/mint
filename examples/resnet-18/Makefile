CC ?= gcc
CFLAGS = -Wall -Wextra -g -O3 -pedantic 

use_pthread ?= 0
ifeq ($(use_pthread), 1)
	CFLAGS += -DMT_USE_PTHREAD -lpthread
endif

use_neon ?= 0
ifeq ($(use_neon), 1)
	CFLAGS += -DMT_USE_NEON
endif

use_apple_accelerate ?= 0
ifeq ($(use_apple_accelerate), 1)
	CFLAGS += -DMT_USE_APPLE_ACCELERATE -framework Accelerate
endif

TARGET = resnet-18
SRC = $(TARGET).c 
LIB_SRC = ../../onnx/*.c

.PHONY: all clean

all: $(TARGET).out

$(TARGET).out: $(SRC) $(LIB_SRC) ../../mint.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LIB_SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET).out *.onnx *.mt
