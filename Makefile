CFLAGS=-Wall -g -pedantic -fopenmp -lblas -I/opt/homebrew/opt/openblas/include

.PHONY: imagenet
imagenet: imagenet.c mint.h
	clang -o imagenet imagenet.c $(CFLAGS)
