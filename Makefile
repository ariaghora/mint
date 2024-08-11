CC?=gcc
CFLAGS=-Wall -O3 -g -pedantic -fopenmp -lblas -I/opt/homebrew/opt/openblas/include

.PHONY: alexnet
alexnet: alexnet.c mint.h
	$(CC) -o alexnet alexnet.c $(CFLAGS)
