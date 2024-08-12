CC?=gcc
CFLAGS=-Wall -g -O3 -pedantic 
OPT_FLAGS=-fopenmp -lblas -I/opt/homebrew/opt/openblas/include

.PHONY: alexnet
alexnet: alexnet.c mint.h
	$(CC) -o alexnet alexnet.c $(CFLAGS) $(OPT_FLAGS)

.PHONY: alexnet_mt
alexnet_mt: alexnet_mt.c mint.h
	$(CC) -o alexnet_mt alexnet_mt.c $(CFLAGS) -lblas
