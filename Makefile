CFLAGS=-Wall -O3 -g -pedantic -lblas -I/opt/homebrew/opt/openblas/include 

.PHONY: alexnet
alexnet: alexnet.c mint.h
	clang -o alexnet alexnet.c $(CFLAGS)
