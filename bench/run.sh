#!/bin/bash

# Regular C
gcc -DMT_USE_PTHREAD -lpthread -O3 -o binop.out binop.c 
time ./binop.out

# With NEON
gcc -DMT_USE_PTHREAD -lpthread -DMT_USE_NEON -DMT_USE_OPENMP -O3 -o binop_neon.out binop.c 
time ./binop_neon.out