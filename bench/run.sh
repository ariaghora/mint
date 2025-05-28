#!/bin/bash

# Regular C
gcc -O3 -o binop.out binop.c 
./binop.out

# With NEON
gcc -DMT_USE_NEON -DMT_USE_OPENMP -O3 -o binop_neon.out binop.c 
./binop_neon.out