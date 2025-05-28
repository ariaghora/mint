# Regular C
gcc -DMT_USE_PTHREAD -lpthread -O3 -o binop.out binop.c 
time ./binop.out

# gcc -DMT_USE_PTHREAD -lpthread -O3 -o matmul.out matmul.c 
# time ./matmul.out

# With NEON
gcc -DMT_USE_PTHREAD -lpthread -DMT_USE_NEON -O3 -o binop_neon.out binop.c 
time ./binop_neon.out

gcc -DMT_USE_PTHREAD -lpthread -DMT_USE_NEON -O3 -o matmul_neon.out matmul.c 
time ./matmul_neon.out