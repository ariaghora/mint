gcc -DMT_USE_PTHREAD -lpthread -DMT_USE_NEON -o tensor.out tensor.c 
./tensor.out
rm -f tensor.out
