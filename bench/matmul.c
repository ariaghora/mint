// clang-format off
#define MT_IMPLEMENTATION
#include "../mint.h"
#include <time.h>
#include <stdio.h>

int main() {
    mt_tensor *t1 = mt_tensor_alloc_fill(MT_ARR_INT(10000, 1000), 2, 1.0f);
    mt_tensor *t2 = mt_tensor_alloc_fill(MT_ARR_INT(1000, 10000), 2, 1.0f);
    mt_tensor *t3 = mt_matmul(t1, t2);

    mt_tensor_free(t1);
    mt_tensor_free(t2);
    mt_tensor_free(t3);


    return 0;
}
