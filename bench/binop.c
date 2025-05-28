// clang-format off
#define MT_IMPLEMENTATION
#include "../mint.h"
#include <time.h>
#include <stdio.h>

int main() {
    clock_t start, end;
    double cpu_time_used;
    
    start = clock();
    
    mt_tensor *t1 = mt_tensor_alloc_fill(MT_ARR_INT(100000, 10000), 2, 1.0f);
    mt_tensor *t2 = mt_tensor_alloc_fill(MT_ARR_INT(100000, 10000), 2, 1.0f);
    mt_tensor *t3 = mt_add(t1, t2);

    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // printf("Execution time: %f seconds\n", cpu_time_used);

    // sum the tensor
    double sum = 0.0;
    for (int i = 0; i < mt_tensor_count_element(t3); i++) {
        sum += t3->data[i];
    }
    printf("Sum: %f\n", sum);

    mt_tensor_free(t1);
    mt_tensor_free(t2);
    mt_tensor_free(t3);

    return 0;
}
