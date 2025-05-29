// clang-format off
#define MT_IMPLEMENTATION
#include "../../mint.h"

int main() {
    /*****************************************************************
     *     NOTE: All tensor values are stored in row-major order
     *****************************************************************/

    //   Create tensor with values
    //   -------------------------
    // Creating a tensor with specified values can be done using
    // mt_tensor_alloc_values(). For example, to create a 2 by 2
    // 2 dimensional tensor (a matrix) we can do this:
    mt_tensor *t = mt_tensor_alloc_values(
        (int[]){2, 2},        // shape
        2,                    // number of dimensions
        (float[]){1, 2, 3, 4} // the values
    );

    // You can access the data directly like this
    for (int i = 0; i< mt_tensor_count_element(t); ++i) {
        printf("%f\n", t->data[i]);
    }

    // There is also a helper function to print tensor
    mt_tensor_print(t);

    // Don't forget to free accordingly.
    mt_tensor_free(t);


    /*****************************************************************
     *                 Simple arithmetic operations
     *****************************************************************/

    // Create two tensors
    mt_tensor *a = mt_tensor_alloc_values(
        (int[]){2, 2},
        2,
        (float[]){1, 2, 3, 4}
    );

    mt_tensor *b = mt_tensor_alloc_values(
        (int[]){2, 2},
        2,
        (float[]){5, 6, 7, 8}
    );

    // Add the two tensors
    mt_tensor *c = mt_add(a, b);

    // Print the result
    printf("a + b = \n");
    mt_tensor_print(c);

    mt_tensor_free(a), mt_tensor_free(b), mt_tensor_free(c);

    /*****************************************************************
     *                         Broadcasting
     *****************************************************************/

    // Let's add two tensors with different shapes, where a is of shape
    // (1, 4) and b is of shape (4, 1). Broadcasting will be applied to
    // make the shapes compatible (like Python's numpy). The result will
    // be a tensor of shape (4, 4).

    a = mt_tensor_alloc_values(
        (int[]){1, 4},
        2,
        (float[]){1, 2, 3, 4}
    );

    b = mt_tensor_alloc_values(
        (int[]){4, 1},
        2,
        (float[]){1, 2, 3, 4}
    );

    // (Element-wise) multiply the two tensors
    c = mt_mul(a, b);

    // Print the result
    printf("a * b = \n");
    mt_tensor_print(c);

    // Free the tensors
    mt_tensor_free(a), mt_tensor_free(b), mt_tensor_free(c);

    return 0;
}
