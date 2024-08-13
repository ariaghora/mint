// clang-format off
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
        MT_ARR_INT(2, 2),        // shape
        2,                       // ndim
        MT_ARR_FLOAT(1, 2, 3, 4) // the values
    );

    // You can access the data directly like this
    for (int i = 0; i< mt_tensor_count_element(t); ++i) {
        printf("%f\n", t->data[i]);
    }

    mt_tensor_debug_info(t);

    // Don't forget to free accordingly.
    mt_tensor_free(t);

    return 0;
}
