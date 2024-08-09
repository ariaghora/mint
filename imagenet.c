#include <assert.h>
#include <stdio.h>

#define MT_USE_OPENMP
#define MT_USE_BLAS
#define MT_USE_IM2ROW_CONV
#include "mint.h"

int main(void) {
  int out_c = 5;

  mt_tensor *x = mt_tensor_alloc_fill(MT_ARR_INT(3, 200, 200), 3, 1);
  mt_tensor *w = mt_tensor_alloc_fill(MT_ARR_INT(out_c, 3, 3, 3), 4, 0.5);
  mt_tensor *b =
      mt_tensor_alloc_values(MT_ARR_INT(out_c), 1, MT_ARR_FLOAT(1, 1));
  mt_tensor *res = mt_convolve_2d(x, w, b, 1, 0);

  for (int i = 0; i < mt__tensor_count_element(res); ++i) {
    printf("%f\n", res->data[i]);
  }

  mt_tensor_free(x);
  mt_tensor_free(w);
  mt_tensor_free(b);
  mt_tensor_free(res);

  mt_tensor *ma = mt_tensor_alloc_random(MT_ARR_INT(3000, 1000), 2);
  mt_tensor *mb = mt_tensor_alloc_random(MT_ARR_INT(1000, 3000), 2);
  res           = mt_matmul(ma, mb);
  mt_tensor_free(res);
  mt_tensor_free(ma);
  mt_tensor_free(mb);
  return 0;
}
