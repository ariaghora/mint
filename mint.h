#ifndef _MINT_H_
#define _MINT_H_

#define MAX_NDIM 5
#define MAX_GLOBAL_TENSOR_COUNT 1000

// The tensor values data type
#define mt_float float

// The tensor representation
typedef struct {
  mt_float *data;

  int ndim;
  int shape[MAX_NDIM];
  int count_deps;
} mt_tensor;

mt_tensor *mt_convolve_2d(mt_tensor *x, mt_tensor *w, mt_tensor *b, int stride,
                          int pad);
mt_tensor *mt_matmul(mt_tensor *a, mt_tensor *b);
mt_tensor *mt_max_pool_2d(mt_tensor *x, mt_tensor *w, mt_tensor *b, int stride,
                          int pad);
mt_tensor *mt_tensor_alloc(int *shape, int ndim);
mt_tensor *mt_tensor_alloc_fill(int *shape, int ndim, mt_float value);
mt_tensor *mt_tensor_alloc_values(int *shape, int ndim, mt_float *values);
mt_tensor *mt_tensor_alloc_random(int *shape, int ndim);
void       mt_tensor_free(mt_tensor *t);

/**
 *
 * IMPLEMENTATION
 *
 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MT_USE_OPENMP
  #include <omp.h>
#endif

#define MT_ARR_INT(...) ((int[]){__VA_ARGS__})
#define MT_ARR_FLOAT(...) ((mt_float[]){__VA_ARGS__})

static int mt__tensor_count_element(mt_tensor *t) {
  int count = 1;
  for (int i = 0; i < t->ndim; ++i) {
    count *= t->shape[i];
  }
  return count;
}

static void mt__im2row(mt_tensor *t, mt_float *row_data, int K_h, int K_w,
                       int stride, int pad) {
  int C_in = t->shape[0], H_in = t->shape[1], W_in = t->shape[2];
  int H_out = (H_in + 2 * pad - K_h) / stride + 1;
  int W_out = (W_in + 2 * pad - K_w) / stride + 1;

  int row = 0;
  for (int h_out = 0; h_out < H_out; h_out++) {
    for (int w_out = 0; w_out < W_out; w_out++) {
      for (int c_in = 0; c_in < C_in; c_in++) {
        for (int kh = 0; kh < K_h; kh++) {
          for (int kw = 0; kw < K_w; kw++) {
            int h_in = h_out * stride + kh - pad;
            int w_in = w_out * stride + kw - pad;

            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
              row_data[row] = t->data[c_in * H_in * W_in + h_in * W_in + w_in];
            } else {
              row_data[row] = 0;
            }
            row++;
          }
        }
      }
    }
  }
}

mt_tensor *mt_convolve_2d(mt_tensor *x, mt_tensor *w, mt_tensor *b, int stride,
                          int pad) {
  assert(x->ndim == 3);
  assert(w->ndim == 4);
  assert(b->ndim == 1);
  assert(x->shape[0] == w->shape[1]);
  assert(b->shape[0] == w->shape[0]);

  int C_in = x->shape[0];
  int H_in = x->shape[1];
  int W_in = x->shape[2];

  int C_out = w->shape[0];
  int K_h   = w->shape[2];
  int K_w   = w->shape[3];

  // Calculate output dimensions with padding
  int H_out = (H_in + 2 * pad - K_h) / stride + 1;
  int W_out = (W_in + 2 * pad - K_w) / stride + 1;

#ifdef MT_USE_IM2ROW_CONV
  int        row_size = C_in * K_h * K_w;
  int        col_size = H_out * W_out;
  mt_tensor *row_data = mt_tensor_alloc(MT_ARR_INT(row_size, col_size), 2);
  mt__im2row(x, row_data->data, K_h, K_w, stride, pad);

  mt_tensor *weight_matrix =
      mt_tensor_alloc(MT_ARR_INT(C_out, C_in * K_h * K_w), 2);
  // Rearrange weights to match row-major order
  for (int c_out = 0; c_out < C_out; c_out++) {
    for (int c_in = 0; c_in < C_in; c_in++) {
      for (int kh = 0; kh < K_h; kh++) {
        for (int kw = 0; kw < K_w; kw++) {
          int w_idx = ((c_out * C_in + c_in) * K_h + kh) * K_w + kw;
          int wm_idx =
              c_out * (C_in * K_h * K_w) + (kh * K_w + kw) * C_in + c_in;
          weight_matrix->data[wm_idx] = w->data[w_idx];
        }
      }
    }
  }
  mt_tensor *output = mt_matmul(weight_matrix, row_data);

  // Add bias
  for (int c_out = 0; c_out < C_out; c_out++) {
    for (int i = 0; i < H_out * W_out; i++) {
      output->data[c_out * H_out * W_out + i] += b->data[c_out];
    }
  }

  mt_tensor_free(row_data);
  mt_tensor_free(weight_matrix);
#else
  // Allocate output tensor
  mt_tensor *output = mt_tensor_alloc(MT_ARR_INT(C_out, H_out, W_out), 3);
  #ifdef MT_USE_OPENMP
    #pragma omp parallel for
  #endif
  for (int c_out = 0; c_out < C_out; c_out++) {
    for (int h_out = 0; h_out < H_out; h_out++) {
      for (int w_out = 0; w_out < W_out; w_out++) {
        mt_float sum = 0.0f;

        for (int c_in = 0; c_in < C_in; c_in++) {
          for (int kh = 0; kh < K_h; kh++) {
            for (int kw = 0; kw < K_w; kw++) {
              int h_in = (int)(h_out * stride + kh - pad);
              int w_in = (int)(w_out * stride + kw - pad);
              if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                mt_float x_val =
                    x->data[c_in * H_in * W_in + h_in * W_in + w_in];
                mt_float w_val = w->data[c_out * C_in * K_h * K_w +
                                         c_in * K_h * K_w + kh * K_w + kw];
                sum += x_val * w_val;
              }
            }
          }
        }

        // Add bias
        sum += b->data[c_out];

        // Set output value with direct indexing
        output->data[c_out * H_out * W_out + h_out * W_out + w_out] = sum;
      }
    }
  }
#endif

  return output;
}

#ifdef MT_USE_BLAS
  #include <cblas.h>
#endif

static mt_tensor *mt__matmul_backend(mt_tensor *a, mt_tensor *b) {
  int m = a->shape[0];
  int n = b->shape[1];
  int k = a->shape[1];

  mt_tensor *c = mt_tensor_alloc(MT_ARR_INT(m, n), 2);

#ifdef MT_USE_BLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a->data,
              k, b->data, n, 0.0f, c->data, n);
#else
  #ifdef USE_OPENMP
    #pragma omp parallel for
  #endif
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int p = 0; p < k; p++) {
        sum += a->data[i * k + p] * b->data[p * n + j];
      }
      c->data[i * n + j] = sum;
    }
  }
#endif

  return c;
}

mt_tensor *mt_matmul(mt_tensor *a, mt_tensor *b) {
  assert(a->ndim == 2);
  assert(b->ndim == 2);
  assert(a->shape[1] == b->shape[0]);

  return mt__matmul_backend(a, b);
}

mt_tensor *mt_tensor_alloc(int *shape, int ndim) {
  assert(ndim <= MAX_NDIM);

  mt_tensor *t  = (mt_tensor *)calloc(1, sizeof(*t));
  t->ndim       = ndim;
  t->count_deps = 0;
  memcpy(t->shape, shape, ndim * sizeof(*shape));

  int numel = mt__tensor_count_element(t);
  t->data   = (mt_float *)calloc(numel, sizeof(mt_float));

  return t;
}

mt_tensor *mt_tensor_alloc_fill(int *shape, int ndim, mt_float value) {
  mt_tensor *t     = mt_tensor_alloc(shape, ndim);
  int        numel = mt__tensor_count_element(t);
  for (int i = 0; i < numel; ++i) {
    t->data[i] = value;
  }
  return t;
}

mt_tensor *mt_tensor_alloc_values(int *shape, int ndim, mt_float *values) {
  mt_tensor *t     = mt_tensor_alloc(shape, ndim);
  int        numel = mt__tensor_count_element(t);
  for (int i = 0; i < numel; ++i) {
    t->data[i] = values[i];
  }
  return t;
}

mt_tensor *mt_tensor_alloc_random(int *shape, int ndim) {
  mt_tensor *t     = mt_tensor_alloc(shape, ndim);
  int        numel = mt__tensor_count_element(t);
  for (int i = 0; i < numel; ++i) {
    t->data[i] = (mt_float)rand() / (mt_float)(RAND_MAX);
  }
  return t;
}

void mt_tensor_free(mt_tensor *t) {
  free(t->data);
  free(t);
}

#endif // !_MINT_H_
