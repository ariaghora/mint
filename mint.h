/* mint - Minimalist tensor library

 ============================================================================
 */

#ifndef _MINT_H_
#define _MINT_H_

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_NDIM 5
#define MAX_GLOBAL_TENSOR_COUNT 1000

// The tensor values data type
#define mt_float float

// The tensor representation
typedef struct mt_tensor {
  mt_float *data;

  int ndim;
  int shape[MAX_NDIM];
  int count_deps;
} mt_tensor;

mt_tensor *mt_adaptive_avg_pool_2d(mt_tensor *x, int out_h, int out_w);
mt_tensor *mt_affine(mt_tensor *x, mt_tensor *w, mt_tensor *b);
mt_tensor *mt_convolve_2d(mt_tensor *x, mt_tensor *w, mt_tensor *b, int stride,
                          int pad);
void       mt_image_standardize(mt_tensor *t, mt_float *mu, mt_float *std);
mt_tensor *mt_matmul(mt_tensor *a, mt_tensor *b);
mt_tensor *mt_max_pool_2d(mt_tensor *x, mt_tensor *w, mt_tensor *b, int stride,
                          int pad);
void       mt_relu_inplace(mt_tensor *t);
void       mt_reshape_inplace(mt_tensor *t, int *new_shape, int new_ndim);
mt_tensor *mt_tensor_alloc(int *shape, int ndim);
mt_tensor *mt_tensor_alloc_fill(int *shape, int ndim, mt_float value);
mt_tensor *mt_tensor_alloc_values(int *shape, int ndim, mt_float *values);
mt_tensor *mt_tensor_alloc_random(int *shape, int ndim);
mt_tensor *mt_tensor_fread(FILE *fp);
void       mt_tensor_free(mt_tensor *t);
mt_tensor *mt_tensor_load_image(char *filename);

/**
 *
 * IMPLEMENTATION
 *
 */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MT_ARR_INT(...) ((int[]){__VA_ARGS__})
#define MT_ARR_FLOAT(...) ((mt_float[]){__VA_ARGS__})

#ifdef NDEBUG
  #define MT_ASSERT(condition, format, ...) ((void)0)
#else
  #define MT_ASSERT(condition, format, ...)                                    \
    do {                                                                       \
      if (!(condition)) {                                                      \
        fprintf(stderr, "\x1b[31m");                                           \
        fprintf(stderr, "Assertion failed [%s:%d]: %s\n", __FILE__, __LINE__,  \
                #condition);                                                   \
        fprintf(stderr, format, __VA_ARGS__);                                  \
        fprintf(stderr, "\n");                                                 \
        fprintf(stderr, "\x1b[0m");                                            \
        abort();                                                               \
      }                                                                        \
    } while (0)
#endif

int mt__tensor_count_element(mt_tensor *t) {
  int count = 1;
  for (int i = 0; i < t->ndim; i++) {
    count *= t->shape[i];
  }
  return count;
}

void mt__im2row(mt_tensor *t, mt_float *row_data, int K_h, int K_w, int stride,
                int pad) {
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

mt_tensor *mt_adaptive_avg_pool_2d(mt_tensor *x, int out_h, int out_w) {
  assert(x->ndim == 3); // Ensure input is a 3D tensor (C, H, W)

  int   channels = x->shape[0];
  int   in_h     = x->shape[1];
  int   in_w     = x->shape[2];
  float stride_h = (float)in_h / out_h;
  float stride_w = (float)in_w / out_w;

  // Allocate output tensor
  mt_tensor *output = mt_tensor_alloc(MT_ARR_INT(channels, out_h, out_w), 3);

  for (int c = 0; c < channels; c++) {
    for (int oh = 0; oh < out_h; oh++) {
      for (int ow = 0; ow < out_w; ow++) {
        int h_start = (int)(oh * stride_h);
        int w_start = (int)(ow * stride_w);
        int h_end   = (int)((oh + 1) * stride_h);
        int w_end   = (int)((ow + 1) * stride_w);

        mt_float sum   = 0.0;
        int      count = 0;

        for (int ih = h_start; ih < h_end; ih++) {
          for (int iw = w_start; iw < w_end; iw++) {
            sum += x->data[c * (in_h * in_w) + ih * in_w + iw];
            count++;
          }
        }

        int didx           = c * (out_h * out_w) + oh * out_w + ow;
        output->data[didx] = sum / count;
      }
    }
  }

  return output;
}

mt_tensor *mt_affine(mt_tensor *x, mt_tensor *w, mt_tensor *b) {
  MT_ASSERT(b->ndim == 1, "`b` must be of dimension 2, found %d", b->ndim);
  MT_ASSERT(w->shape[1] == b->shape[0],
            "Width of `w` (%d) must match length of `b` (%d)", w->shape[1],
            b->shape[0]);

  mt_tensor *res = mt_matmul(x, w);

  // add bias
  int batch_size  = res->shape[0];
  int output_size = res->shape[1];

  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < output_size; j++) {
      res->data[i * output_size + j] += b->data[j];
    }
  }

  return res;
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
  // Create im2col matrix
  int        im2col_rows = C_in * K_h * K_w;
  int        im2col_cols = H_out * W_out;
  mt_tensor *im2col = mt_tensor_alloc(MT_ARR_INT(im2col_rows, im2col_cols), 2);

  // Perform im2col operation
  #ifdef MT_USE_OPENMP
    #pragma omp parallel for collapse(2)
  #endif
  for (int i = 0; i < im2col_cols; i++) {
    for (int j = 0; j < im2col_rows; j++) {
      int w_out = i % W_out;
      int h_out = (i / W_out) % H_out;
      int c_in  = j / (K_h * K_w);
      int k_h   = (j / K_w) % K_h;
      int k_w   = j % K_w;

      int h_in = h_out * stride + k_h - pad;
      int w_in = w_out * stride + k_w - pad;

      if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
        im2col->data[j * im2col_cols + i] =
            x->data[c_in * H_in * W_in + h_in * W_in + w_in];
      } else {
        im2col->data[j * im2col_cols + i] = 0;
      }
    }
  }

  // Reshape weights
  mt_tensor *reshaped_w =
      mt_tensor_alloc(MT_ARR_INT(C_out, C_in * K_h * K_w), 2);
  memcpy(reshaped_w->data, w->data,
         C_out * C_in * K_h * K_w * sizeof(mt_float));

  // Perform matrix multiplication
  mt_tensor *output_2d = mt_matmul(reshaped_w, im2col);

  // Reshape output and add bias
  mt_tensor *output = mt_tensor_alloc(MT_ARR_INT(C_out, H_out, W_out), 3);
  #ifdef MT_USE_OPENMP
    #pragma omp parallel for collapse(3)
  #endif
  for (int c = 0; c < C_out; c++) {
    for (int h = 0; h < H_out; h++) {
      for (int w = 0; w < W_out; w++) {
        int idx = c * H_out * W_out + h * W_out + w;
        output->data[idx] =
            output_2d->data[c * H_out * W_out + h * W_out + w] + b->data[c];
      }
    }
  }

  // Free temporary tensors
  mt_tensor_free(im2col);
  mt_tensor_free(reshaped_w);
  mt_tensor_free(output_2d);

  return output;
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

mt_tensor *mt__matmul_backend(mt_tensor *a, mt_tensor *b) {
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

void mt_image_standardize(mt_tensor *t, mt_float *mu, mt_float *std) {
  assert(t->ndim == 3);
  int h = t->shape[1];
  int w = t->shape[2];
  for (int c = 0; c < 3; ++c) {
    for (int row = 0; row < h; ++row) {
      for (int col = 0; col < w; ++col) {
        int index      = c * h * w + row * w + col;
        t->data[index] = (t->data[index] - mu[c]) / std[c];
      }
    }
  }
}

mt_tensor *mt_matmul(mt_tensor *a, mt_tensor *b) {
  assert(a->ndim == 2);
  assert(b->ndim == 2);
  assert(a->shape[1] == b->shape[0]);

  return mt__matmul_backend(a, b);
}

mt_tensor *mt_maxpool_2d(mt_tensor *x, int kernel_size, int stride, int pad) {
  assert(x->ndim == 3);

  int C    = x->shape[0];
  int H_in = x->shape[1];
  int W_in = x->shape[2];

  // Calculate output dimensions with padding
  int H_out = (H_in + 2 * pad - kernel_size) / stride + 1;
  int W_out = (W_in + 2 * pad - kernel_size) / stride + 1;

  // Allocate output tensor
  mt_tensor *output = mt_tensor_alloc(MT_ARR_INT(C, H_out, W_out), 3);

#ifdef MT_USE_OPENMP
  #pragma omp parallel for
#endif
  for (int c = 0; c < C; c++) {
    for (int h_out = 0; h_out < H_out; h_out++) {
      for (int w_out = 0; w_out < W_out; w_out++) {
        mt_float max_val = -INFINITY;

        for (int kh = 0; kh < kernel_size; kh++) {
          for (int kw = 0; kw < kernel_size; kw++) {
            int h_in = h_out * stride + kh - pad;
            int w_in = w_out * stride + kw - pad;

            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
              mt_float val = x->data[c * H_in * W_in + h_in * W_in + w_in];
              if (val > max_val) {
                max_val = val;
              }
            }
          }
        }

        // Set output value with direct indexing
        output->data[c * H_out * W_out + h_out * W_out + w_out] = max_val;
      }
    }
  }

  return output;
}

void mt_relu_inplace(mt_tensor *t) {
  // in-place relu activation
#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
  for (int i = 0; i < mt__tensor_count_element(t); ++i) {
    t->data[i] = t->data[i] >= 0 ? t->data[i] : 0;
  }
}

void mt_reshape_inplace(mt_tensor *t, int *new_shape, int new_ndim) {
  int tensor_old_element_len = mt__tensor_count_element(t);

  // zero-out old shape for the sake of safety
  for (int i = 0; i < t->ndim; ++i)
    t->shape[i] = 0;

  int tensor_new_element_len = 1;
  for (int i = 0; i < new_ndim; ++i) {
    tensor_new_element_len *= new_shape[i];
    t->shape[i] = new_shape[i];
  }

  MT_ASSERT(tensor_old_element_len == tensor_new_element_len,
            "tensor with length %d cannot be reshaped into length of %d",
            tensor_old_element_len, tensor_new_element_len);
  t->ndim = new_ndim;
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

mt_tensor *mt_tensor_fread(FILE *fp) {
  int *ndim = (int *)calloc(1, sizeof(int));
  fread(ndim, sizeof(int), 1, fp);

  int *shape = (int *)calloc(*ndim, sizeof(int));
  fread(shape, sizeof(int), *ndim, fp);

  int tensor_numel = 1;
  for (int i = 0; i < *ndim; i++)
    tensor_numel *= shape[i];
  mt_float *values = (mt_float *)calloc(tensor_numel, sizeof(mt_float));
  fread(values, sizeof(mt_float), tensor_numel, fp);

  mt_tensor *t = mt_tensor_alloc_values(shape, *ndim, values);

  free(ndim), free(shape), free(values);
  return t;
}

void mt_tensor_free(mt_tensor *t) {
  free(t->data);
  free(t);
}

#ifdef MT_USE_STB_IMAGE
mt_tensor *mt_tensor_load_image(char *filename) {
  int            w, h, c;
  unsigned char *data = stbi_load(filename, &w, &h, &c, 0);
  if (data == NULL) {
    printf("ERROR: cannot load %s\n", filename);
    exit(1);
  }

  mt_tensor *t = mt_tensor_alloc(MT_ARR_INT(c, h, w), 3);

  // HWC to CHW, normalize to 0-1 range
  for (int row = 0; row < h; row++) {
    for (int col = 0; col < w; col++) {
      for (int chan = 0; chan < c; chan++) {
        t->data[chan * h * w + row * w + col] =
            (mt_float)data[(row * w + col) * c + chan] / 255.0;
      }
    }
  }

  stbi_image_free(data);
  return t;
}
#endif

#ifdef __cplusplus
}
#endif // !__cplusplus
#endif // !_MINT_H_
