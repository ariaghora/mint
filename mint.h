/*

                                M I N T

                       A minimalist tensor library


  Mint is a single-file header only library for tensor manipulation. It also
enables importing and executing *some* of neural net models. Mint aims to be
dependency-free and easily distributed, but it is possible to integrate with
the other libraries such as BLAS if needed.

  Some of notable features:
- NumPy style broadcasting
- BLAS backend (optional)
- OpenMP acceleration (optional)


****************************************************************************
  TABLE OF CONTENTS                                                    MT001
****************************************************************************

    [MT001] Table of contents
    [MT002] Usage
    [MT003] Concepts
    [MT004] Compile-time options
    [MT005] Mint APIs
    [MT006] Mint implementations

    Tips: you can search faster within this file using the code MTXXX


****************************************************************************
  USAGE                                                                MT002
****************************************************************************

  Do this:

  > #define MT_IMPLEMENTATION

  before you include this file in *one* C or C++ file. For example:

  > #include ...
  > #include ...
  > #include ...
  >
  > #define MT_IMPLEMENTATION
  > #include "mint.h"


****************************************************************************
  CONCEPTS                                                             MT003
****************************************************************************

  Tensor (mt_tensor)
  --------------------------------------------------------------------------
  A multi-dimensional matrix representation. The data type by default is set
to C float, defined as `mt_float`. This can be overridden easily like this:

  > #define mt_float YOUR_FLOAT
  > #include "mint.h"
  >
  > // ...rest of your code


  Model (mt_model)
  --------------------------------------------------------------------------
  Mint provides a functionality to load a pretrained model. Currently it can
only load models converted from ONNX format. The script to convert ONNX into
*.mt (Mint model format) is provided in `scripts` directory. The mint format
is specified as follows:

         ┌──────────────────────────────────────────────────────────┐
         │ MODEL HEADER                                             │
         │┌────────────────────────────────────────────────────────┐│
         ││ 4 bytes: Layer Count                                   ││
         │├────────────────────────────────────────────────────────┤│
         ││ 4 bytes: Tensor Count                                  ││
         │└────────────────────────────────────────────────────────┘│
         ├──────────────────────────────────────────────────────────┤
         │ MODEL DATA                                               │
         │┌────────────────────────────────────────────────────────┐│
         ││ LAYER HEADER                                           ││
         ││┌──────────────────────────────────────────────────────┐││
         │││ 4 bytes: layer kind                                  │││
         │││ 4 bytes: layer ID                                    │││
         │││ 4 bytes: prev_count, count of dependencies           │││
         │││ 4 * prev_count bytes: list of dependency IDs         │││
         │││ 4 bytes: next_count, count of dependents             │││
         │││ 4 * next_count bytes: list of dependent ID           │││
         │││ 4 bytes: input_count, count of input tensors         │││
         │││ 4 * input_count bytes: list of input tensor IDs      │││
         │││ 4 bytes: output_count, count of output tensors       │││
         │││ 4 * output_count bytes: list of output tensor IDs    │││
         ││└──────────────────────────────────────────────────────┘││
         │├────────────────────────────────────────────────────────┤│
         ││ LAYER DATA                                             ││
         ││            Content depends on layer kind               ││
         │└────────────────────────────────────────────────────────┘│
         │                          . . .                           │
         │┌────────────────────────────────────────────────────────┐│
         ││ LAYER HEADER                                           ││
         │├────────────────────────────────────────────────────────┤│
         ││ LAYER DATA                                             ││
         │└────────────────────────────────────────────────────────┘│
         ├──────────────────────────────────────────────────────────┤
         │┌─────────────┐┌─────────────┐             ┌─────────────┐│
         ││ TENSOR DATA ││ TENSOR DATA │    . . .    │ TENSOR DATA ││
         │└─────────────┘└─────────────┘             └─────────────┘│
         └──────────────────────────────────────────────────────────┘


****************************************************************************
  COMPILE-TIME OPTIONS                                                 MT004
****************************************************************************

  All compile-time options should be placed before including `mint.h`.


  NDEBUG
  --------------------------------------------------------------------------
  When enabled, assertion and debug logging will be disabled.


  MT_USE_STB_IMAGE
  --------------------------------------------------------------------------
  Whether or not to use image loading functionality supported by stb_image.h
for example, mt_tensor_load_image. If you enable this, then you must include
`stb_image.h` BEFORE incuding `mint.h`. For example:

  > #define STB_IMAGE_IMPLEMENTATION
  > #include "stb_image.h"
  >
  > #define MT_USE_STB_IMAGE
  > #include "mint.h"
  >
  > // ...rest of your code
  >


  MT_USE_BLAS
  --------------------------------------------------------------------------
  Whether or not to use BLAS. This will accelerate some operations involving
matrix multiplication. You must also include `cblas.h` yourself right before
including `mint.h`. You need to link your program with BLAS by adding -lblas
compiler flag.


  MT_USE_IM2COL_CONV
  --------------------------------------------------------------------------
  When enabled, mint will use im2col operation to convert the input data and
the convolution kernels into matrices. Then, instead of performing a regular
convolution, mint will treat convolution operation to be performed as a more
efficient matrix multiplication. This way, we effectively unroll convolution
operation.

  More on this: https://arxiv.org/pdf/1410.0759

  Notes:
1. It is recommended to enable this flag along with MT_USE_BLAS flag for the
   optimal result.
2. Convolution by matrix multiplication with im2col requires more memory.

                                                                             */

#ifndef _MINT_H_
#define _MINT_H_

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// The tensor values data type
#ifndef mt_float
#define mt_float float
#endif

/***************************************************************************
  MINT APIs                                                            MT005
 **************************************************************************/

/*
 * Tensor operation API
 */

typedef struct mt_tensor mt_tensor;

// Adaptive version of average pooling. This typically allows the use of any
// arbitrary input size to obtain consistent size for the intermediate layer
// representation.
mt_tensor *mt_adaptive_avg_pool_2d(mt_tensor *x, int out_h, int out_w);
// Element-wise addition
mt_tensor *mt_add(mt_tensor *a, mt_tensor *b);
// Affine transformation, i.e., matmul(x, w) + b. Tensor b needs to have one
// dimension with length of matmul result's trailing dimension. The addition
// operation will broadcast b along matmul(x, w)'s first dimension.
mt_tensor *mt_affine(mt_tensor *x, mt_tensor *w, mt_tensor *b);
// Average pooling
mt_tensor *mt_avg_pool_2d(mt_tensor *x, int kernel_size, int stride, int pad);
// Convolution 2d
mt_tensor *mt_convolve_2d(mt_tensor *x, mt_tensor *w, mt_tensor *b, int stride,
                          int pad);
// Element-wise division
mt_tensor *mt_div(mt_tensor *a, mt_tensor *b);
// Pooling by taking average of each channel, reducing each channel's matrix
// into a single value, i.e., the mean.
mt_tensor *mt_global_avg_pool_2d(mt_tensor *x);
// Standardize tensor RGB image. Both mu and std must have 3 elements.
void       mt_image_standardize(mt_tensor *t, mt_float *mu, mt_float *std);
// Local response norm, as introduced in AlexNet paper
mt_tensor *mt_local_response_norm(mt_tensor *t, int size, mt_float alpha,
                                  mt_float beta, mt_float k);
// Matrix multiplication. Both a and b must have 2 dimensions.
mt_tensor *mt_matmul(mt_tensor *a, mt_tensor *b);
// Max-pooling
mt_tensor *mt_maxpool_2d(mt_tensor *x, int kernel_size, int stride, int pad);
// Element-wise multiplication
mt_tensor *mt_mul(mt_tensor *a, mt_tensor *b);
// Relu activation function, in-place version.
void       mt_relu_inplace(mt_tensor *t);
// Element-wise subtraction
mt_tensor *mt_sub(mt_tensor *a, mt_tensor *b);

/*
 * Tensor memory management API
 */
// Allocate tensor without value initialization
mt_tensor *mt_tensor_alloc(int *shape, int ndim);
// Allocate tensor and fill the data with a consant value
mt_tensor *mt_tensor_alloc_fill(int *shape, int ndim, mt_float value);
// Allocate tensor and fill the data with a specified array of values
mt_tensor *mt_tensor_alloc_values(int *shape, int ndim, mt_float *values);
// Allocate tensor and fill the data with random values, ranging from 0 to 1
mt_tensor *mt_tensor_alloc_random(int *shape, int ndim);
// Get the number of elements of a tensor
int        mt_tensor_count_element(mt_tensor *t);
// Helper function to print tensor summary
void       mt_tensor_debug_info(mt_tensor *t);
// Helper function to read tensor from a file pointer. The file need to obey
// a certain format.
mt_tensor *mt_tensor_fread(FILE *fp);
// Free tensor
void       mt_tensor_free(mt_tensor *t);
// Load image as a tensor with shape of CxHxW. C is the number of channel, H
// is the image height, and W is the image width.
mt_tensor *mt_tensor_load_image(char *filename);
// Reshape tensor in-place. The old and new shape should be compatible.
void mt_tensor_reshape_inplace(mt_tensor *t, int *new_shape, int new_ndim);

/*
 * Model API
 */

typedef struct mt_model mt_model;

typedef enum {
    MT_LAYER_UNKNOWN,
    MT_LAYER_ADD,
    MT_LAYER_AVG_POOL_2D,
    MT_LAYER_CONV_2D,
    MT_LAYER_DENSE,
    MT_LAYER_DIV,
    MT_LAYER_FLATTEN,
    MT_LAYER_GLOBAL_AVG_POOL,
    MT_LAYER_LOCAL_RESPONSE_NORM,
    MT_LAYER_MAX_POOL_2D,
    MT_LAYER_MUL,
    MT_LAYER_RELU,
    MT_LAYER_SIGMOID,
    MT_LAYER_SUB,
} mt_layer_kind;

mt_model  *mt_model_load(const char *filename, int input_in_batch);
void       mt_model_free(mt_model *model);
mt_tensor *mt_model_get_output(mt_model *model, const char *name);
void       mt_model_run(mt_model *model);
void       mt_model_set_input(mt_model *model, const char *name, mt_tensor *t);

/***************************************************************************
  MINT IMPLEMENTATION                                                  MT006
 **************************************************************************/

#ifdef MT_IMPLEMENTATION

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LAYER_COUNT 1000
#define MAX_LAYER_INPUT_COUNT 10
#define MAX_LAYER_OUTPUT_COUNT 10
#define MAX_LAYER_PREV_COUNT 5
#define MAX_LAYER_NEXT_COUNT 5
#define MAX_MODEL_INITIALIZER_COUNT 1000
#define MAX_TENSOR_NDIM 5
#define MAX_INPUT_OUTPUT_COUNT 5
#define MAX_INPUT_OUTPUT_NAME_LEN 50

typedef struct mt_tensor {
    mt_float *data;

    int ndim;
    int shape[MAX_TENSOR_NDIM];
    int count_deps;
} mt_tensor;

typedef struct {
    int           id;
    mt_layer_kind kind;
    // This member holds data of different layer types. Some layers do not have
    // any data/attribute to store, such as ReLU, simple binary operations, etc.
    // In that case, they are not listed here.
    union {
        // MT_LAYER_AVG_POOL_2D
        struct {
            int size;
            int stride;
            int pad;
        } avg_pool_2d;

        // MT_LAYER_CONV_2D
        struct {
            int w_id;
            int b_id;
            int stride;
            int pad;
        } conv_2d;

        // MT_LAYER_DENSE
        struct {
            int w_id;
            int b_id;
        } dense;

        // MT_LAYER_FLATTEN
        struct {
            int axis;
        } flatten;

        // MT_LAYER_LOCAL_RESPONSE_NORM
        struct {
            int      size;
            mt_float alpha;
            mt_float beta;
            mt_float bias;
        } local_response_norm;

        // MT_LAYER_MAX_POOL_2D
        struct {
            int size;
            int stride;
            int pad;
        } max_pool_2d;
    } data;
    int prev_count;
    int prev[MAX_LAYER_PREV_COUNT];
    int next_count;
    int next[MAX_LAYER_NEXT_COUNT];
    int input_count;
    int inputs[MAX_LAYER_INPUT_COUNT];
    int output_count;
    int outputs[MAX_LAYER_OUTPUT_COUNT];
} mt_layer;

typedef struct mt_model {
    int        layer_count;
    int        tensor_count;
    mt_layer  *layers[MAX_LAYER_COUNT];
    mt_tensor *tensors[MAX_MODEL_INITIALIZER_COUNT];
    int        input_count;
    struct {
        int  id;
        char name[MAX_INPUT_OUTPUT_NAME_LEN];
    } inputs[MAX_INPUT_OUTPUT_COUNT];
    int output_count;
    struct {
        int  id;
        char name[MAX_INPUT_OUTPUT_COUNT];
    } outputs[10];
} mt_model;

#define MT_ARR_INT(...) ((int[]){__VA_ARGS__})
#define MT_ARR_FLOAT(...) ((mt_float[]){__VA_ARGS__})

#ifdef NDEBUG
#define MT_ASSERT_F(condition, format, ...) ((void)0)
#define DEBUG_LOG_F(format, ...) ((void)0)
#else
#define MT_ASSERT_F(condition, format, ...)                                    \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "\x1b[31m");                                       \
            fprintf(stderr, "Assertion failed [%s:%d]: %s\n", __FILE__,        \
                    __LINE__, #condition);                                     \
            fprintf(stderr, format, __VA_ARGS__);                              \
            fprintf(stderr, "\n");                                             \
            fprintf(stderr, "\x1b[0m");                                        \
            abort();                                                           \
        }                                                                      \
    } while (0)

#define DEBUG_LOG_F(format, ...)                                               \
    do {                                                                       \
        fprintf(stderr, "\x1b[33m");                                           \
        fprintf(stderr, "DEBUG [%s:%d]: ", __FILE__, __LINE__);                \
        fprintf(stderr, format, __VA_ARGS__);                                  \
        fprintf(stderr, "\x1b[0m\n");                                          \
    } while (0)
#endif

#ifdef NDEBUG
#define MT_ASSERT(condition, msg) ((void)0)
#else
#define MT_ASSERT(condition, msg)                                              \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "\x1b[31m");                                       \
            fprintf(stderr, "Assertion failed [%s:%d]: %s\n", __FILE__,        \
                    __LINE__, #condition);                                     \
            fprintf(stderr, msg);                                              \
            fprintf(stderr, "\n");                                             \
            fprintf(stderr, "\x1b[0m");                                        \
            abort();                                                           \
        }                                                                      \
    } while (0)
#endif

#define ERROR(msg)                                                             \
    do {                                                                       \
        fprintf(stderr,                                                        \
                "\x1b[31m"                                                     \
                "[ERROR] %s: %s\n"                                             \
                "\x1b[0m",                                                     \
                msg, strerror(errno));                                         \
        exit(1);                                                               \
    } while (0)

#define ERROR_F(fmt, ...)                                                      \
    do {                                                                       \
        fprintf(stderr,                                                        \
                "\x1b[31m"                                                     \
                "[ERROR] " fmt ": %s\n"                                        \
                "\x1b[0m",                                                     \
                __VA_ARGS__, strerror(errno));                                 \
        exit(1);                                                               \
    } while (0)

int mt_tensor_count_element(mt_tensor *t) {
    int count = 1;
    for (int i = 0; i < t->ndim; i++) {
        count *= t->shape[i];
    }
    return count;
}

mt_tensor *mt_adaptive_avg_pool_2d(mt_tensor *x, int out_h, int out_w) {
    MT_ASSERT_F(x->ndim == 3,
                "input tensor must have 3 dimensions (an image), found %d",
                x->ndim);

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

// Helper function to calculate the broadcasted shape
void mt__calc_broadcast_shape(int *shape1, int ndim1, int *shape2, int ndim2,
                              int *result_shape, int *result_ndim) {
    *result_ndim = (ndim1 > ndim2) ? ndim1 : ndim2;

    for (int i = 0; i < *result_ndim; i++) {
        int dim1 =
            (i < *result_ndim - ndim1) ? 1 : shape1[i - (*result_ndim - ndim1)];
        int dim2 =
            (i < *result_ndim - ndim2) ? 1 : shape2[i - (*result_ndim - ndim2)];

        if (dim1 == dim2) {
            result_shape[i] = dim1;
        } else if (dim1 == 1 || dim2 == 1) {
            result_shape[i] = (dim1 > dim2) ? dim1 : dim2;
        } else {
            fprintf(stderr, "Shapes are not compatible for broadcasting\n");
            exit(1);
        }
    }
}

static void mt__calc_strides(int *shape, int ndim, int *strides) {
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

// General binary operator.
// NOTE (Aria): This is meant to be used internally.
mt_tensor *mt__binop(mt_tensor *a, mt_tensor *b,
                     mt_float f(mt_float, mt_float)) {
    int result_shape[MAX_TENSOR_NDIM];
    int result_ndim;
    mt__calc_broadcast_shape(a->shape, a->ndim, b->shape, b->ndim, result_shape,
                             &result_ndim);

    mt_tensor *result = mt_tensor_alloc(result_shape, result_ndim);

    int a_strides[MAX_TENSOR_NDIM], b_strides[MAX_TENSOR_NDIM],
        result_strides[MAX_TENSOR_NDIM];

    mt__calc_strides(a->shape, a->ndim, a_strides);
    mt__calc_strides(b->shape, b->ndim, b_strides);
    mt__calc_strides(result_shape, result_ndim, result_strides);

    int a_broadcast_shape[MAX_TENSOR_NDIM], b_broadcast_shape[MAX_TENSOR_NDIM];
    for (int i = 0; i < result_ndim; i++) {
        a_broadcast_shape[i] = (i < result_ndim - a->ndim)
                                   ? 1
                                   : a->shape[i - (result_ndim - a->ndim)];
        b_broadcast_shape[i] = (i < result_ndim - b->ndim)
                                   ? 1
                                   : b->shape[i - (result_ndim - b->ndim)];
    }

    int numel = mt_tensor_count_element(result);

#pragma omp parallel for
    // TODO: optimize for special ndims, identic shape, and tensor-scalar ops
    for (int i = 0; i < numel; i++) {
        int indices[MAX_TENSOR_NDIM];
        int temp = i;
        for (int j = 0; j < result_ndim; j++) {
            indices[j] = temp / result_strides[j];
            temp %= result_strides[j];
        }

        int a_index = 0, b_index = 0;
        for (int j = 0; j < result_ndim; j++) {
            a_index += (indices[j] % a_broadcast_shape[j]) *
                       (j < result_ndim - a->ndim
                            ? 0
                            : a_strides[j - (result_ndim - a->ndim)]);
            b_index += (indices[j] % b_broadcast_shape[j]) *
                       (j < result_ndim - b->ndim
                            ? 0
                            : b_strides[j - (result_ndim - b->ndim)]);
        }

        result->data[i] = f(a->data[a_index], b->data[b_index]);
    }

    return result;
}

static mt_float mt__s_add(mt_float a, mt_float b) { return a + b; }
mt_tensor      *mt_add(mt_tensor *a, mt_tensor *b) {
    return mt__binop(a, b, mt__s_add);
}

mt_tensor *mt_affine(mt_tensor *x, mt_tensor *w, mt_tensor *b) {
    MT_ASSERT_F(b->ndim == 1, "`b` must be of dimension 2, found %d", b->ndim);
    MT_ASSERT_F(w->shape[1] == b->shape[0],
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

mt_tensor *mt_avg_pool_2d(mt_tensor *x, int kernel_size, int stride, int pad) {
    MT_ASSERT(x->ndim == 3, "Input tensor must be 3-dimensional");
    int C    = x->shape[0];
    int H_in = x->shape[1];
    int W_in = x->shape[2];

    // Calculate output dimensions with padding
    int H_out = (H_in + 2 * pad - kernel_size) / stride + 1;
    int W_out = (W_in + 2 * pad - kernel_size) / stride + 1;

    // Allocate output tensor
    mt_tensor *output = mt_tensor_alloc(MT_ARR_INT(C, H_out, W_out), 3);

#pragma omp parallel for collapse(3)
    for (int c = 0; c < C; c++) {
        for (int h_out = 0; h_out < H_out; h_out++) {
            for (int w_out = 0; w_out < W_out; w_out++) {
                mt_float sum   = 0.0f;
                int      count = 0;

                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int h_in = h_out * stride + kh - pad;
                        int w_in = w_out * stride + kw - pad;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 &&
                            w_in < W_in) {
                            sum +=
                                x->data[c * H_in * W_in + h_in * W_in + w_in];
                            count++;
                        }
                    }
                }

                // Calculate average and set output value
                mt_float avg = (count > 0) ? sum / count : 0.0f;
                output->data[c * H_out * W_out + h_out * W_out + w_out] = avg;
            }
        }
    }

    return output;
}

mt_tensor *mt_convolve_2d(mt_tensor *x, mt_tensor *w, mt_tensor *b, int stride,
                          int pad) {
    MT_ASSERT(x->ndim == 3, "");
    MT_ASSERT(w->ndim == 4, "");
    MT_ASSERT(b->ndim == 1, "");
    MT_ASSERT(x->shape[0] == w->shape[1], "");
    MT_ASSERT(b->shape[0] == w->shape[0], "");

    int C_in = x->shape[0];
    int H_in = x->shape[1];
    int W_in = x->shape[2];

    int C_out = w->shape[0];
    int K_h   = w->shape[2];
    int K_w   = w->shape[3];

    // Calculate output dimensions with padding
    int H_out = (H_in + 2 * pad - K_h) / stride + 1;
    int W_out = (W_in + 2 * pad - K_w) / stride + 1;

#ifdef MT_USE_IM2COL_CONV
    // Create im2col matrix
    int        im2col_rows = C_in * K_h * K_w;
    int        im2col_cols = H_out * W_out;
    mt_tensor *im2col =
        mt_tensor_alloc(MT_ARR_INT(im2col_rows, im2col_cols), 2);

#pragma omp parallel for collapse(2)
    // Perform im2col operation
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
#pragma omp parallel for collapse(3)
    for (int c = 0; c < C_out; c++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                int idx = c * H_out * W_out + h * W_out + w;
                output->data[idx] =
                    output_2d->data[c * H_out * W_out + h * W_out + w] +
                    b->data[c];
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
#pragma omp parallel for
    for (int c_out = 0; c_out < C_out; c_out++) {
        for (int h_out = 0; h_out < H_out; h_out++) {
            for (int w_out = 0; w_out < W_out; w_out++) {
                mt_float sum = 0.0f;

                for (int c_in = 0; c_in < C_in; c_in++) {
                    for (int kh = 0; kh < K_h; kh++) {
                        for (int kw = 0; kw < K_w; kw++) {
                            int h_in = (int)(h_out * stride + kh - pad);
                            int w_in = (int)(w_out * stride + kw - pad);
                            if (h_in >= 0 && h_in < H_in && w_in >= 0 &&
                                w_in < W_in) {
                                mt_float x_val = x->data[c_in * H_in * W_in +
                                                         h_in * W_in + w_in];
                                mt_float w_val =
                                    w->data[c_out * C_in * K_h * K_w +
                                            c_in * K_h * K_w + kh * K_w + kw];
                                sum += x_val * w_val;
                            }
                        }
                    }
                }

                // Add bias
                sum += b->data[c_out];

                // Set output value with direct indexing
                output->data[c_out * H_out * W_out + h_out * W_out + w_out] =
                    sum;
            }
        }
    }
#endif

    return output;
}

mt_tensor *mt__matmul_backend(mt_tensor *a, mt_tensor *b) {
    int m = a->ndim == 1 ? 1 : a->shape[0];
    int n = b->ndim == 1 ? b->shape[0] : b->shape[1];
    int k = a->ndim == 1 ? a->shape[0] : a->shape[1];

    int tda = a->ndim == 1 ? a->shape[0] : a->shape[1];
    int ldb = b->ndim == 1 ? 1 : b->shape[0];

    MT_ASSERT_F(a->ndim <= 2, "A must have <= 2 dimensions, got %d dimension",
                a->ndim);
    MT_ASSERT_F(b->ndim <= 2, "A must have <= 2 dimensions, got %d dimension",
                b->ndim);
    MT_ASSERT_F(tda == ldb,
                "incompatible shape for matrix multiplication (%d and %d)", tda,
                ldb);

    mt_tensor *c = mt_tensor_alloc(MT_ARR_INT(m, n), 2);

#ifdef MT_USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
                a->data, k, b->data, n, 0.0f, c->data, n);
#else
#pragma omp parallel for
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

static mt_float mt__s_div(mt_float a, mt_float b) { return a / b; }
mt_tensor      *mt_div(mt_tensor *a, mt_tensor *b) {
    return mt__binop(a, b, mt__s_div);
}

mt_tensor *mt_global_avg_pool_2d(mt_tensor *x) {
    MT_ASSERT(x->ndim == 3, "Input tensor must be 3-dimensional");
    int C = x->shape[0];
    int H = x->shape[1];
    int W = x->shape[2];

    // Allocate output tensor of shape (C, 1, 1)
    mt_tensor *output = mt_tensor_alloc(MT_ARR_INT(C, 1, 1), 3);

#pragma omp parallel for
    // Perform global average pooling
    for (int c = 0; c < C; c++) {
        mt_float sum = 0.0f;
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                sum += x->data[c * H * W + h * W + w];
            }
        }
        output->data[c] = sum / (H * W);
    }

    return output;
}

void mt_image_standardize(mt_tensor *t, mt_float *mu, mt_float *std) {
    MT_ASSERT(t->ndim == 3, "");
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
    return mt__matmul_backend(a, b);
}

mt_tensor *mt_maxpool_2d(mt_tensor *x, int kernel_size, int stride, int pad) {
    MT_ASSERT(x->ndim == 3, "");

    int C    = x->shape[0];
    int H_in = x->shape[1];
    int W_in = x->shape[2];

    // Calculate output dimensions with padding
    int H_out = (H_in + 2 * pad - kernel_size) / stride + 1;
    int W_out = (W_in + 2 * pad - kernel_size) / stride + 1;

    // Allocate output tensor
    mt_tensor *output = mt_tensor_alloc(MT_ARR_INT(C, H_out, W_out), 3);

#pragma omp parallel for
    for (int c = 0; c < C; c++) {
        for (int h_out = 0; h_out < H_out; h_out++) {
            for (int w_out = 0; w_out < W_out; w_out++) {
                mt_float max_val = -INFINITY;

                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int h_in = h_out * stride + kh - pad;
                        int w_in = w_out * stride + kw - pad;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 &&
                            w_in < W_in) {
                            mt_float val =
                                x->data[c * H_in * W_in + h_in * W_in + w_in];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                }

                // Set output value with direct indexing
                output->data[c * H_out * W_out + h_out * W_out + w_out] =
                    max_val;
            }
        }
    }

    return output;
}

static mt_float mt__s_mul(mt_float a, mt_float b) { return a * b; }
mt_tensor      *mt_mul(mt_tensor *a, mt_tensor *b) {
    return mt__binop(a, b, mt__s_mul);
}

void        mt_relu_inplace(mt_tensor *t) {
    // in-place relu activation
#pragma omp parallel for
    for (int i = 0; i < mt_tensor_count_element(t); ++i) {
        t->data[i] = t->data[i] >= 0 ? t->data[i] : 0;
    }
}

static mt_float mt__s_sub(mt_float a, mt_float b) { return a - b; }
mt_tensor      *mt_sub(mt_tensor *a, mt_tensor *b) {
    return mt__binop(a, b, mt__s_sub);
}

void mt_tensor_reshape_inplace(mt_tensor *t, int *new_shape, int new_ndim) {
#ifndef NDEBUG
    int tensor_old_element_len = mt_tensor_count_element(t);
#endif

    // zero-out old shape for the sake of safety
    for (int i = 0; i < t->ndim; ++i)
        t->shape[i] = 0;
    int tensor_new_element_len = 1;
    for (int i = 0; i < new_ndim; ++i) {
        tensor_new_element_len *= new_shape[i];
        t->shape[i] = new_shape[i];
    }
    MT_ASSERT_F(tensor_old_element_len == tensor_new_element_len,
                "tensor with length %d cannot be reshaped into length of %d",
                tensor_old_element_len, tensor_new_element_len);
    t->ndim = new_ndim;
}

mt_tensor *mt_tensor_alloc(int *shape, int ndim) {
    MT_ASSERT(ndim <= MAX_TENSOR_NDIM, "");

    mt_tensor *t  = (mt_tensor *)calloc(1, sizeof(*t));
    t->ndim       = ndim;
    t->count_deps = 0;
    memcpy(t->shape, shape, ndim * sizeof(*shape));

    int numel = mt_tensor_count_element(t);
    t->data   = (mt_float *)calloc(numel, sizeof(mt_float));

    return t;
}

mt_tensor *mt_tensor_alloc_fill(int *shape, int ndim, mt_float value) {
    mt_tensor *t     = mt_tensor_alloc(shape, ndim);
    int        numel = mt_tensor_count_element(t);
    for (int i = 0; i < numel; ++i) {
        t->data[i] = value;
    }
    return t;
}

mt_tensor *mt_tensor_alloc_values(int *shape, int ndim, mt_float *values) {
    mt_tensor *t     = mt_tensor_alloc(shape, ndim);
    int        numel = mt_tensor_count_element(t);
    for (int i = 0; i < numel; ++i) {
        t->data[i] = values[i];
    }
    return t;
}

mt_tensor *mt_tensor_alloc_random(int *shape, int ndim) {
    mt_tensor *t     = mt_tensor_alloc(shape, ndim);
    int        numel = mt_tensor_count_element(t);
    for (int i = 0; i < numel; ++i) {
        t->data[i] = (mt_float)rand() / (mt_float)(RAND_MAX);
    }
    return t;
}

void mt_tensor_debug_info(mt_tensor *t) {
    printf("ndim  : %d\n", t->ndim);
    printf("shape : [");
    for (int i = 0; i < t->ndim; ++i) {
        printf("%d", t->shape[i]);
        if (i < t->ndim - 1)
            printf(", ");
    }
    printf("]\n");
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
    if (t->data != NULL)
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

mt_model *mt_model_load(const char *filename, int input_in_batch) {
    FILE *fp = fopen(filename, "rb");
    MT_ASSERT_F(fp != NULL, "failed to open %s", filename);
    mt_model *model = (mt_model *)malloc(sizeof(*model));

    // First, we read model header.
    fread(&model->layer_count, sizeof(int), 1, fp);
    fread(&model->tensor_count, sizeof(int), 1, fp);
    DEBUG_LOG_F("model has %d nodes and %d tensors", model->layer_count,
                model->tensor_count);

    // Read layers and tensors
    for (int i = 0; i < model->layer_count; ++i) {
        mt_layer *layer = (mt_layer *)malloc(sizeof(*layer));

        // Read layer header
        fread(&layer->kind, sizeof(int), 1, fp);
        fread(&layer->id, sizeof(int), 1, fp);
        fread(&layer->prev_count, sizeof(int), 1, fp);
        fread(&layer->prev, sizeof(int), layer->prev_count, fp);
        fread(&layer->next_count, sizeof(int), 1, fp);
        fread(&layer->next, sizeof(int), layer->next_count, fp);
        fread(&layer->input_count, sizeof(int), 1, fp);
        fread(&layer->inputs, sizeof(int), layer->input_count, fp);
        fread(&layer->output_count, sizeof(int), 1, fp);
        fread(&layer->outputs, sizeof(int), layer->output_count, fp);

        if (layer->kind == MT_LAYER_ADD) {
            // nothing to read
        } else if (layer->kind == MT_LAYER_AVG_POOL_2D) {
            fread(&layer->data.avg_pool_2d.size, sizeof(int), 1, fp);
            fread(&layer->data.avg_pool_2d.stride, sizeof(int), 1, fp);
            fread(&layer->data.avg_pool_2d.pad, sizeof(int), 1, fp);
        } else if (layer->kind == MT_LAYER_CONV_2D) {
            fread(&layer->data.conv_2d.stride, sizeof(int), 1, fp);
            fread(&layer->data.conv_2d.pad, sizeof(int), 1, fp);
            int w_idx                = layer->inputs[1];
            layer->data.conv_2d.w_id = w_idx;
            model->tensors[w_idx]    = mt_tensor_fread(fp);
            int b_idx                = layer->inputs[2];
            layer->data.conv_2d.b_id = b_idx;
            model->tensors[b_idx]    = mt_tensor_fread(fp);
        } else if (layer->kind == MT_LAYER_DENSE) {
            int w_idx                = layer->inputs[1];
            layer->data.conv_2d.w_id = w_idx;
            model->tensors[w_idx]    = mt_tensor_fread(fp);
            int b_idx                = layer->inputs[2];
            layer->data.conv_2d.b_id = b_idx;
            model->tensors[b_idx]    = mt_tensor_fread(fp);
        } else if (layer->kind == MT_LAYER_MAX_POOL_2D) {
            fread(&layer->data.max_pool_2d.size, sizeof(int), 1, fp);
            fread(&layer->data.max_pool_2d.stride, sizeof(int), 1, fp);
            fread(&layer->data.max_pool_2d.pad, sizeof(int), 1, fp);
        } else if (layer->kind == MT_LAYER_FLATTEN) {
            fread(&layer->data.flatten.axis, sizeof(int), 1, fp);

            // if model input is in batch, axis should be decreased by one
            if (input_in_batch) {
                DEBUG_LOG_F("Input is in batch but only a single sample is "
                            "considered. Changing axis %d to %d",
                            layer->data.flatten.axis,
                            layer->data.flatten.axis - 1);
                layer->data.flatten.axis--;
            }
        } else if (layer->kind == MT_LAYER_GLOBAL_AVG_POOL) {
            // nothing to read
        } else if (layer->kind == MT_LAYER_RELU) {
            // nothing to read
        } else if (layer->kind == MT_LAYER_LOCAL_RESPONSE_NORM) {
            fread(&layer->data.local_response_norm.size, sizeof(int), 1, fp);
            fread(&layer->data.local_response_norm.alpha, sizeof(mt_float), 1,
                  fp);
            fread(&layer->data.local_response_norm.beta, sizeof(mt_float), 1,
                  fp);
            fread(&layer->data.local_response_norm.bias, sizeof(mt_float), 1,
                  fp);
        } else {
            if (layer->kind == MT_LAYER_UNKNOWN) {
                printf("unknown layer detected, possibly because its existence "
                       "was "
                       "detected but the details were not parsed\n");
            } else {
                printf("layer kind %d is not supported yet\n", layer->kind);
            }
            exit(1);
        }

        model->layers[i] = layer;
    }

    // write footer
    fread(&model->input_count, sizeof(int), 1, fp);
    for (int i = 0; i < model->input_count; ++i) {
        fread(&model->inputs[i].name, sizeof(char), MAX_INPUT_OUTPUT_NAME_LEN,
              fp);
        fread(&model->inputs[i].id, sizeof(int), 1, fp);
    }
    fread(&model->output_count, sizeof(int), 1, fp);
    for (int i = 0; i < model->output_count; ++i) {
        fread(&model->outputs[i].name, sizeof(char), MAX_INPUT_OUTPUT_NAME_LEN,
              fp);
        fread(&model->outputs[i].id, sizeof(int), 1, fp);
    }
    DEBUG_LOG_F("model graph has %d input(s)", model->input_count);
    DEBUG_LOG_F("model graph has %d output(s)", model->output_count);

    return model;
}

void mt_model_free(mt_model *model) {
    for (int i = 0; i < model->tensor_count; ++i) {
        if (model->tensors[i] != NULL)
            mt_tensor_free(model->tensors[i]);
    }
    for (int i = 0; i < model->layer_count; ++i)
        free(model->layers[i]);
    free(model);
}

int arr_int_contains(int *arr, int len, int to_find) {
    for (int i = 0; i < len; ++i) {
        if (to_find == arr[i])
            return 1;
    }
    return 0;
}

void mt__toposort(mt_layer *l, mt_model *model, int *sorted_ids,
                  int *sorted_len) {
    if (arr_int_contains(sorted_ids, *sorted_len, l->id))
        return;

    for (int i = 0; i < l->prev_count; ++i) {
        mt_layer *l_child = model->layers[l->prev[i]];
        mt__toposort(l_child, model, sorted_ids, sorted_len);
    }
    sorted_ids[*sorted_len] = l->id;
    *sorted_len             = *sorted_len + 1;
}

void mt__layer_forward(mt_layer *l, mt_model *model) {
    mt_tensor *res = NULL;
    DEBUG_LOG_F("executing layer id %d (type %d)", l->id, l->kind);

    switch (l->kind) {
    case MT_LAYER_ADD: {
        mt_tensor *a                  = model->tensors[l->inputs[0]];
        mt_tensor *b                  = model->tensors[l->inputs[1]];
        res                           = mt_add(a, b);
        model->tensors[l->outputs[0]] = res;
        break;
    }
    case MT_LAYER_AVG_POOL_2D: {
        mt_tensor *input = model->tensors[l->inputs[0]];
        res =
            mt_avg_pool_2d(input, l->data.avg_pool_2d.size,
                           l->data.avg_pool_2d.stride, l->data.avg_pool_2d.pad);
        model->tensors[l->outputs[0]] = res;
        break;
    }
    case MT_LAYER_CONV_2D: {
        mt_tensor *input = model->tensors[l->inputs[0]];
        res = mt_convolve_2d(input, model->tensors[l->data.conv_2d.w_id],
                             model->tensors[l->data.conv_2d.b_id],
                             l->data.conv_2d.stride, l->data.conv_2d.pad);
        model->tensors[l->outputs[0]] = res;
        break;
    }
    case MT_LAYER_DENSE: {
        mt_tensor *input = model->tensors[l->inputs[0]];
        res              = mt_affine(input, model->tensors[l->data.dense.w_id],
                                     model->tensors[l->data.dense.b_id]);
        model->tensors[l->outputs[0]] = res;
        break;
    }
    case MT_LAYER_FLATTEN: {
        mt_tensor *input = model->tensors[l->inputs[0]];
        res = mt_tensor_alloc_values(input->shape, input->ndim, input->data);

        int start_axis = l->data.flatten.axis;
        MT_ASSERT_F(start_axis >= 0,
                    "flatten axis should be non-negative, found %d",
                    start_axis);
        MT_ASSERT_F(start_axis <= input->ndim - 1,
                    "flatten axis (%d) is greater than input shape bound (with "
                    "ndim=%d)",
                    start_axis, input->ndim);

        // If input ndim is 4 with shape (2, 2, 2, 2) and start_axis is 1, then
        // the resulting shape will be (2, 2 * 2 * 2) and ndim will be 2. If the
        // start_axis is 2, then the resulting shape will be (2, 2, 2 * 2) and
        // ndim will be 3. Thus, out_ndim would be start_axis+1;
        int out_ndim = start_axis + 1;

        // trailing dim size will be the product of start_axis-th size till last
        // axis shape
        int trailing_dim_size = 1;
        for (int i = start_axis; i < input->ndim; ++i) {
            trailing_dim_size *= input->shape[i];
            if (i > start_axis)
                res->shape[i] = 0;
        }
        res->shape[start_axis] = trailing_dim_size;
        res->ndim              = out_ndim;

        model->tensors[l->outputs[0]] = res;
        break;
    }
    case MT_LAYER_GLOBAL_AVG_POOL: {
        mt_tensor *input              = model->tensors[l->inputs[0]];
        res                           = mt_global_avg_pool_2d(input);
        model->tensors[l->outputs[0]] = res;
        break;
    }
    case MT_LAYER_MAX_POOL_2D: {
        mt_tensor *input = model->tensors[l->inputs[0]];
        res =
            mt_maxpool_2d(input, l->data.max_pool_2d.size,
                          l->data.max_pool_2d.stride, l->data.max_pool_2d.pad);
        model->tensors[l->outputs[0]] = res;
        break;
    }
    case MT_LAYER_RELU: {
        mt_tensor *input = model->tensors[l->inputs[0]];
        res = mt_tensor_alloc_values(input->shape, input->ndim, input->data);
        mt_relu_inplace(res);
        model->tensors[l->outputs[0]] = res;
        break;
    }
    default:
        printf("Cannot execute layer with type %d yet\n", l->kind);
        exit(1);
    }
}

mt_tensor *mt_model_get_output(mt_model *model, const char *name) {
    int output_tensor_idx = -1;
    for (int i = 0; i < model->output_count; ++i) {
        if (strcmp(name, model->outputs[i].name) == 0) {
            output_tensor_idx = model->outputs[i].id;
            break;
        }
    }

    MT_ASSERT_F(output_tensor_idx != -1, "no output with name \'%s\'", name);
    mt_tensor *t      = model->tensors[output_tensor_idx];
    mt_tensor *t_copy = mt_tensor_alloc_values(t->shape, t->ndim, t->data);
    return t_copy;
}

void mt_model_run(mt_model *model) {
    int  sorted_ids[MAX_LAYER_COUNT] = {0};
    int *sorted_len_ptr = (int *)calloc(1, sizeof(*sorted_len_ptr));

    mt_layer *terminals[MAX_LAYER_COUNT];
    int       terminal_count = 0;
    // Get all terminal layers
    for (int i = 0; i < model->layer_count; ++i) {
        if (model->layers[i]->next_count == 0) {
            terminals[terminal_count++] = model->layers[i];
            DEBUG_LOG_F("found terminal layer with ID %d",
                        model->layers[i]->id);
        }
    }

    // Perform topological sort on the layers
    for (int i = 0; i < terminal_count; ++i) {
        mt_layer *l = terminals[i];
        mt__toposort(l, model, sorted_ids, sorted_len_ptr);
    }

    // Execute forward
    for (int i = 0; i < *sorted_len_ptr; ++i) {
        mt_layer *l = model->layers[sorted_ids[i]];
        mt__layer_forward(l, model);
    }

    free(sorted_len_ptr);
}

void mt_model_set_input(mt_model *model, const char *name, mt_tensor *t) {
    int input_tensor_idx = -1;
    for (int i = 0; i < model->input_count; ++i) {
        if (strcmp(name, model->inputs[i].name) == 0) {
            input_tensor_idx = model->inputs[i].id;
            break;
        }
    }

    MT_ASSERT_F(input_tensor_idx != -1, "no input with name\'%s\'", name);
    mt_tensor *t_copy = mt_tensor_alloc_values(t->shape, t->ndim, t->data);
    model->tensors[input_tensor_idx] = t_copy;
}
#endif // !__MT_IMPLEMENTATION

#ifdef __cplusplus
}
#endif // !__cplusplus
#endif // !_MINT_H_
