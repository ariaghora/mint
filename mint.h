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
// Element-wise exponentiation
mt_tensor *mt_exp(mt_tensor *a);
// Pooling by taking average of each channel, reducing each channel's matrix
// into a single value, i.e., the mean.
mt_tensor *mt_global_avg_pool_2d(mt_tensor *x);
// Resize image to a certain target using bilinear interpolation
mt_tensor *mt_image_resize(mt_tensor *t, int target_height, int target_width);
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
// Swap tensor's dimensions
mt_tensor *mt_tensor_permute_dims(mt_tensor *t, int *dims);
// Tensor slice
mt_tensor *mt_tensor_slice(mt_tensor *t, int *starts, int *ends, int *axes,
                           int *steps, int num_axes);
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
    MT_LAYER_EXP,
    MT_LAYER_FLATTEN,
    MT_LAYER_GLOBAL_AVG_POOL,
    MT_LAYER_LOCAL_RESPONSE_NORM,
    MT_LAYER_MAX_POOL_2D,
    MT_LAYER_MUL,
    MT_LAYER_RELU,
    MT_LAYER_RESHAPE,
    MT_LAYER_SIGMOID,
    MT_LAYER_SUB,
    MT_LAYER_TRANSPOSE,
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

#include <limits.h>
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

#define MATMUL_BLOCK_SIZE 64

typedef struct mt_tensor {
    mt_float *data;

    int ndim;
    int shape[MAX_TENSOR_NDIM];
    int count_deps;
} mt_tensor;

typedef struct {
    int           id;
    mt_layer_kind kind;
    // This member holds data of different layer types. Some layers do not
    // have any data/attribute to store, such as ReLU, simple binary
    // operations, etc. In that case, they are not listed here.
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
                "[ERROR] %s\n"                                                 \
                "\x1b[0m",                                                     \
                msg);                                                          \
        exit(1);                                                               \
    } while (0)

#define ERROR_F(fmt, ...)                                                      \
    do {                                                                       \
        fprintf(stderr,                                                        \
                "\x1b[31m"                                                     \
                "[ERROR] " fmt ": %s\n"                                        \
                "\x1b[0m",                                                     \
                __VA_ARGS__);                                                  \
        exit(1);                                                               \
    } while (0)

int mt_tensor_count_element(mt_tensor *t) {
    int count = 1;
    for (int i = 0; i < t->ndim; i++) {
        count *= t->shape[i];
    }
    return count;
}

// Some helper functions
static int mt__product(const int *arr, int n) {
    int result = 1;
    for (int i = 0; i < n; i++) {
        result *= arr[i];
    }
    return result;
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

// Optimized 1D broadcasting
static void mt__binop_1d(mt_float *a, int a_size, mt_float *b, int b_size,
                         mt_float *result, int result_size,
                         mt_float f(mt_float, mt_float)) {
    if (a_size == result_size && b_size == 1) {
        for (int i = 0; i < result_size; i++) {
            result[i] = f(a[i], b[0]);
        }
    } else if (b_size == result_size && a_size == 1) {
        for (int i = 0; i < result_size; i++) {
            result[i] = f(a[0], b[i]);
        }
    } else {
        for (int i = 0; i < result_size; i++) {
            result[i] = f(a[i % a_size], b[i % b_size]);
        }
    }
}

// Optimized 2D broadcasting
static void mt__binop_2d(mt_float *a, int *a_shape, mt_float *b, int *b_shape,
                         mt_float *result, int *result_shape,
                         mt_float f(mt_float, mt_float)) {
    int a_rows = a_shape[0], a_cols = a_shape[1];
    int b_rows = b_shape[0], b_cols = b_shape[1];
    int result_rows = result_shape[0], result_cols = result_shape[1];
    for (int i = 0; i < result_rows; i++) {
        for (int j = 0; j < result_cols; j++) {
            int a_i = i % a_rows, a_j = j % a_cols;
            int b_i = i % b_rows, b_j = j % b_cols;
            result[i * result_cols + j] =
                f(a[a_i * a_cols + a_j], b[b_i * b_cols + b_j]);
        }
    }
}

// Optimized 3D broadcasting
static void mt__binop_3d(mt_float *a, int *a_shape, mt_float *b, int *b_shape,
                         mt_float *result, int *result_shape,
                         mt_float f(mt_float, mt_float)) {
    int a_dim0 = a_shape[0], a_dim1 = a_shape[1], a_dim2 = a_shape[2];
    int b_dim0 = b_shape[0], b_dim1 = b_shape[1], b_dim2 = b_shape[2];
    int result_dim0 = result_shape[0], result_dim1 = result_shape[1],
        result_dim2 = result_shape[2];

    for (int i = 0; i < result_dim0; i++) {
        for (int j = 0; j < result_dim1; j++) {
            for (int k = 0; k < result_dim2; k++) {
                int a_i = i % a_dim0, a_j = j % a_dim1, a_k = k % a_dim2;
                int b_i = i % b_dim0, b_j = j % b_dim1, b_k = k % b_dim2;
                result[(i * result_dim1 + j) * result_dim2 + k] =
                    f(a[(a_i * a_dim1 + a_j) * a_dim2 + a_k],
                      b[(b_i * b_dim1 + b_j) * b_dim2 + b_k]);
            }
        }
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

    if (result_ndim == 1) {
        mt__binop_1d(a->data, a->shape[0], b->data, b->shape[0], result->data,
                     result_shape[0], f);
    } else if (result_ndim == 2) {
        mt__binop_2d(a->data, a->shape, b->data, b->shape, result->data,
                     result_shape, f);
    } else if (result_ndim == 3) {
        mt__binop_3d(a->data, a->shape, b->data, b->shape, result->data,
                     result_shape, f);
    } else {
        int a_strides[MAX_TENSOR_NDIM], b_strides[MAX_TENSOR_NDIM],
            result_strides[MAX_TENSOR_NDIM];

        mt__calc_strides(a->shape, a->ndim, a_strides);
        mt__calc_strides(b->shape, b->ndim, b_strides);
        mt__calc_strides(result_shape, result_ndim, result_strides);

        int a_broadcast_shape[MAX_TENSOR_NDIM],
            b_broadcast_shape[MAX_TENSOR_NDIM];
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
        // TODO: optimize for special ndims, identic shape, and tensor-scalar
        // ops
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
    }

    return result;
}

mt_tensor *mt__unop(mt_tensor *t, mt_float f(mt_float)) {
    mt_tensor *output = mt_tensor_alloc(t->shape, t->ndim);
    for (int i = 0; i > mt_tensor_count_element(t); ++i) {
        output->data[i] = f(t->data[i]);
    }
    return output;
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
    // Blocked matrix multiplication
    for (int i0 = 0; i0 < m; i0 += MATMUL_BLOCK_SIZE) {
        for (int j0 = 0; j0 < n; j0 += MATMUL_BLOCK_SIZE) {
            for (int k0 = 0; k0 < k; k0 += MATMUL_BLOCK_SIZE) {
                int max_i =
                    (i0 + MATMUL_BLOCK_SIZE < m) ? i0 + MATMUL_BLOCK_SIZE : m;
                int max_j =
                    (j0 + MATMUL_BLOCK_SIZE < n) ? j0 + MATMUL_BLOCK_SIZE : n;
                int max_k =
                    (k0 + MATMUL_BLOCK_SIZE < k) ? k0 + MATMUL_BLOCK_SIZE : k;

                for (int i = i0; i < max_i; i++) {
                    for (int k = k0; k < max_k; k++) {
                        float a_ik = a->data[i * tda + k];
                        for (int j = j0; j < max_j; j++) {
                            c->data[i * n + j] += a_ik * b->data[k * n + j];
                        }
                    }
                }
            }
        }
    }
#endif

    return c;
}

static mt_float mt__s_div(mt_float a, mt_float b) { return a / b; }
mt_tensor      *mt_div(mt_tensor *a, mt_tensor *b) {
    return mt__binop(a, b, mt__s_div);
}

static mt_float mt__s_exp(mt_float x) { return exp(x); }
mt_tensor      *mt_exp(mt_tensor *t) { return mt__unop(t, mt__s_exp); }

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

mt_tensor *mt_image_resize(mt_tensor *img, int target_height,
                           int target_width) {
    MT_ASSERT(img->ndim == 3,
              "Input tensor must be 3-dimensional (CHW format)");

    int channels   = img->shape[0];
    int src_height = img->shape[1];
    int src_width  = img->shape[2];

    mt_tensor *resized =
        mt_tensor_alloc(MT_ARR_INT(channels, target_height, target_width), 3);

    float height_scale = (float)src_height / target_height;
    float width_scale  = (float)src_width / target_width;

    // Pre-compute source y coordinates and their weights
    float *src_y  = (float *)malloc(target_height * sizeof(float));
    int   *src_y0 = (int *)malloc(target_height * sizeof(int));
    float *dy     = (float *)malloc(target_height * sizeof(float));

    for (int y = 0; y < target_height; y++) {
        src_y[y]  = y * height_scale;
        src_y0[y] = (int)src_y[y];
        dy[y]     = src_y[y] - src_y0[y];
    }

    // Pre-compute source x coordinates and their weights
    float *src_x  = (float *)malloc(target_width * sizeof(float));
    int   *src_x0 = (int *)malloc(target_width * sizeof(int));
    float *dx     = (float *)malloc(target_width * sizeof(float));

    for (int x = 0; x < target_width; x++) {
        src_x[x]  = x * width_scale;
        src_x0[x] = (int)src_x[x];
        dx[x]     = src_x[x] - src_x0[x];
    }

    // Main resizing loop
    for (int c = 0; c < channels; c++) {
        mt_float *src_channel = img->data + c * src_height * src_width;
        mt_float *dst_channel =
            resized->data + c * target_height * target_width;

        for (int y = 0; y < target_height; y++) {
            int   y0        = src_y0[y];
            int   y1        = (y0 < src_height - 1) ? y0 + 1 : y0;
            float weight_y0 = 1 - dy[y];
            float weight_y1 = dy[y];

            for (int x = 0; x < target_width; x++) {
                int   x0        = src_x0[x];
                int   x1        = (x0 < src_width - 1) ? x0 + 1 : x0;
                float weight_x0 = 1 - dx[x];
                float weight_x1 = dx[x];

                float val =
                    weight_y0 * (weight_x0 * src_channel[y0 * src_width + x0] +
                                 weight_x1 * src_channel[y0 * src_width + x1]) +
                    weight_y1 * (weight_x0 * src_channel[y1 * src_width + x0] +
                                 weight_x1 * src_channel[y1 * src_width + x1]);

                dst_channel[y * target_width + x] = val;
            }
        }
    }

    free(src_y);
    free(src_y0);
    free(dy);
    free(src_x);
    free(src_x0);
    free(dx);

    return resized;
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

// General permute function
mt_tensor *mt__permute(mt_tensor *input, const int *dims, int ndim) {
    MT_ASSERT(input->ndim == ndim,
              "Input tensor dimensions must match permutation dimensions");

    // Create new shape and strides based on permutation
    int new_shape[MAX_TENSOR_NDIM];
    int new_strides[MAX_TENSOR_NDIM];
    int old_strides[MAX_TENSOR_NDIM];

    old_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        old_strides[i] = old_strides[i + 1] * input->shape[i + 1];
    }

    for (int i = 0; i < ndim; i++) {
        new_shape[i]   = input->shape[dims[i]];
        new_strides[i] = old_strides[dims[i]];
    }

    // Allocate new tensor
    mt_tensor *output = mt_tensor_alloc(new_shape, ndim);

    // Perform permutation
    int total_elements = mt__product(input->shape, ndim);

    // Determine the fastest changing dimension in the output
    int fastest_dim = 0;
    for (int i = 1; i < ndim; i++) {
        if (new_strides[i] < new_strides[fastest_dim]) {
            fastest_dim = i;
        }
    }

    for (int i = 0; i < new_shape[fastest_dim]; i++) {
        // Calculate the base index for this slice
        int base_index = i * new_strides[fastest_dim];

        // Iterate over the rest of the dimensions
        for (int j = 0; j < total_elements / new_shape[fastest_dim]; j++) {
            int old_index = 0;
            int new_index = base_index;
            int temp      = j;

            for (int k = 0; k < ndim; k++) {
                if (k != fastest_dim) {
                    int dim_index = temp % new_shape[k];
                    temp /= new_shape[k];
                    old_index += dim_index * old_strides[dims[k]];
                    new_index += dim_index * new_strides[k];
                }
            }

            output->data[new_index] = input->data[old_index];
        }
    }

    return output;
}

// Specialized function for 2D permute
mt_tensor *mt__permute_2d(mt_tensor *input, const int *dims) {
    MT_ASSERT(input->ndim == 2, "Input tensor must be 2-dimensional");

    int        new_shape[2] = {input->shape[dims[0]], input->shape[dims[1]]};
    mt_tensor *output       = mt_tensor_alloc(new_shape, 2);

    int h = input->shape[0], w = input->shape[1];

    if (dims[0] == 0 && dims[1] == 1) {
        memcpy(output->data, input->data, h * w * sizeof(mt_float));
    } else {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                output->data[i * h + j] = input->data[j * w + i];
            }
        }
    }

    return output;
}

// Specialized function for 3D permute
mt_tensor *mt__permute_3d(mt_tensor *input, const int *dims) {
    MT_ASSERT(input->ndim == 3, "Input tensor must be 3-dimensional");

    int        new_shape[3] = {input->shape[dims[0]], input->shape[dims[1]],
                               input->shape[dims[2]]};
    mt_tensor *output       = mt_tensor_alloc(new_shape, 3);

    int d = input->shape[0], h = input->shape[1], w = input->shape[2];
    int new_d = new_shape[0], new_h = new_shape[1], new_w = new_shape[2];

    for (int i = 0; i < new_d; i++) {
        for (int j = 0; j < new_h; j++) {
            for (int k = 0; k < new_w; k++) {
                int old_i = dims[0] == 0 ? i : dims[1] == 0 ? j : k;
                int old_j = dims[0] == 1 ? i : dims[1] == 1 ? j : k;
                int old_k = dims[0] == 2 ? i : dims[1] == 2 ? j : k;
                output->data[(i * new_h + j) * new_w + k] =
                    input->data[(old_i * h + old_j) * w + old_k];
            }
        }
    }

    return output;
}

// Specialized function for 4D permute
mt_tensor *mt__permute_4d(mt_tensor *input, const int *dims) {
    MT_ASSERT(input->ndim == 4, "Input tensor must be 4-dimensional");

    int        new_shape[4] = {input->shape[dims[0]], input->shape[dims[1]],
                               input->shape[dims[2]], input->shape[dims[3]]};
    mt_tensor *output       = mt_tensor_alloc(new_shape, 4);

    int a = input->shape[0], b = input->shape[1], c = input->shape[2],
        d     = input->shape[3];
    int new_a = new_shape[0], new_b = new_shape[1], new_c = new_shape[2],
        new_d = new_shape[3];

    for (int i = 0; i < new_a; i++) {
        for (int j = 0; j < new_b; j++) {
            for (int k = 0; k < new_c; k++) {
                for (int l = 0; l < new_d; l++) {
                    int old_i = dims[0] == 0   ? i
                                : dims[1] == 0 ? j
                                : dims[2] == 0 ? k
                                               : l;
                    int old_j = dims[0] == 1   ? i
                                : dims[1] == 1 ? j
                                : dims[2] == 1 ? k
                                               : l;
                    int old_k = dims[0] == 2   ? i
                                : dims[1] == 2 ? j
                                : dims[2] == 2 ? k
                                               : l;
                    int old_l = dims[0] == 3   ? i
                                : dims[1] == 3 ? j
                                : dims[2] == 3 ? k
                                               : l;
                    output->data[((i * new_b + j) * new_c + k) * new_d + l] =
                        input->data[((old_i * b + old_j) * c + old_k) * d +
                                    old_l];
                }
            }
        }
    }

    return output;
}

// Main permute function that selects the appropriate implementation
mt_tensor *mt_tensor_permute_dims(mt_tensor *t, int *dims) {
    switch (t->ndim) {
    case 2:
        return mt__permute_2d(t, dims);
    case 3:
        return mt__permute_3d(t, dims);
    case 4:
        return mt__permute_4d(t, dims);
    default:
        return mt__permute(t, dims, t->ndim);
    }
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

// Helper function to handle negative indices and clamping
static int adjust_index(int index, int dim, int step) {
    if (index < 0) {
        index += dim;
    }
    if (step > 0) {
        return (index < 0) ? 0 : (index > dim) ? dim : index;
    } else {
        return (index < -1) ? -1 : (index >= dim) ? (dim - 1) : index;
    }
}

mt_tensor *mt_tensor_slice(mt_tensor *input, int *starts, int *ends, int *axes,
                           int *steps, int num_axes) {
    int rank = input->ndim;
    int effective_starts[MAX_TENSOR_NDIM], effective_ends[MAX_TENSOR_NDIM],
        effective_steps[MAX_TENSOR_NDIM];
    int output_shape[MAX_TENSOR_NDIM];

    for (int i = 0; i < num_axes; ++i) {
        MT_ASSERT_F(
            steps[i] >= 0,
            "cannot slice with negative steps, but step on axis %d is %d",
            axes[i], steps[i]);
    }

    // Initialize effective values
    for (int i = 0; i < rank; i++) {
        effective_starts[i] = 0;
        effective_ends[i]   = input->shape[i];
        effective_steps[i]  = 1;
    }

    // Adjust starts, ends, and steps based on provided axes
    for (int i = 0; i < num_axes; i++) {
        int axis = (axes != NULL) ? axes[i] : i;
        if (axis < 0)
            axis += rank;

        effective_starts[axis] =
            adjust_index(starts[i], input->shape[axis], steps ? steps[i] : 1);
        effective_ends[axis] =
            adjust_index(ends[i], input->shape[axis], steps ? steps[i] : 1);
        effective_steps[axis] = steps ? steps[i] : 1;

        // Handle INT_MAX and INT_MIN for ends
        if (ends[i] == INT_MAX)
            effective_ends[axis] = input->shape[axis];
        if (ends[i] == INT_MIN)
            effective_ends[axis] = -1;

        // Calculate output shape
        int slice_length = (effective_ends[axis] - effective_starts[axis] +
                            effective_steps[axis] - 1) /
                           effective_steps[axis];
        output_shape[axis] = (slice_length < 0) ? 0 : slice_length;
    }

    // Allocate output tensor
    mt_tensor *output = mt_tensor_alloc(output_shape, rank);

    // Perform slicing
    int input_indices[MAX_TENSOR_NDIM]  = {0};
    int output_indices[MAX_TENSOR_NDIM] = {0};
    int total_elements                  = 1;
    for (int i = 0; i < rank; i++) {
        total_elements *= output_shape[i];
    }

    for (int i = 0; i < total_elements; i++) {
        // Calculate input indices
        for (int j = 0; j < rank; j++) {
            input_indices[j] =
                effective_starts[j] + output_indices[j] * effective_steps[j];
        }

        // Copy data
        int input_flat_index  = 0;
        int output_flat_index = 0;
        int input_stride      = 1;
        int output_stride     = 1;

        for (int j = rank - 1; j >= 0; j--) {
            input_flat_index += input_indices[j] * input_stride;
            output_flat_index += output_indices[j] * output_stride;
            input_stride *= input->shape[j];
            output_stride *= output_shape[j];
        }

        output->data[output_flat_index] = input->data[input_flat_index];

        // Update output indices
        for (int j = rank - 1; j >= 0; j--) {
            output_indices[j]++;
            if (output_indices[j] < output_shape[j]) {
                break;
            }
            output_indices[j] = 0;
        }
    }

    return output;
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
    printf("data : [");
    long len = mt_tensor_count_element(t);
    for (int i = 0; i < len; ++i) {
        printf("%f", t->data[i]);
        if (i < len - 1)
            printf(", ");
    }
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

typedef struct {
    unsigned char *data;
    size_t         pos;
    size_t         size;
} mt_reader;

size_t mt_reader_read(void *ptr, size_t size, size_t count, mt_reader *stream) {
    size_t total = size * count;
    if (stream->pos + total > stream->size) {
        total = stream->size - stream->pos;
    }
    memcpy(ptr, stream->data + stream->pos, total);
    stream->pos += total;
    return total / size;
}

mt_tensor *mt_tensor_memread(mt_reader *mp) {
    int *ndim = (int *)calloc(1, sizeof(int));
    mt_reader_read(ndim, sizeof(int), 1, mp);

    int *shape = (int *)calloc(*ndim, sizeof(int));
    mt_reader_read(shape, sizeof(int), *ndim, mp);

    int tensor_numel = 1;
    for (int i = 0; i < *ndim; i++)
        tensor_numel *= shape[i];
    mt_float *values = (mt_float *)calloc(tensor_numel, sizeof(mt_float));
    mt_reader_read(values, sizeof(mt_float), tensor_numel, mp);

    mt_tensor *t = mt_tensor_alloc_values(shape, *ndim, values);

    free(ndim), free(shape), free(values);
    return t;
}

mt_model *mt_model_load_from_mem(unsigned char *model_bytes, size_t len,
                                 int input_in_batch) {

    mt_reader mp    = (mt_reader){model_bytes, 0, len};
    mt_model *model = (mt_model *)malloc(sizeof(*model));

    // First, we read model header.
    mt_reader_read(&model->layer_count, sizeof(int), 1, &mp);
    mt_reader_read(&model->tensor_count, sizeof(int), 1, &mp);
    DEBUG_LOG_F("model has %d nodes and %d tensors", model->layer_count,
                model->tensor_count);

    // Read layers and tensors
    for (int i = 0; i < model->layer_count; ++i) {
        mt_layer *layer = (mt_layer *)malloc(sizeof(*layer));

        // Read layer header
        mt_reader_read(&layer->kind, sizeof(int), 1, &mp);
        mt_reader_read(&layer->id, sizeof(int), 1, &mp);
        mt_reader_read(&layer->prev_count, sizeof(int), 1, &mp);
        mt_reader_read(&layer->prev, sizeof(int), layer->prev_count, &mp);
        mt_reader_read(&layer->next_count, sizeof(int), 1, &mp);
        mt_reader_read(&layer->next, sizeof(int), layer->next_count, &mp);
        mt_reader_read(&layer->input_count, sizeof(int), 1, &mp);
        mt_reader_read(&layer->inputs, sizeof(int), layer->input_count, &mp);
        mt_reader_read(&layer->output_count, sizeof(int), 1, &mp);
        mt_reader_read(&layer->outputs, sizeof(int), layer->output_count, &mp);

        if (layer->kind == MT_LAYER_ADD) {
            // nothing to read
        } else if (layer->kind == MT_LAYER_AVG_POOL_2D) {
            mt_reader_read(&layer->data.avg_pool_2d.size, sizeof(int), 1, &mp);
            mt_reader_read(&layer->data.avg_pool_2d.stride, sizeof(int), 1,
                           &mp);
            mt_reader_read(&layer->data.avg_pool_2d.pad, sizeof(int), 1, &mp);
        } else if (layer->kind == MT_LAYER_CONV_2D) {
            mt_reader_read(&layer->data.conv_2d.stride, sizeof(int), 1, &mp);
            mt_reader_read(&layer->data.conv_2d.pad, sizeof(int), 1, &mp);
            int w_idx                = layer->inputs[1];
            layer->data.conv_2d.w_id = w_idx;
            model->tensors[w_idx]    = mt_tensor_memread(&mp);
            int b_idx                = layer->inputs[2];
            layer->data.conv_2d.b_id = b_idx;
            model->tensors[b_idx]    = mt_tensor_memread(&mp);
        } else if (layer->kind == MT_LAYER_DENSE) {
            int w_idx                = layer->inputs[1];
            layer->data.conv_2d.w_id = w_idx;
            model->tensors[w_idx]    = mt_tensor_memread(&mp);
            int b_idx                = layer->inputs[2];
            layer->data.conv_2d.b_id = b_idx;
            model->tensors[b_idx]    = mt_tensor_memread(&mp);
        } else if (layer->kind == MT_LAYER_MAX_POOL_2D) {
            mt_reader_read(&layer->data.max_pool_2d.size, sizeof(int), 1, &mp);
            mt_reader_read(&layer->data.max_pool_2d.stride, sizeof(int), 1,
                           &mp);
            mt_reader_read(&layer->data.max_pool_2d.pad, sizeof(int), 1, &mp);
        } else if (layer->kind == MT_LAYER_FLATTEN) {
            mt_reader_read(&layer->data.flatten.axis, sizeof(int), 1, &mp);

            // if model input is in batch, axis should be decreased by one
            if (input_in_batch) {
                DEBUG_LOG_F("Input is in batch but only a single sa&mple is "
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
            mt_reader_read(&layer->data.local_response_norm.size, sizeof(int),
                           1, &mp);
            mt_reader_read(&layer->data.local_response_norm.alpha,
                           sizeof(mt_float), 1, &mp);
            mt_reader_read(&layer->data.local_response_norm.beta,
                           sizeof(mt_float), 1, &mp);
            mt_reader_read(&layer->data.local_response_norm.bias,
                           sizeof(mt_float), 1, &mp);
        } else {
            if (layer->kind == MT_LAYER_UNKNOWN) {
                printf("unknown layer detected, possibly because its "
                       "existence "
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
    mt_reader_read(&model->input_count, sizeof(int), 1, &mp);
    for (int i = 0; i < model->input_count; ++i) {
        mt_reader_read(&model->inputs[i].name, sizeof(char),
                       MAX_INPUT_OUTPUT_NAME_LEN, &mp);
        mt_reader_read(&model->inputs[i].id, sizeof(int), 1, &mp);
    }
    mt_reader_read(&model->output_count, sizeof(int), 1, &mp);
    for (int i = 0; i < model->output_count; ++i) {
        mt_reader_read(&model->outputs[i].name, sizeof(char),
                       MAX_INPUT_OUTPUT_NAME_LEN, &mp);
        mt_reader_read(&model->outputs[i].id, sizeof(int), 1, &mp);
    }
    DEBUG_LOG_F("model graph has %d input(s)", model->input_count);
    DEBUG_LOG_F("model graph has %d output(s)", model->output_count);

    return model;
}

mt_model *mt_model_load(const char *filename, int input_in_batch) {
    FILE *fp = fopen(filename, "rb");
    MT_ASSERT_F(fp != NULL, "failed to open %s", filename);

    fseek(fp, 0, SEEK_END);
    long filelen = ftell(fp);
    rewind(fp);

    // mt_model *model  = (mt_model *)malloc(sizeof(*model));
    unsigned char *buffer = (unsigned char *)malloc(filelen * sizeof(char));
    // Read in the entire model file in a buffer
    fread(buffer, filelen, 1, fp);
    mt_model *model = mt_model_load_from_mem(buffer, filelen, input_in_batch);

    free(buffer);
    fclose(fp);

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

        // If input ndim is 4 with shape (2, 2, 2, 2) and start_axis is 1,
        // then the resulting shape will be (2, 2 * 2 * 2) and ndim will
        // be 2. If the start_axis is 2, then the resulting shape will be
        // (2, 2, 2 * 2) and ndim will be 3. Thus, out_ndim would be
        // start_axis+1;
        int out_ndim = start_axis + 1;

        // trailing dim size will be the product of start_axis-th size till
        // last axis shape
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
