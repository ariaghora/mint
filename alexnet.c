#include <assert.h>
#include <stdio.h>

#include <cblas.h>

#define STB_IMAGE_IMPLEMENTATION
#include "vendors/stb_image.h"

#define MT_USE_BLAS
#define MT_USE_STB_IMAGE
// #define MT_USE_IM2ROW_CONV
#include "mint.h"

typedef struct Conv2d {
  mt_tensor *w;
  mt_tensor *b;
  int        stride;
  int        pad;
} Conv2d;

typedef struct Dense {
  mt_tensor *w;
  mt_tensor *b;
} Dense;

Conv2d *conv2d_fread(FILE *fp) {
  Conv2d *conv = (Conv2d *)malloc(sizeof(*conv));
  conv->w      = mt_tensor_fread(fp);
  conv->b      = mt_tensor_fread(fp);
  fread(&conv->stride, sizeof(int), 1, fp);
  fread(&conv->pad, sizeof(int), 1, fp);
  return conv;
}

mt_tensor *conv2d_forward(Conv2d *conv, mt_tensor *x) {
  return mt_convolve_2d(x, conv->w, conv->b, conv->stride, conv->pad);
}

void conv2d_free(Conv2d *conv) {
  mt_tensor_free(conv->w);
  mt_tensor_free(conv->b);
  free(conv);
}

Dense *dense_fread(FILE *fp) {
  Dense *d = (Dense *)malloc(sizeof(*d));
  d->w     = mt_tensor_fread(fp);
  d->b     = mt_tensor_fread(fp);
  return d;
}

mt_tensor *dense_forward(Dense *d, mt_tensor *x) {
  return mt_affine(x, d->w, d->b);
}

void dense_free(Dense *d) {
  mt_tensor_free(d->w);
  mt_tensor_free(d->b);
  free(d);
}

int main(void) {
  mt_tensor *img = mt_tensor_load_image("car.jpg");
  mt_image_standardize(img, MT_ARR_FLOAT(0.485, 0.456, 0.406),
                       MT_ARR_FLOAT(0.229, 0.224, 0.225));

  FILE *fp = fopen("alexnet.dat", "rb");
  assert(fp != NULL);

  // Load first 5 consecutive convolution layer weights
  Conv2d *conv_layers[5];
  for (int i = 0; i < 5; i++)
    conv_layers[i] = conv2d_fread(fp);

  // Load next 3 consecutive dense layer weights
  Dense *dense_layers[3];
  for (int i = 0; i < 3; i++)
    dense_layers[i] = dense_fread(fp);

  //================= INFERENCE:START ==================
  // Convolution part
  mt_tensor *h1 = conv2d_forward(conv_layers[0], img);
  mt_relu_inplace(h1);
  mt_tensor *h1_pool = mt_maxpool_2d(h1, 3, 2, 0);
  mt_tensor_free(h1);

  mt_tensor *h2 = conv2d_forward(conv_layers[1], h1_pool);
  mt_relu_inplace(h2);
  mt_tensor *h2_pool = mt_maxpool_2d(h2, 3, 2, 0);
  mt_tensor_free(h1_pool);
  mt_tensor_free(h2);

  mt_tensor *h3 = conv2d_forward(conv_layers[2], h2_pool);
  mt_relu_inplace(h3);
  mt_tensor_free(h2_pool);

  mt_tensor *h4 = conv2d_forward(conv_layers[3], h3);
  mt_relu_inplace(h4);
  mt_tensor_free(h3);

  mt_tensor *h5 = conv2d_forward(conv_layers[4], h4);
  mt_relu_inplace(h5);
  mt_tensor *h5_pool = mt_adaptive_avg_pool_2d(h5, 6, 6);
  mt_tensor_free(h5);

  // Flatten
  mt_reshape_inplace(h5_pool, MT_ARR_INT(1, h5_pool->shape[0] * 36), 2);

  // Dense part
  mt_tensor *fc1 = dense_forward(dense_layers[0], h5_pool);
  mt_relu_inplace(fc1);
  mt_tensor *fc2 = dense_forward(dense_layers[1], fc1);
  mt_relu_inplace(fc2);
  mt_tensor *fc3 = dense_forward(dense_layers[2], fc2);

  //================= INFERENCE:END ==================
  int      arg_max = 0;
  mt_float cur_max = fc3->data[arg_max];
  for (int i = 0; i < mt__tensor_count_element(fc3); ++i) {
    if (fc3->data[i] > cur_max) {
      cur_max = fc3->data[i];
      arg_max = i;
    }
  }
  printf("argmax=%d\n", arg_max);
  printf("score=%f\n", cur_max);

  // Cleanups
  mt_tensor_free(img);
  for (int i = 0; i < 5; i++)
    conv2d_free(conv_layers[i]);
  for (int i = 0; i < 3; i++)
    dense_free(dense_layers[i]);

  return 0;
}
