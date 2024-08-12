#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "vendors/stb_image.h"

#define MT_USE_BLAS
#include <cblas.h>

#define MT_USE_IM2COL_CONV
#define MT_USE_STB_IMAGE
#define NDEBUG
#include "mint.h"

int main(void) {
    mt_model  *model = mt_model_load("alexnet.mt", 1);
    mt_tensor *image = mt_tensor_load_image("./leopard.jpg");

    float *mean = MT_ARR_FLOAT(0.485, 0.456, 0.406);
    float *std  = MT_ARR_FLOAT(0.229, 0.224, 0.225);
    mt_image_standardize(image, mean, std);

    mt_model_set_input(model, "input", image);
    mt_model_run(model);

    mt_tensor *output = mt_model_get_output(model, "output");

    int      arg_max = 0;
    mt_float cur_max = output->data[arg_max];
    for (int i = 0; i < mt_tensor_count_element(output); ++i) {
        if (output->data[i] > cur_max) {
            cur_max = output->data[i];
            arg_max = i;
        }
    }
    printf("class %d\n", arg_max);

    mt_tensor_free(output);
    mt_tensor_free(image);
    mt_model_free(model);
    return 0;
}
