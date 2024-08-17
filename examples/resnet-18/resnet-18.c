#define STB_IMAGE_IMPLEMENTATION
#include "../../vendors/stb_image.h"

#define MT_USE_BLAS
#include <cblas.h>

// Comment line below to enable logging and assertion
#define NDEBUG

#define MT_USE_IM2COL_CONV
#define MT_USE_STB_IMAGE
#define MT_IMPLEMENTATION
#include "../../mint.h"

int main(int argc, char **argv) {
    // clang-format off
    char *class_labels[] = {
        #include "class-list.inc"
    };
    // clang-format on

    if (argc != 3) {
        printf("This program requires 2 arguments: 1) the path to resnet.mt "
               "and 2) the path image to predict");
        exit(1);
    }

    mt_model  *model = mt_model_load(argv[1], 1);
    mt_tensor *image = mt_tensor_load_image(argv[2]);

    float *mean = MT_ARR_FLOAT(0.485, 0.456, 0.406);
    float *std  = MT_ARR_FLOAT(0.229, 0.224, 0.225);
    mt_image_standardize(image, mean, std);

    mt_tensor_unsqueeze_inplace(image, 0);

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
    printf("class label: %s\n", class_labels[arg_max]);

    mt_tensor_free(output);
    mt_tensor_free(image);
    mt_model_free(model);
    return 0;
}
