#define MT_IMPLEMENTATION
#define MT_ONNX_IMPLEMENTATION
#include "../../mint_onnx.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return 1;
    }
    const char *filename = argv[1];
    mt_model   *model    = mt_onnx_read_file(filename);
    if (model == NULL) {
        printf("Failed to load model\n");
        return 1;
    }
    printf("Model %s loaded successfully\n", filename);

    mt_model_free(model);
    return 0;
}