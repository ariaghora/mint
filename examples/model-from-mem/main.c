#include <stdlib.h>

#define MT_IMPLEMENTATION
#include "../../mint.h"

int main() {

    FILE *fileptr = fopen("resnet-18.mt", "rb");
    fseek(fileptr, 0, SEEK_END);
    long filelen = ftell(fileptr);
    rewind(fileptr);

    unsigned char *buffer = (unsigned char *)malloc(filelen * sizeof(char));
    // Read in the entire model file
    fread(buffer, filelen, 1, fileptr);

    // Load model from the memory
    mt_model *model = mt_model_load_from_mem(buffer, filelen);

    // Do something with the model here
    // ...
    // ...

    // Cleanup
    mt_model_free(model);
    free(buffer);
    fclose(fileptr);
    return 0;
}
