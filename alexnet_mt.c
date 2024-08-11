#include "mint.h"
#include <stdio.h>

int main(void) {
  mt_model *model = mt_model_load("alexnet.mt");
  for (int i = 0; i < model->layer_count; ++i) {
    if (model->layers[i]->kind == MT_LAYER_CONV_2D) {
      // printf("axis %d\n", model->layers[i]->data.flatten.axis);
      printf("%d\n",
             model->tensors[model->layers[i]->data.conv_2d.w_id]->shape[0]);
    }
  }

  mt_model_free(model);
  return 0;
}
