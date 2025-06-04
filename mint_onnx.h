#ifndef MINT_ONNX_H
#define MINT_ONNX_H

#include "mint.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

MTDEF mt_model *mt_onnx_read_file(const char *filename);
MTDEF mt_model *mt_onnx_read_mem(unsigned char *model_bytes,
                                 size_t         model_bytes_len);

#ifdef __cplusplus
}
#endif

#ifdef MT_ONNX_IMPLEMENTATION

#include "onnx/onnx.proto3.pb-c.h"
#include <stdio.h>
#include <stdlib.h>

#if !defined(UNUSED)
#define UNUSED(x) ((void)(x))
#endif

MTDEF mt_layer_kind mt_onnx__get_layer_kind(const char *op_type) {
    if (strcmp(op_type, "Conv") == 0) {
        return MT_LAYER_CONV_2D;
    } else if (strcmp(op_type, "Relu") == 0) {
        return MT_LAYER_RELU;
    } else if (strcmp(op_type, "MaxPool") == 0) {
        return MT_LAYER_MAX_POOL_2D;
    } else if (strcmp(op_type, "Add") == 0) {
        return MT_LAYER_ADD;
    } else if (strcmp(op_type, "Exp") == 0) {
        return MT_LAYER_EXP;
    } else if (strcmp(op_type, "GlobalAveragePool") == 0) {
        return MT_LAYER_GLOBAL_AVG_POOL;
    } else if (strcmp(op_type, "Flatten") == 0) {
        return MT_LAYER_FLATTEN;
    } else if (strcmp(op_type, "Gemm") == 0) {
        return MT_LAYER_DENSE;
    }
    return MT_LAYER_UNKNOWN;
}

MTDEF mt_model *mt_onnx_read_file(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL)
        return NULL;
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    unsigned char *model_bytes = (unsigned char *)malloc(file_size);
    fread(model_bytes, 1, file_size, file);
    fclose(file);
    mt_model *model = mt_onnx_read_mem(model_bytes, file_size);
    free(model_bytes);
    return model;
}

MTDEF Onnx__TensorProto *
mt_onnx__get_tensor_proto(Onnx__ModelProto *model_proto, const char *name) {
    Onnx__TensorProto **initializers = model_proto->graph->initializer;
    for (size_t i = 0; i < model_proto->graph->n_initializer; i++) {
        Onnx__TensorProto *tensor_proto = initializers[i];
        if (strcmp(tensor_proto->name, name) == 0) {
            return tensor_proto;
        }
    }
    ERROR_F("Tensor %s not found", name);
    exit(1);
}

// We maintain table for model's nodes.
typedef struct node_info_t {
    int   id;
    char *name;
    char *input_names[MAX_INPUT_OUTPUT_COUNT];
    int   input_count;
    char *output_names[MAX_INPUT_OUTPUT_COUNT];
    int   output_count;
} node_info_t;

typedef struct tensor_info_t {
    int   id;
    char *name;
} tensor_info_t;

MTDEF int mt_onnx__get_tensor_id(tensor_info_t *tensor_infos, int n_tensor,
                                 const char *name) {
    for (int i = 0; i < n_tensor; i++) {
        if (strcmp(tensor_infos[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

MTDEF mt_tensor *
mt_onnx__tensor_proto_to_mt_tensor(Onnx__TensorProto *tensor_proto) {
    int shape[tensor_proto->n_dims];
    for (size_t i = 0; i < tensor_proto->n_dims; i++) {
        shape[i] = tensor_proto->dims[i];
    }

    mt_tensor *tensor    = mt_tensor_alloc(shape, tensor_proto->n_dims);
    size_t     n_element = mt_tensor_count_element(tensor);

    Onnx__TensorProto__DataType dtype =
        (Onnx__TensorProto__DataType)tensor_proto->data_type;

    switch (dtype) {
    case ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT:
        if (tensor_proto->n_float_data > 0) {
            for (size_t i = 0; i < n_element && i < tensor_proto->n_float_data;
                 i++) {
                tensor->data[i] = tensor_proto->float_data[i];
            }
        } else if (tensor_proto->raw_data.len > 0) {
            mt_float *float_data = (mt_float *)tensor_proto->raw_data.data;
            for (size_t i = 0; i < n_element; i++) {
                tensor->data[i] = (mt_float)float_data[i];
            }
        } else {
            ERROR_F("No data found in tensor %s", tensor_proto->name);
        }
        break;
    default:
        ERROR_F("Unsupported data type: %d", dtype);
    }
    return tensor;
}

MTDEF void mt_onnx__make_conv(mt_layer *layer, int opset,
                              Onnx__NodeProto *node_proto) {
    layer->data.conv_2d.w_id = layer->inputs[1];
    layer->data.conv_2d.b_id = layer->inputs[2];

    // Default values:
    memcpy(layer->data.conv_2d.dilations, (int[]){1, 1}, 2);
    memcpy(layer->data.conv_2d.pads, (int[]){0, 0, 0, 0}, 4);
    layer->data.conv_2d.group    = 1;
    layer->data.conv_2d.auto_pad = 0;
    layer->data.conv_2d.stride   = 1;

    // Read attributes of conv2d layer
    // There is no significant difference from opset 1 to 22, except for
    // the type constraints, which are irrelevant to us.
    if (opset >= 1 && opset <= 22) {
        for (size_t i = 0; i < node_proto->n_attribute; i++) {
            Onnx__AttributeProto *attribute_proto = node_proto->attribute[i];
            if (strcmp(attribute_proto->name, "dilations") == 0) {
                memcpy(layer->data.conv_2d.dilations, attribute_proto->ints,
                       attribute_proto->n_ints);
            } else if (strcmp(attribute_proto->name, "pads") == 0) {
                memcpy(layer->data.conv_2d.pads, attribute_proto->ints,
                       attribute_proto->n_ints);
            } else if (strcmp(attribute_proto->name, "group") == 0) {
                layer->data.conv_2d.group = attribute_proto->i;
            } else if (strcmp(attribute_proto->name, "auto_pad") == 0) {
                if (strcmp((const char *)attribute_proto->s.data, "NOTSET") ==
                    0) {
                    layer->data.conv_2d.auto_pad = 0;
                } else if (strcmp((const char *)attribute_proto->s.data,
                                  "VALID") == 0) {
                    layer->data.conv_2d.auto_pad = 1;
                } else if (strcmp((const char *)attribute_proto->s.data,
                                  "SAME_UPPER") == 0) {
                    layer->data.conv_2d.auto_pad = 2;
                } else if (strcmp((const char *)attribute_proto->s.data,
                                  "SAME_LOWER") == 0) {
                    layer->data.conv_2d.auto_pad = 3;
                } else {
                    layer->data.conv_2d.auto_pad = 0; // default to NOTSET
                }
            } else if (strcmp(attribute_proto->name, "strides") == 0) {
                // ensure all strides are the same
                for (size_t j = 1; j < attribute_proto->n_ints; j++) {
                    if (attribute_proto->ints[j] != attribute_proto->ints[0]) {
                        ERROR_F(
                            "Cannot handle different strides yet: %lld != %lld",
                            attribute_proto->ints[j], attribute_proto->ints[0]);
                        exit(1);
                    }
                }
                layer->data.conv_2d.stride = attribute_proto->ints[0];
            }
        }
    } else {
        ERROR_F("Opset %d is not supported for %s", opset, node_proto->op_type);
        exit(1);
    }
}

MTDEF void mt_onnx__make_dense(mt_model *model, mt_layer *layer, int opset,
                               Onnx__NodeProto *node_proto) {
    layer->data.dense.w_id = layer->inputs[1];
    layer->data.dense.b_id = layer->inputs[2];

    // Default values:
    layer->data.dense.trans_a = 0;
    layer->data.dense.trans_b = 0;

    // Read attributes of dense layer: trans_a, trans_b
    if (opset >= 1 && opset <= 22) {
        for (size_t i = 0; i < node_proto->n_attribute; i++) {
            Onnx__AttributeProto *attribute_proto = node_proto->attribute[i];
            if (strcmp(attribute_proto->name, "transA") == 0) {
                layer->data.dense.trans_a = attribute_proto->i;
            } else if (strcmp(attribute_proto->name, "transB") == 0) {
                layer->data.dense.trans_b = attribute_proto->i;
            }
        }
    } else {
        ERROR_F("Opset %d is not supported for %s", opset, node_proto->op_type);
        exit(1);
    }

    // Some providers transpose the weight matrix, so we need to handle that.
    if (layer->data.dense.trans_b == 1) {
        mt_tensor *w = model->tensors[layer->data.dense.w_id];
        MT_ASSERT_F(w->ndim == 2, "w must be 2 dimensional, got %d", w->ndim);
        mt_tensor *wt = mt_tensor_permute_dims(w, (int[]){1, 0});
        mt_tensor_free(w);
        model->tensors[layer->data.dense.w_id] = wt;
    }
}

MTDEF void mt_onnx__make_max_pool(mt_layer *layer, int opset,
                                  Onnx__NodeProto *node_proto) {
    // Default values:
    layer->data.max_pool_2d.auto_pad = 0;
    layer->data.max_pool_2d.size     = 2;
    layer->data.max_pool_2d.stride   = 1;
    memcpy(layer->data.max_pool_2d.pads, (int[]){0, 0, 0, 0}, 4);

    // Read attributes of max_pool_2d layer: auto_pad, size, stride, pads.
    if (opset >= 1 && opset <= 22) {
        for (size_t i = 0; i < node_proto->n_attribute; i++) {
            Onnx__AttributeProto *attribute_proto = node_proto->attribute[i];
            if (strcmp(attribute_proto->name, "auto_pad") == 0) {
                if (strcmp((const char *)attribute_proto->s.data, "NOTSET") ==
                    0) {
                    layer->data.max_pool_2d.auto_pad = 0;
                } else if (strcmp((const char *)attribute_proto->s.data,
                                  "VALID") == 0) {
                    layer->data.max_pool_2d.auto_pad = 1;
                } else if (strcmp((const char *)attribute_proto->s.data,
                                  "SAME_UPPER") == 0) {
                    layer->data.max_pool_2d.auto_pad = 2;
                } else if (strcmp((const char *)attribute_proto->s.data,
                                  "SAME_LOWER") == 0) {
                    layer->data.max_pool_2d.auto_pad = 3;
                } else {
                    layer->data.max_pool_2d.auto_pad = 0; // default to NOTSET
                }
            } else if (strcmp(attribute_proto->name, "size") == 0) {
                layer->data.max_pool_2d.size = attribute_proto->i;
            } else if (strcmp(attribute_proto->name, "stride") == 0) {
                layer->data.max_pool_2d.stride = attribute_proto->i;
            } else if (strcmp(attribute_proto->name, "pads") == 0) {
                memcpy(layer->data.max_pool_2d.pads, attribute_proto->ints,
                       attribute_proto->n_ints);
            } else if (strcmp(attribute_proto->name, "ceil_mode") == 0) {
                // Ensure ceil_mode is 0
                if (attribute_proto->i != 0) {
                    ERROR_F("non-zero ceil_mode is not supported for %s",
                            node_proto->op_type);
                    exit(1);
                }
            } else if (strcmp(attribute_proto->name, "storage_order") == 0) {
                ERROR_F("storage_order is not supported for %s",
                        node_proto->op_type);
                exit(1);
            } else if (strcmp(attribute_proto->name, "dilations") == 0) {
                // ensure all dilations are 1, otherwise, panic
                for (size_t j = 1; j < attribute_proto->n_ints; j++) {
                    if (attribute_proto->ints[j] != 1) {
                        ERROR("All dilations must be 1 for max_pool_2d");
                        exit(1);
                    }
                }
            }
        }
    } else {
        ERROR_F("Opset %d is not supported for %s", opset, node_proto->op_type);
        exit(1);
    }
}

MTDEF void mt_onnx__make_flatten(mt_layer *layer, int opset,
                                 Onnx__NodeProto *node_proto) {
    // Default values:
    layer->data.flatten.axis = 1;

    // Read attributes of flatten layer: axis
    if (opset >= 1 && opset <= 22) {
        for (size_t i = 0; i < node_proto->n_attribute; i++) {
            Onnx__AttributeProto *attribute_proto = node_proto->attribute[i];
            if (strcmp(attribute_proto->name, "axis") == 0) {
                layer->data.flatten.axis = attribute_proto->i;
            }
        }
    } else {
        ERROR_F("Opset %d is not supported for %s", opset, node_proto->op_type);
        exit(1);
    }
}

MTDEF mt_model *mt_onnx_read_mem(unsigned char *model_bytes,
                                 size_t         model_bytes_len) {
    Onnx__ModelProto *model_proto = onnx__model_proto__unpack(
        NULL, model_bytes_len, (const uint8_t *)model_bytes);
    int opset = model_proto->opset_import[0]->version;

    DEBUG_LOG_F("Producer name    : %s", model_proto->producer_name);
    DEBUG_LOG_F("Producer version : %s", model_proto->producer_version);
    DEBUG_LOG_F("Model version    : %lld", model_proto->model_version);
    DEBUG_LOG_F("Opset            : %d", opset);

    mt_model *model = (mt_model *)MT_MALLOC(sizeof(mt_model));
    if (model == NULL) {
        ERROR("Failed to allocate memory for model");
        return NULL;
    }
    model->layer_count = model_proto->graph->n_node;

    // Pass 1:
    // Collect node info, including name, input names, output names, input
    // count,
    node_info_t node_infos[MAX_LAYER_COUNT];
    for (size_t i = 0; i < model_proto->graph->n_node; i++) {
        Onnx__NodeProto *node_proto = model_proto->graph->node[i];
        node_infos[i].id            = i;
        node_infos[i].input_count   = node_proto->n_input;
        node_infos[i].output_count  = node_proto->n_output;
        node_infos[i].name          = node_proto->name;
        for (size_t j = 0; j < node_proto->n_input; j++) {
            node_infos[i].input_names[j] = node_proto->input[j];
        }
        for (size_t j = 0; j < node_proto->n_output; j++) {
            node_infos[i].output_names[j] = node_proto->output[j];
        }
    }

    // Pass 2:
    // Collect tensor info, including name, id
    tensor_info_t tensor_infos[MAX_MODEL_INITIALIZER_COUNT];
    for (size_t i = 0; i < model_proto->graph->n_initializer; i++) {
        Onnx__TensorProto *tensor_proto = model_proto->graph->initializer[i];
        tensor_infos[i].id              = i;
        tensor_infos[i].name            = tensor_proto->name;
        model->tensors[i] = mt_onnx__tensor_proto_to_mt_tensor(tensor_proto);
    }
    for (size_t i = 0; i < model_proto->graph->n_input; i++) {
        int idx = i + model_proto->graph->n_initializer;

        Onnx__ValueInfoProto *value_info_proto = model_proto->graph->input[i];
        tensor_infos[idx].id                   = idx;
        tensor_infos[idx].name                 = value_info_proto->name;
        model->tensors[idx]                    = NULL;
        model->inputs[i].id                    = idx;
        strncpy(model->inputs[i].name, value_info_proto->name,
                MAX_INPUT_OUTPUT_NAME_LEN);
    }
    for (size_t i = 0; i < model_proto->graph->n_output; i++) {
        int idx =
            i + model_proto->graph->n_initializer + model_proto->graph->n_input;
        Onnx__ValueInfoProto *value_info_proto = model_proto->graph->output[i];
        tensor_infos[idx].id                   = idx;
        tensor_infos[idx].name                 = value_info_proto->name;
        model->tensors[idx]                    = NULL;
        model->outputs[i].id                   = idx;
        strncpy(model->outputs[i].name, value_info_proto->name,
                MAX_INPUT_OUTPUT_NAME_LEN);
    }

    // Pass 3:
    // Read layers (nodes)
    Onnx__NodeProto **nodes = model_proto->graph->node;
    DEBUG_LOG_F("Nodes: %zu", model_proto->graph->n_node);
    for (size_t i = 0; i < model_proto->graph->n_node; i++) {
        Onnx__NodeProto *node_proto = nodes[i];

        mt_layer     *layer = (mt_layer *)MT_MALLOC(sizeof(mt_layer));
        mt_layer_kind kind  = mt_onnx__get_layer_kind(node_proto->op_type);
        layer->kind         = kind;
        layer->input_count  = node_proto->n_input;
        layer->output_count = node_proto->n_output;
        layer->id           = i;
        model->layers[i]    = layer;

        for (size_t j = 0; j < node_proto->n_input; j++) {
            int input_id = mt_onnx__get_tensor_id(
                tensor_infos,
                model_proto->graph->n_initializer + model_proto->graph->n_input,
                node_proto->input[j]);
            layer->inputs[j] = input_id;
        }
        for (size_t j = 0; j < node_proto->n_output; j++) {
            int output_id = mt_onnx__get_tensor_id(
                tensor_infos,
                model_proto->graph->n_initializer + model_proto->graph->n_input,
                node_proto->output[j]);
            layer->outputs[j] = output_id;
        }

        switch (kind) {
        case MT_LAYER_CONV_2D:
            mt_onnx__make_conv(layer, opset, node_proto);
            break;
        case MT_LAYER_MAX_POOL_2D:
            mt_onnx__make_max_pool(layer, opset, node_proto);
            break;
        case MT_LAYER_FLATTEN:
            mt_onnx__make_flatten(layer, opset, node_proto);
            break;
        case MT_LAYER_DENSE:
            mt_onnx__make_dense(model, layer, opset, node_proto);
            break;
        // Pass through, since there's no data to parse
        case MT_LAYER_ADD:
        case MT_LAYER_EXP:
        case MT_LAYER_RELU:
        case MT_LAYER_GLOBAL_AVG_POOL:
            break;
        default:
            ERROR_F("Node %zu: %s is not supported", i, node_proto->op_type);
            return NULL;
        }

        // Try to find previous nodes (not input) that should be connected to
        // this node. If that node has output that matches any of this node's
        // inputs, then we found the previous node.
        for (size_t j = 0; j < i; j++) {
            node_info_t prev_node_info = node_infos[j];
            for (size_t k = 0; k < (size_t)prev_node_info.output_count; k++) {
                char *output_name = prev_node_info.output_names[k];
                for (size_t l = 0; l < node_proto->n_input; l++) {
                    if (strcmp(output_name, node_proto->input[l]) == 0) {
                        // Record the connection in the model's layer structure
                        if (layer->prev_count < MAX_LAYER_PREV_COUNT) {
                            layer->prev[layer->prev_count++] = j;

                            // Also update the next connection for the previous
                            // layer
                            mt_layer *prev_layer = model->layers[j];
                            if (prev_layer->next_count < MAX_LAYER_NEXT_COUNT) {
                                prev_layer->next[prev_layer->next_count++] = i;
                            }
                        }
                        // Once we've found a connection for this input, move to
                        // the next input
                        break;
                    }
                }
            }
        }
    }

    onnx__model_proto__free_unpacked(model_proto, NULL);
    return model;
}
#endif

#endif // MINT_ONNX_H