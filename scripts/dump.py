from collections import OrderedDict
from enum import Enum
from io import BufferedWriter
from typing import Any, Dict, List, OrderedDict
from torchvision import models, os

import numpy as np
import onnx
import torch

# The ordering must be identical to that of `mint.h`
LayerKind = Enum(
    "LayerKind",
    [
        "UNKNOWN",
        "AVG_POOL_2D",
        "CONV_2D",
        "DENSE",
        "FLATTEN",
        "LOCAL_RESPONSE_NORM",
        "MAX_POOL_2D",
        "RELU",
        "SIGMOID",
    ],
    start=0,
)


def onnx_tensor_to_torch(onnx_tensor) -> torch.Tensor:
    return torch.tensor(onnx_tensor.float_data or onnx_tensor.int64_data)


def onnx_tensor_to_numpy(onnx_tensor):
    return onnx.numpy_helper.to_array(onnx_tensor)


ACCEPTED_OPSET = 12


def parse_onnx(filename: str) -> Dict[str, Any]:
    onnx_model = onnx.load_model(filename, load_external_data=True)
    curr_version = onnx_model.opset_import[0].version

    if curr_version != ACCEPTED_OPSET:
        print(
            f"Only opset version {ACCEPTED_OPSET} is supported. Attempting your version ({curr_version}) to {ACCEPTED_OPSET}..."
        )
        onnx_model = onnx.version_converter.convert_version(onnx_model, ACCEPTED_OPSET)

    # Initialize the custom structure
    model_structure = {"tensors": [], "nodes": [], "edges": []}

    # Helper function to convert ONNX tensor to numpy array
    def onnx_tensor_to_numpy(onnx_tensor):
        return onnx.numpy_helper.to_array(onnx_tensor)

    # Process model inputs and initializers
    name_to_id = {}
    for input_info in onnx_model.graph.input:
        name_to_id[input_info.name] = len(model_structure["tensors"])
        model_structure["tensors"].append(None)  # Placeholder for input tensor

    for initializer in onnx_model.graph.initializer:
        name_to_id[initializer.name] = len(model_structure["tensors"])
        model_structure["tensors"].append(onnx_tensor_to_numpy(initializer))

    # Process nodes
    for node in onnx_model.graph.node:
        node_info = {
            "op_type": node.op_type,
            "inputs": [],
            "outputs": [],
            "attributes": {},
            "prev": [],
            "next": [],
        }

        # Process inputs
        for input_name in node.input:
            node_info["inputs"].append(name_to_id[input_name])

        # Process outputs
        for output_name in node.output:
            name_to_id[output_name] = len(model_structure["tensors"])
            model_structure["tensors"].append(None)  # Placeholder for output tensor
            node_info["outputs"].append(name_to_id[output_name])

        # Process attributes
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.FLOAT:
                node_info["attributes"][attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.INT:
                node_info["attributes"][attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.STRING:
                node_info["attributes"][attr.name] = attr.s.decode("utf-8")
            elif attr.type == onnx.AttributeProto.TENSOR:
                node_info["attributes"][attr.name] = onnx_tensor_to_numpy(attr.t)
            elif attr.type == onnx.AttributeProto.FLOATS:
                node_info["attributes"][attr.name] = list(attr.floats)
            elif attr.type == onnx.AttributeProto.INTS:
                node_info["attributes"][attr.name] = list(attr.ints)
            else:
                raise AttributeError(f"{attr.type} not supported")

        # Add node to the structure
        model_structure["nodes"].append(node_info)

    # Process edges and update prev/next information
    tensor_to_nodes = {i: [] for i in range(len(model_structure["tensors"]))}
    for node_id, node_info in enumerate(model_structure["nodes"]):
        for input_id in node_info["inputs"]:
            model_structure["edges"].append((input_id, node_id))
            for prev_node_id in tensor_to_nodes[input_id]:
                model_structure["nodes"][prev_node_id]["next"].append(node_id)
                node_info["prev"].append(prev_node_id)
        for output_id in node_info["outputs"]:
            tensor_to_nodes[output_id].append(node_id)

    return model_structure


def write_ndarray(f: BufferedWriter, x: np.ndarray):
    x = np.ascontiguousarray(x, dtype=np.float32)
    np.array(x.ndim, dtype=np.int32).tofile(f)
    np.array(x.shape, dtype=np.int32).tofile(f)
    f.write(x.tobytes())


def write_layer_header(f: BufferedWriter, kind_val: int, node: Dict[str, Any]):
    # write node info (kind, id, prev, next, inputs, outputs, stride, pad)
    np.array(kind_val, dtype=np.int32).tofile(f)
    np.array(id, dtype=np.int32).tofile(f)
    np.array(len(node["prev"]), dtype=np.int32).tofile(f)
    np.array(node["prev"], dtype=np.int32).tofile(f)
    np.array(len(node["next"]), dtype=np.int32).tofile(f)
    np.array(node["next"], dtype=np.int32).tofile(f)
    np.array(len(node["inputs"]), dtype=np.int32).tofile(f)
    np.array(node["inputs"], dtype=np.int32).tofile(f)
    np.array(len(node["outputs"]), dtype=np.int32).tofile(f)
    np.array(node["outputs"], dtype=np.int32).tofile(f)


def write_relu(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.RELU.value, node)
    print(f"wrote Relu {id}")


def write_lrn(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.LOCAL_RESPONSE_NORM.value, node)
    np.array(node["attributes"]["size"], dtype=np.int32).tofile(f)
    np.array(node["attributes"]["alpha"], dtype=np.float32).tofile(f)
    np.array(node["attributes"]["beta"], dtype=np.float32).tofile(f)
    np.array(node["attributes"]["bias"], dtype=np.float32).tofile(f)
    print(f"wrote LRN {id}")


def write_max_pool(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.MAX_POOL_2D.value, node)
    shape = node["attributes"]["kernel_shape"]
    strides = node["attributes"]["strides"]
    pads = node["attributes"]["pads"]
    assert all(v == shape[0] for v in shape)
    assert all(v == strides[0] for v in strides)
    assert all(v == pads[0] for v in pads), pads

    np.array(shape[0], dtype=np.int32).tofile(f)
    np.array(strides[0], dtype=np.int32).tofile(f)
    np.array(pads[0], dtype=np.int32).tofile(f)
    print(f"MaxPool {id}")


def write_avg_pool_2d(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.AVG_POOL_2D.value, node)
    kernel_shape = node["attributes"]["kernel_shape"]
    strides = node["attributes"]["strides"]
    assert all(v == kernel_shape[0] for v in kernel_shape)
    assert all(v == strides[0] for v in strides)
    # size
    np.array(kernel_shape[0], dtype=np.int32).tofile(f)
    # stride
    np.array(strides[0], dtype=np.int32).tofile(f)
    print(f"AveragePool {id}")


def write_conv(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.CONV_2D.value, node)

    # NOTE: we refuse mutliple different stride and pad values
    strides = node["attributes"]["strides"]
    pads = node["attributes"]["pads"]
    assert all(v == strides[0] for v in strides)
    assert all(v == pads[0] for v in pads)
    np.array(strides[0], dtype=np.int32).tofile(f)
    np.array(pads[0], dtype=np.int32).tofile(f)

    # write w
    w_idx = node["inputs"][1]
    w = tensors[w_idx]
    write_ndarray(f, w)

    # write b
    b_idx = node["inputs"][2]
    b = tensors[b_idx]
    write_ndarray(f, b)
    print(f"wrote Conv {id}")


def write_dense(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.DENSE.value, node)
    trans_a = node["attributes"].get("transA", 0)
    trans_b = node["attributes"].get("transB", 1)

    assert trans_a == 0, "matrix must not be transposed"

    # write w
    w_idx = node["inputs"][1]
    w = tensors[w_idx]
    if trans_b:
        w = w.T
    write_ndarray(f, w)

    # write b
    b_idx = node["inputs"][2]
    b = tensors[b_idx]
    write_ndarray(f, b)
    print(f"wrote Gemm {id}")


def write_flatten(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.FLATTEN.value, node)
    np.array(node["attributes"]["axis"], dtype=np.int32).tofile(f)
    print(f"wrote Flatten {id}")


if __name__ == "__main__":
    MODEL_PATH_IN = "alexnet.ONNX"
    MODEL_PATH_OUT = "alexnet.mt"

    # Load the pre-trained AlexNet model
    alexnet = models.alexnet(pretrained=True)

    # Set the model to evaluation mode
    alexnet.eval()

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export the model to ONNX
    torch.onnx.export(
        alexnet,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        MODEL_PATH_IN,  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )
    model = parse_onnx(MODEL_PATH_IN)

    err = []
    with open(MODEL_PATH_OUT, "wb") as f:
        # Model header
        np.array(len(model["nodes"]), dtype=np.int32).tofile(f)
        np.array(len(model["tensors"]), dtype=np.int32).tofile(f)

        for id, node in enumerate(model["nodes"]):
            match node["op_type"]:
                case "AveragePool":
                    write_avg_pool_2d(f, id, node, model["tensors"])
                case "Conv":
                    write_conv(f, id, node, model["tensors"])
                case "Flatten":
                    write_flatten(f, id, node, model["tensors"])
                case "Gemm":
                    write_dense(f, id, node, model["tensors"])
                case "LRN":
                    write_lrn(f, id, node, model["tensors"])
                case "MaxPool":
                    write_max_pool(f, id, node, model["tensors"])
                case "Relu":
                    write_relu(f, id, node, model["tensors"])
                case _:
                    if not node["op_type"] in err:
                        err.append(node["op_type"])
    if err:
        print(f"following layer kinds were unable to be processed: {err}")
        to_keep = input("keep model? (Y/n) ").strip().lower() == "y"
        if not to_keep:
            os.remove(MODEL_PATH_OUT)
            print("removed")
