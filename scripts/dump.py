import os
import sys
from enum import Enum
from io import BufferedWriter
from typing import Any, Dict, List

import math
import numpy as np
import onnx
import torch

ACCEPTED_OPSET = 12
MAX_INPUT_OUTPUT_NAME_LEN = 50

# The ordering must be identical to that of `mint.h`
LayerKind = Enum(
    "LayerKind",
    [
        "UNKNOWN",
        "ADD",
        "AVG_POOL_2D",
        "CAST",
        "CONCAT",
        "CONSTANT",
        "CONV_2D",
        "DENSE",
        "DIV",
        "DROPOUT",
        "EXP",
        "FLATTEN",
        "GLOBAL_AVG_POOL",
        "INSTANCE_NORMALIZATION",
        "LEAKY_RELU",
        "LOCAL_RESPONSE_NORM",
        "LOG",
        "MAX_POOL_2D",
        "MUL",
        "PAD",
        "POW",
        "RELU",
        "RESHAPE",
        "RESIZE",
        "SIGMOID",
        "SLICE",
        "SOFTMAX",
        "SPLIT",
        "SUB",
        "TANH",
        "TRANSPOSE",
    ],
    start=0,
)


def onnx_tensor_to_torch(onnx_tensor) -> torch.Tensor:
    return torch.tensor(onnx_tensor.float_data or onnx_tensor.int64_data)


def onnx_tensor_to_numpy(onnx_tensor):
    return onnx.numpy_helper.to_array(onnx_tensor)


def parse_onnx(filename: str) -> Dict[str, Any]:
    onnx_model = onnx.load_model(filename, load_external_data=True)
    curr_version = onnx_model.opset_import[0].version

    if curr_version != ACCEPTED_OPSET:
        print(
            f"Only opset version {ACCEPTED_OPSET} is supported. Attempting your version ({curr_version}) to {ACCEPTED_OPSET}..."
        )
        onnx_model = onnx.version_converter.convert_version(onnx_model, ACCEPTED_OPSET)

    # Initialize the custom structure
    model_structure = {
        "tensors": [],
        "nodes": [],
        "edges": [],
        "inputs": [],
        "outputs": [],
    }

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
        val = onnx_tensor_to_numpy(initializer)
        model_structure["tensors"].append(val)

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

    # list model input and output (name, id) pairs
    inputs = [(i.name, name_to_id[i.name]) for i in onnx_model.graph.input]
    outputs = [(o.name, name_to_id[o.name]) for o in onnx_model.graph.output]
    model_structure["inputs"] = inputs
    model_structure["outputs"] = outputs

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


def write_log(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.LOG.value, node)
    print(f"wrote Log {id}")


def write_instance_normalization(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.INSTANCE_NORMALIZATION.value, node)
    eps = node["attributes"].get("epsilon", 1e-05)
    np.array(eps, dtype=np.float32).tofile(f)

    # scale_idx = node["inputs"][1]
    # scale = tensors[scale_idx]
    # b_idx = node["inputs"][2]
    # b = tensors[b_idx]
    # write_ndarray(f, scale)
    # write_ndarray(f, b)
    print(f"wrote InstanceNormalization {id}")


def write_leaky_relu(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.LEAKY_RELU.value, node)
    np.array(node["attributes"].get("alpha", 0.01), dtype=np.float32).tofile(f)
    print(f"wrote LeakyRelu {id}")


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

    pads = node["attributes"].get("pads", [0]*4)
    auto_pad = node["attributes"].get("auto_pad", "NOTSET")

    auto_pad_id = 0
    match auto_pad:
        case "NOTSET": auto_pad_id = 0
        case "VALID": auto_pad_id = 1
        case "SAME_UPPER": auto_pad_id = 2
        case "SAME_LOWER": auto_pad_id = 3

    assert all(v == shape[0] for v in shape)
    assert all(v == strides[0] for v in strides)

    np.array(shape[0], dtype=np.int32).tofile(f)
    np.array(strides[0], dtype=np.int32).tofile(f)
    np.array(auto_pad_id, dtype=np.int32).tofile(f)
    np.array(pads, dtype=np.int32).tofile(f)
    print(f"wrote MaxPool {id}")


def write_add(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.ADD.value, node)
    print(f"wrote Add {id}")


def write_concat(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.CONCAT.value, node)
    np.array(node["attributes"]["axis"], dtype=np.int32).tofile(f)
    print(f"wrote Concat {id}")


def write_constant(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    # Move out attribute `value` into tensors
    out_idx = node["outputs"][0]
    tensors[out_idx] = node["attributes"]["value"]
    write_layer_header(f, LayerKind.CONSTANT.value, node)
    print(f"wrote Constant {id}")


def write_sub(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.SUB.value, node)
    print(f"wrote Sub {id}")


def write_mul(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.MUL.value, node)
    print(f"wrote Mul {id}")


def write_div(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.DIV.value, node)
    print(f"wrote Div {id}")


def write_dropout(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.DROPOUT.value, node)
    print("Currently dropout does not write any information")
    print(f"wrote Dropout {id}")


def write_exp(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.EXP.value, node)
    print(f"wrote Exp {id}")


def write_avg_pool_2d(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.AVG_POOL_2D.value, node)
    kernel_shape = node["attributes"]["kernel_shape"]
    strides = node["attributes"]["strides"]
    pads = node["attributes"].get("pads", [0, 0, 0, 0])
    assert all(v == kernel_shape[0] for v in kernel_shape)
    assert all(v == strides[0] for v in strides)
    assert all(v == pads[0] for v in pads)
    # size
    np.array(kernel_shape[0], dtype=np.int32).tofile(f)
    # stride
    np.array(strides[0], dtype=np.int32).tofile(f)
    # pad
    np.array(pads[0], dtype=np.int32).tofile(f)
    print(f"wrote AveragePool {id}")


def write_conv(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.CONV_2D.value, node)

    dilations = node["attributes"].get("dilations", [1, 1])
    group = node["attributes"].get("group", 1)

    # NOTE: we refuse mutliple different stride values for now
    strides = node["attributes"]["strides"]

    pads = node["attributes"].get("pads", [0]*4)
    auto_pad = node["attributes"].get("auto_pad", "NOTSET")

    auto_pad_id = 0
    match auto_pad:
        case "NOTSET": auto_pad_id = 0
        case "VALID": auto_pad_id = 1
        case "SAME_UPPER": auto_pad_id = 2
        case "SAME_LOWER": auto_pad_id = 3


    assert all(v == strides[0] for v in strides)
    np.array(strides[0], dtype=np.int32).tofile(f)
    np.array(auto_pad_id, dtype=np.int32).tofile(f)
    np.array(pads, dtype=np.int32).tofile(f)
    np.array(dilations, dtype=np.int32).tofile(f)
    np.array(group, dtype=np.int32).tofile(f)

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
    if trans_b:
        tensors[w_idx] = tensors[w_idx].T

    print(f"wrote Gemm {id}")


def write_flatten(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.FLATTEN.value, node)
    np.array(node["attributes"]["axis"], dtype=np.int32).tofile(f)
    print(f"wrote Flatten {id}")


def write_global_avg_pool(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.GLOBAL_AVG_POOL.value, node)
    print(f"wrote GlobalAveragePool {id}")


def write_pad(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.PAD.value, node)

    # mode = node["attributes"]["mode"].lower()
    # match mode:
    #     case "reflect":
    #         np.array(0, dtype=np.int32).tofile(f)

    print(f"wrote Pad {id}")


def write_pow(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.POW.value, node)
    print(f"wrote Pow {id}")

def write_relu(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.RELU.value, node)
    print(f"wrote Relu {id}")


def write_reshape(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.RESHAPE.value, node)
    print(f"wrote Reshape {id}")


def write_resize(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.RESIZE.value, node)
    mode = node["attributes"]["mode"].lower()

    mode_idx = 0
    match mode:
        case "linear":
            mode_idx = 0
        case "nearest":
            mode_idx = 1
        case "cubic":
            mode_idx = 2

    np.array(mode_idx, dtype=np.int32).tofile(f)

    print(f"wrote Resize {id}")


def write_sigmoid(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.SIGMOID.value, node)
    print(f"wrote Sigmoid {id}")

def write_slice(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.SLICE.value, node)
    print(f"wrote Slice {id}")

def write_softmax(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.SOFTMAX.value, node)
    np.array(node["attributes"].get("axis", -1), dtype=np.int32).tofile(f)
    print(f"wrote Softmax {id}")


def write_split(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.SPLIT.value, node)
    axis = node["attributes"]["axis"]
    splits = node["attributes"]["split"]
    np.array(axis, dtype=np.int32).tofile(f)
    np.array(len(splits), dtype=np.int32).tofile(f)
    np.array(splits, dtype=np.int32).tofile(f)
    print(f"wrote Split {id}")


def write_tanh(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.TANH.value, node)
    print(f"wrote Tanh {id}")


def write_transpose(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.TRANSPOSE.value, node)
    perm = node["attributes"]["perm"]
    ndim = len(perm)

    np.array(ndim, dtype=np.int32).tofile(f)
    np.array(perm, dtype=np.int32).tofile(f)
    print(f"wrote Transpose {id}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("dump.py requires one argument, i.e., the ONNX model path")
        exit(1)

    model_path_in = sys.argv[1]
    model_path_out = model_path_in.replace(".onnx", ".mt")

    model = parse_onnx(model_path_in)

    err = []
    with open(model_path_out, "wb") as f:
        # Write model headers
        np.array(len(model["nodes"]), dtype=np.int32).tofile(f)

        # Write node data
        for id, node in enumerate(model["nodes"]):
            match node["op_type"]:
                # Simple binop
                case "Add":
                    write_add(f, id, node, model["tensors"])
                case "Concat":
                    write_concat(f, id, node, model["tensors"])
                case "Constant":
                    write_constant(f, id, node, model["tensors"])
                case "Sub":
                    write_sub(f, id, node, model["tensors"])
                case "Mul":
                    write_mul(f, id, node, model["tensors"])
                case "Div":
                    write_div(f, id, node, model["tensors"])
                case "Dropout":
                    write_dropout(f, id, node, model["tensors"])
                case "Exp":
                    write_exp(f, id, node, model["tensors"])
                case "AveragePool":
                    write_avg_pool_2d(f, id, node, model["tensors"])
                case "Conv":
                    write_conv(f, id, node, model["tensors"])
                case "Flatten":
                    write_flatten(f, id, node, model["tensors"])
                case "Gemm":
                    write_dense(f, id, node, model["tensors"])
                case "GlobalAveragePool":
                    write_global_avg_pool(f, id, node, model["tensors"])
                case "InstanceNormalization":
                    write_instance_normalization(f, id, node, model["tensors"])
                case "LeakyRelu":
                    write_leaky_relu(f, id, node, model["tensors"])
                case "LRN":
                    write_lrn(f, id, node, model["tensors"])
                case "Log":
                    write_log(f, id, node, model["tensors"])
                case "MaxPool":
                    write_max_pool(f, id, node, model["tensors"])
                case "Pad":
                    write_pad(f, id, node, model["tensors"])
                case "Pow":
                    write_pow(f, id, node, model["tensors"])
                case "Relu":
                    write_relu(f, id, node, model["tensors"])
                case "Reshape":
                    write_reshape(f, id, node, model["tensors"])
                case "Resize":
                    write_resize(f, id, node, model["tensors"])
                case "Sigmoid":
                    write_sigmoid(f, id, node, model["tensors"])
                case "Slice":
                    write_slice(f, id, node, model["tensors"])
                case "Softmax":
                    write_softmax(f, id, node, model["tensors"])
                case "Split":
                    write_split(f, id, node, model["tensors"])
                case "Tanh":
                    write_tanh(f, id, node, model["tensors"])
                case "Transpose":
                    write_transpose(f, id, node, model["tensors"])
                case _:
                    if not node["op_type"] in err:
                        err.append(node["op_type"])

        # write initializers (data, id)
        non_null_tensors = [(i, t) for i, t in enumerate(model["tensors"]) if t is not None]
        initializer_len = len(non_null_tensors)
        np.array(initializer_len, dtype=np.int32).tofile(f)
        for i, t in non_null_tensors:
            if t is None:
                continue
            else:
                write_ndarray(f, t)
                np.array(i, dtype=np.int32).tofile(f)

        # write input and output lens, and (name, id) info
        print("Writing initializers...")
        input_len = len(model["inputs"])
        np.array(input_len, dtype=np.int32).tofile(f)
        for name, id in model["inputs"]:
            np.frombuffer(
                name.encode("utf-8").ljust(MAX_INPUT_OUTPUT_NAME_LEN, b"\0"), dtype="S1"
            ).tofile(f)
            np.array(id, dtype=np.int32).tofile(f)

        print("Writing input and output info...")
        output_len = len(model["outputs"])
        np.array(output_len, dtype=np.int32).tofile(f)
        for name, id in model["outputs"]:
            np.frombuffer(
                name.encode("utf-8").ljust(MAX_INPUT_OUTPUT_NAME_LEN, b"\0"), dtype="S1"
            ).tofile(f)
            np.array(id, dtype=np.int32).tofile(f)

    if err:
        print(f"following layer kinds were unable to be processed: {err}")
        to_keep = input("keep model? (Y/n) ").strip().lower() == "y"
        if not to_keep:
            os.remove(model_path_out)
            print("removed")

    print("Done")
