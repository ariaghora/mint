import os
import sys
from enum import Enum
from io import BufferedWriter
from typing import Any, Dict, List

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
        "CONV_2D",
        "DENSE",
        "DIV",
        "EXP",
        "FLATTEN",
        "GLOBAL_AVG_POOL",
        "LOCAL_RESPONSE_NORM",
        "MAX_POOL_2D",
        "MUL",
        "RELU",
        "SIGMOID",
        "SUB",
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
    print(f"wrote MaxPool {id}")


def write_add(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.ADD.value, node)
    print(f"wrote Add {id}")


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


def write_global_avg_pool(
    f: BufferedWriter, id: int, node: Dict[str, Any], tensors: List[np.ndarray]
):
    write_layer_header(f, LayerKind.GLOBAL_AVG_POOL.value, node)
    print(f"wrote GlobalAveragePool {id}")


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
        np.array(len(model["tensors"]), dtype=np.int32).tofile(f)

        # Write node data
        for id, node in enumerate(model["nodes"]):
            match node["op_type"]:
                # Simple binop
                case "Add":
                    write_add(f, id, node, model["tensors"])
                case "Sub":
                    write_sub(f, id, node, model["tensors"])
                case "Mul":
                    write_mul(f, id, node, model["tensors"])
                case "Div":
                    write_div(f, id, node, model["tensors"])

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
                case "LRN":
                    write_lrn(f, id, node, model["tensors"])
                case "MaxPool":
                    write_max_pool(f, id, node, model["tensors"])
                case "Relu":
                    write_relu(f, id, node, model["tensors"])
                case _:
                    if not node["op_type"] in err:
                        err.append(node["op_type"])

        # write input and output lens, and (name, id) info
        input_len = len(model["inputs"])
        np.array(input_len, dtype=np.int32).tofile(f)
        for name, id in model["inputs"]:
            np.frombuffer(
                name.encode("utf-8").ljust(MAX_INPUT_OUTPUT_NAME_LEN, b"\0"), dtype="S1"
            ).tofile(f)
            np.array(id, dtype=np.int32).tofile(f)

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
