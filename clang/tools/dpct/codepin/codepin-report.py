#!/usr/bin/env python3
#
# Copyright (C) Intel Corporation
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from multiprocessing import freeze_support
import argparse
import json
import os
import sys
from collections.abc import Container
import math
from argparse import RawTextHelpFormatter

UUID = "ID"
CHECKPOINT = "CheckPoint"
DATA = "Data"
TYPE = "Type"
FREE_MEM = "Free Device Memory"
TOTAL_MEM = "Total Device Memory"
TIME_ELAPSED = "Elapse Time(ms)"
DEVICE_NAME = "Device Name"
DEVICE_ID = "Device ID"
STREAM_ADDRESS = "Stream Address"
INDEX = "Index"
ADDRESS = "Address"
ERROR_MATCH_PATTERN = "Unable to find the corresponding serialization function"
CODEPIN_REPORT_FILE = os.path.join(os.getcwd(), "CodePin_Report.csv")
GRAPH_FILENAME = "CodePin_DataFlowGraph"
CODEPIN_GRAPH_FILE_PATH = os.path.join(os.getcwd())
ERROR_CSV_PATTERN = "CUDA Meta Data ID, SYCL Meta Data ID, Type, Detail\n"
EPSILON_FILE = ""
CODEPIN_RANDOM_SEED = "CodePin Random Seed"
CODEPIN_SAMPLING_THRESHOLD = "CodePin Sampling Threshold"
CODEPIN_SAMPLING_PERCENT = "CodePin Sampling Percent"
CODEPIN_RANDOM_SEED_VALUE = "N/A"
CODEPIN_SAMPLING_THRESHOLD_VALUE = "N/A"
CODEPIN_SAMPLING_PERCENT_VALUE = "N/A"

# Reference: https://en.wikipedia.org/wiki/Machine_epsilon
# bfloat16 reference: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
default_epsilons = {
    "bf16_abs_tol": 7.81e-03,  # 2^-7
    "fp16_abs_tol": 9.77e-04,  # 2^-10
    "float_abs_tol": 1.19e-07,  # 2^-23
    "double_abs_tol": 2.22e-16,  # 2^-52
    "rel_tol": 1e-3,
}

mismatch_var_map = {"epi" : {}, "pro" : {}}

# Raise the warning message when the data is not matched.
def data_value_dismatch_error(value1, value2):
    return comparison_error(
        f" and [ERROR: DATA VALUE MISMATCH] the CUDA value {value1} differs from the SYCL value {value2}."
    )


def no_serialization_function_error():
    return comparison_error(
        f" and [ERROR: NO SERIALIZATION FUNCTION]the CUDA or SYCL value cannot be dumped, lack of dump function. Please report it to the tool developer."
    )


def data_missed_error(name):
    return comparison_error(
        f" and [ERROR: DATA MISSED] Cannot find the {name} in SYCL Json.\n"
    )


def data_length_dismatch_error():
    return comparison_error(
        f" and [ERROR: DATA LENGTH MISMATCH] The size of CUDA and SYCL data are mismatch."
    )


def print_checkpoint_length_dismatch_warning(cuda_list, sycl_list):
    print(
        f"[ERROR: CHECKPOINT LENGTH MISMATCH] \n CUDA CodePin list length is: {len(cuda_list)}. \n SYCL CodePin list length is: {len(sycl_list)}.\n"
    )


def prepare_failed_log(cuda_id, sycl_id, log_type, detail):
    return ",".join([cuda_id, sycl_id, log_type, detail]) + "\n"


def prolog_dismatch_but_epilog_match(id):
    api_name = id.split(":")[0]
    detail = f"[WARNING: METADATA MISMATCH] The pair of prolog data {id} are mismatched, and the corresponding pair of epilog data matches. This mismatch may be caused by the initialized memory or argument used in the API {api_name}.\n"
    return prepare_failed_log(id, id, "Data value", detail)


def get_missing_key_log(id):
    detail = f"[ERROR: METADATA MISSING] Cannot find the checkpoint: {id} in the execution log dataset of instrumented SYCL code.\n"
    return prepare_failed_log(id, "Missing", "Execution path", detail)


def get_data_value_mismatch_log(id, message):
    return prepare_failed_log(
        id, id, "Data value", "The location of failed ID " + message
    )


def is_container(obj):
    return isinstance(obj, Container) and not isinstance(obj, str)


def is_both_container(obj1, obj2):
    return is_container(obj1) and is_container(obj2)


def is_container_with_type(obj1, obj2, obj_type):
    if is_both_container(obj1, obj2):
        if type(obj1) == type(obj2) and type(obj1) == obj_type:
            return True
    return False


class comparison_error(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def compare_float_value(data1, data2, type):
    global EPSILON_FILE
    if EPSILON_FILE is None:
        epsilons = default_epsilons
    else:
        try:
            with open(EPSILON_FILE, "r") as f:
                epsilons = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            epsilons = default_epsilons
    abs_type = type + "_abs_tol"
    if abs_type not in epsilons:
        raise comparison_error(f" type '{type}' is not supported yet.")
    abs_tol = epsilons[abs_type]
    rel_tol = epsilons["rel_tol"]
    if not abs_tol:
        abs_tol = default_epsilons[abs_type]
    if not rel_tol:
        rel_tol = default_epsilons["rel_tol"]
    if not math.isclose(data1, data2, rel_tol=rel_tol, abs_tol=abs_tol):
        raise comparison_error(
            f": {type} values {data1} and {data2} are not close enough [Floating Point comparison fail]"
        )

    return True


def compare_data_value(data1, data2, var_name, var_type):
    if data1 == ERROR_MATCH_PATTERN or data2 == ERROR_MATCH_PATTERN:
        raise no_serialization_function_error()
    try:
        if var_type in ["bf16", "fp16", "float", "double"]:
            compare_float_value(data1, data2, var_type)
        elif data1 != data2:
            raise data_value_dismatch_error(data1, data2)
    except comparison_error as e:
        raise comparison_error(f"{var_name}{e.message}")


def compare_list_value(cuda_list, sycl_list, var_name, var_type):
    for i in range(len(cuda_list)):
        local_var_name = var_name + "->[" + str(i) + "]"
        if is_both_container(cuda_list[i], sycl_list[i]):
            compare_container_value(
                cuda_list[i], sycl_list[i], local_var_name, var_type
            )
            continue
        else:
            compare_data_value(cuda_list[i], sycl_list[i], local_var_name, var_type)
            continue


def compare_dict_value(cuda_dict, sycl_dict, var_name, var_type):
    for name, data in cuda_dict.items():
        if name == INDEX or name == ADDRESS:
            continue
        if name not in sycl_dict:
            raise data_missed_error(name)
        if is_both_container(data, sycl_dict[name]):
            local_var_name = var_name + '->"' + name + '"'
            compare_container_value(data, sycl_dict[name], local_var_name, var_type)
            continue
        else:
            if name == TYPE:  # Check the Data only, ignore the key is 'Type'
                var_type = data
                continue
            local_var_name = var_name + '->"' + name + '"'
            compare_data_value(data, sycl_dict[name], local_var_name, var_type)


def compare_container_value(cuda_value, sycl_value, var_name, var_type=""):
    if len(cuda_value) != len(sycl_value):
        raise data_length_dismatch_error()
    if is_container_with_type(cuda_value, sycl_value, list):
        return compare_list_value(cuda_value, sycl_value, var_name, var_type)
    elif is_container_with_type(cuda_value, sycl_value, dict):
        return compare_dict_value(cuda_value, sycl_value, var_name, var_type)


def compare_cp_var(cuda_var, sycl_var, var_name):
    compare_container_value(cuda_var, sycl_var, var_name)


def compare_checkpoint(phase, index, cuda_checkpoint, sycl_checkpoint):
    error_messages = []
    mismatch_var_map[phase][index] = {}
    for var_name, cuda_var in cuda_checkpoint.items():
        sycl_var = sycl_checkpoint.get(var_name)
        if cuda_var is not None and sycl_var is not None:
            try:
                compare_cp_var(cuda_var, sycl_var, var_name)
            except Exception as e:
                mismatch_var_map[phase][index][var_name] = True
                error_messages.append(str(e))
            continue
    if error_messages:
        raise comparison_error(
            "Errors occurred during comparison: " + "; ".join(error_messages)
        )


def is_checkpoint_length_dismatch(cuda_list, sycl_list):
    if len(cuda_list) != len(sycl_list):
        print_checkpoint_length_dismatch_warning(cuda_list, sycl_list)

def compare_checkpoint_list(
    ordered_pro_id_cuda,
    ordered_epi_id_cuda,
    ordered_pro_id_sycl,
    ordered_epi_id_sycl,
    cuda_prolog_checkpoint_list,
    cuda_epilog_checkpoint_list,
    sycl_prolog_checkpoint_list,
    sycl_epilog_checkpoint_list,
    match_checkpoint_num,
    dismatch_checkpoint_num,
    checkpoint_size,
):
    failed_log = ""
    is_checkpoint_length_dismatch(
        cuda_prolog_checkpoint_list, sycl_prolog_checkpoint_list
    )
    is_checkpoint_length_dismatch(
        cuda_epilog_checkpoint_list, sycl_epilog_checkpoint_list
    )
    failed_epilog = []
    cuda_epi_size = len(ordered_epi_id_cuda)
    for index in range(cuda_epi_size):
        id = ordered_epi_id_cuda[index]
        cuda_epilog_checkpoint = cuda_epilog_checkpoint_list[id]
        checkpoint_size += 1
        if not sycl_epilog_checkpoint_list.get(id):
            failed_log += get_missing_key_log(id)
            dismatch_checkpoint_num += 1
            continue
        try:
            sycl_id = ordered_epi_id_sycl[index]
            compare_checkpoint(
                "epi", index, cuda_epilog_checkpoint, sycl_epilog_checkpoint_list.get(sycl_id)
            )
            match_checkpoint_num += 1
        except comparison_error as e:
            failed_log += get_data_value_mismatch_log(id, e.message)
            failed_epilog.append(id)
            dismatch_checkpoint_num += 1
            continue
    cuda_pro_size = len(ordered_pro_id_cuda)
    for index in range(cuda_pro_size):
        id = ordered_pro_id_cuda[index]
        cuda_prolog_checkpoint = cuda_prolog_checkpoint_list[id]
        checkpoint_size += 1
        if not sycl_prolog_checkpoint_list.get(id):
            failed_log += get_missing_key_log(id)
            dismatch_checkpoint_num += 1
            continue
        try:
            sycl_id = ordered_pro_id_sycl[index]
            compare_checkpoint(
                "pro", index, cuda_prolog_checkpoint, sycl_prolog_checkpoint_list.get(sycl_id)
            )
            match_checkpoint_num += 1
        except comparison_error as e:
            # Check if the pair of epilog data {id} are mismatched
            # If the pair of epilog data is matched, then skip the prolog data check.
            epilog_id = id.replace(":prolog:", ":epilog:")
            if epilog_id in failed_epilog:
                failed_log += get_data_value_mismatch_log(id, e.message)
                dismatch_checkpoint_num += 1
            else:
                failed_log += prolog_dismatch_but_epilog_match(id)
            continue
    return match_checkpoint_num, dismatch_checkpoint_num, checkpoint_size, failed_log


def parse_json(json_str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e.msg}")
        print(f"At line {e.lineno}, column {e.colno}")
        return None


def read_data_from_json_file(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit(2)
    with open(file_path) as f:
        dc_array = parse_json(f.read())
        if len(dc_array) > 0 and CODEPIN_RANDOM_SEED in dc_array[0]:
            global CODEPIN_RANDOM_SEED_VALUE
            CODEPIN_RANDOM_SEED_VALUE = dc_array[0][CODEPIN_RANDOM_SEED]
            global CODEPIN_SAMPLING_THRESHOLD_VALUE
            CODEPIN_SAMPLING_THRESHOLD_VALUE = dc_array[0][CODEPIN_SAMPLING_THRESHOLD]
            global CODEPIN_SAMPLING_PERCENT_VALUE
            CODEPIN_SAMPLING_PERCENT_VALUE = dc_array[0][CODEPIN_SAMPLING_PERCENT]
            return dc_array[1:]
        return dc_array


def get_checkpoint_list_from_json_file(file_path):
    checkpoint_list = {}
    used_mem_dic = {}
    elapsed_time_dic = {}
    json_data_list = read_data_from_json_file(file_path)
    prolog_checkpoint_list = {}
    epilog_checkpoint_list = {}
    ordered_pro_id = []
    ordered_epi_id = []
    device_stream_dic = {}
    for item in json_data_list:
        id = item[UUID]
        if "prolog" in id:
            ordered_pro_id.append(id)
            prolog_checkpoint_list[id] = item.get(CHECKPOINT, {})
        elif "epilog" in id:
            ordered_epi_id.append(id)
            epilog_checkpoint_list[id] = item.get(CHECKPOINT, {})
        device_stream_dic[id] = (item[DEVICE_ID], item[STREAM_ADDRESS], item[DEVICE_NAME])
        total_mem = item[TOTAL_MEM]
        free_mem = item[FREE_MEM]
        used_mem = int(total_mem) - int(free_mem)
        used_mem_dic[id] = used_mem
        time_elapsed = item[TIME_ELAPSED]
        elapsed_time_dic[id] = float(time_elapsed)
    return (
        prolog_checkpoint_list,
        epilog_checkpoint_list,
        used_mem_dic,
        elapsed_time_dic,
        device_stream_dic,
        ordered_pro_id,
        ordered_epi_id
    )


def get_bottleneck(cp_list):
    bottleneck_id = "N/A"
    if len(cp_list) > 0:
        bottleneck_id = list(cp_list.keys())[0]
    max_time = 0.0
    for id, time in cp_list.items():
        if time > max_time:
            bottleneck_id = id
            max_time = time
    return (bottleneck_id, max_time)


def get_memory_used(cp_list):
    cp_id = "N/A"
    if len(cp_list) > 0:
        cp_id = list(cp_list.keys())[0]
    max_mem = 0
    for id, used_mem in cp_list.items():
        if used_mem > max_mem:
            cp_id = id
            max_mem = used_mem
    return (cp_id, max_mem)

# Generates a data flow graph using the graphviz library to visualize the kernel execution status
# and comparison results between CUDA and SYCL dumped data.
# 1. Identify Kernel Outputs
#    The function iterates over the list of CUDA checkpoints to identify output variables for each kernel. 
#    It compares the prolog and epilog checkpoints to determine if a variable has changed, indicating it is 
#    an output.
# 2. Analyze Checkpoints to Build Layers
#    It iterates over the list of CUDA checkpoints again to build the layers of the graph. A layer refers to
#    a specific grouping of nodes that represents a specific kernel execution process.
#    For each layer, it has three sub-layers:
#      - Input Variable Nodes Layer: The input nodes for each kernel and their attributes (name, type, address, 
#        color) are stored.
#      - Kernel Node Layer: Information about the kernel node, such as stream ID and source location, is collected.
#      - Output Variable Nodes Layer: The identified output nodes for each kernel and their attributes are 
#        also stored.
# 3. Create Graph Nodes and Edges
#    After building the layers, the function creates the graph nodes and edges. The nodes, including device info 
#    nodes, input nodes, kernel nodes, and output nodes, are added to the graph. Edges are added between input 
#    nodes and kernel nodes, and between kernel nodes and output nodes to represent the data flow. Each variable 
#    node will be tagged with V + num, where num is the version of the variable. The initial version is 0, and 
#    num will be incremented by one when it changes. The color of the node is red if the variable value is 
#    mismatched between CUDA and SYCL execution results.
# 4. Render Graph
#    Finally, the function renders the graph into a PDF file and saves it in the current directory.
def generate_data_flow_graph(
    device_stream_dic_cuda,
    device_stream_dic_sycl,
    elapse_time_cuda,
    elapse_time_sycl,
    ordered_pro_id_cuda,
    ordered_epi_id_cuda,
    ordered_pro_id_sycl,
    ordered_epi_id_sycl,
    cuda_prolog_checkpoint_list,
    cuda_epilog_checkpoint_list,
    sycl_prolog_checkpoint_list,
    sycl_epilog_checkpoint_list
):
    try:
        import graphviz as gv
    except ImportError:
        print("Module graphviz is not installed in the current environment, which is required to generate data flow graph. Please use package installer for Python like \'pip install graphviz\' to install it.")
        return
    dot = gv.Digraph(GRAPH_FILENAME)
    dot.attr(layout='nop2')
    list_size = len(cuda_prolog_checkpoint_list)
    datapoint_map = {}
    layers = {}
    stream_id_map = {}
    max_input_node_num = {}
    device_num = 0
    kernel_output_map = {}
    device_name_map_cuda = {}
    device_name_map_sycl = {}
# 1. Identify Kernel Outputs
    for index in range(list_size):
        pid = ordered_pro_id_cuda[index]
        segs = pid.split(":")
        if segs[0] not in kernel_output_map:
            kernel_output_map[segs[0]] = {}
        pcheckpoint = cuda_prolog_checkpoint_list[pid]
        echeckpoint = cuda_epilog_checkpoint_list[ordered_epi_id_cuda[index]]
        for arg in pcheckpoint:
            if (pcheckpoint[arg]["Type"] == "Pointer") or (pcheckpoint[arg]["Type"] == "Array"):
                if pcheckpoint[arg] != echeckpoint[arg]:
                    kernel_output_map[segs[0]][pcheckpoint[arg]["Index"]] = True
# 2. Analyze Checkpoints to Build Layers
    for index in range(list_size):
        pid = ordered_pro_id_cuda[index]
        pid_sycl = ordered_pro_id_sycl[index]
        segs = pid.split(":")
        pcheckpoint = cuda_prolog_checkpoint_list[pid]
        echeckpoint = cuda_epilog_checkpoint_list[ordered_epi_id_cuda[index]]
        input_nodes = []
        output_nodes = []
        kernel_node = {}
        device = int(device_stream_dic_cuda[pid][0])
        stream = device_stream_dic_cuda[pid][1]
        if device not in device_name_map_cuda:
            device_name_map_cuda[device] = device_stream_dic_cuda[pid][2]
            device_name_map_sycl[device] = device_stream_dic_sycl[pid_sycl][2]
        if device not in max_input_node_num:
            max_input_node_num[device] = 0
        if device not in layers:
            device_num += 1
            layers[device] = []
        if device not in stream_id_map:
            stream_id_map[device] = {}
        if stream in stream_id_map[device]:
            stream_id = stream_id_map[device][stream]
        else:
            stream_id_map[device][stream] = len(stream_id_map[device])
            stream_id = stream_id_map[device][stream]
        kernel_node["stream"] = stream_id
        kernel_node["index"] = segs[0] + "_V" + str(index)
        kernel_node["comment"] = "Kernel: " + segs[0] + "\nType: CUDA" + "\nDevice Name: " + \
                                 device_name_map_cuda[device] + "\nDevice Id: " + str(device) + \
                                 "\nStream Id: " + str(stream_id) + "\nElapse Time(ms): " + \
                                 str(elapse_time_cuda[ordered_epi_id_cuda[index]]) + "\nLocation: " + \
                                 segs[1] + ":" + segs[2] + ":" + segs[3]
        for arg in pcheckpoint:
            input_node = {}
            output_node = {}
            input_node["name"] = arg
            if arg in mismatch_var_map["pro"][index]:
                input_node["color"] = "red"
            else :
                input_node["color"] = "black"
            input_node["type"] = pcheckpoint[arg]["Type"]
            input_node["address"] = pcheckpoint[arg]["Address"]
            if input_node["address"] not in datapoint_map :
                datapoint_map[input_node["address"]] = 0
            input_node["version"] = str(datapoint_map[input_node["address"]])
            if (input_node["type"] == "Pointer") | (input_node["type"] == "Array"):
                if (pcheckpoint[arg] != echeckpoint[arg]) or (pcheckpoint[arg]["Index"] in kernel_output_map[segs[0]]):
                    datapoint_map[input_node["address"]] += 1
                    output_node["name"] = arg
                    if arg in mismatch_var_map["epi"][index]:
                        output_node["color"] = "red"
                    else :
                        output_node["color"] = "black"
                    output_node["type"] = echeckpoint[arg]["Type"]
                    output_node["address"] = echeckpoint[arg]["Address"]
                    output_node["version"] = str(datapoint_map[input_node["address"]])
            input_nodes.append(input_node)
            if len(output_node):
                output_nodes.append(output_node)
        if len(input_node) > max_input_node_num[device]:
          max_input_node_num[device] = len(input_node)
        layers[device].append({"in" : input_nodes, "out" : output_nodes, "kernel" : kernel_node})
# 3. Create Graph Nodes and Edges
    layer_right = 0
    device_num = len(stream_id_map)
    total_width = 0
    for device_id in range(device_num):
        stream_num = len(stream_id_map[device_id])
        layer_width = max(stream_num * 1000, max_input_node_num[device_id] * 400)
        total_width += layer_width
        layer_right += layer_width
        layer_top = -50
        stream_step = layer_width / stream_num
        layer_size = len(layers[device_id])
        for index in range(layer_size):
            input_node_num = len(layers[device_id][index]["in"])
            if input_node_num:
                input_right_step = layer_width / input_node_num
                input_right_pos = input_right_step / 2
                for node in layers[device_id][index]["in"]:
                    dot.node(node["name"] + str(index) + "i", node["name"] + ":V" + str(node["version"]) + ":" + node["address"], color=node["color"], penwidth="3" if node["color"] == "red" else "1", pos=str(input_right_pos) + "," + str(layer_top) + "!")
                    input_right_pos += input_right_step
                layer_top -= 150

            kernel_node = layers[device_id][index]["kernel"]
            dot.node(kernel_node["index"], kernel_node["comment"], shape="box", pos=str(stream_step * kernel_node["stream"] + stream_step / 2) + "," + str(layer_top) + "!")
            layer_top -= 150

            output_node_num = len(layers[device_id][index]["out"])
            if output_node_num:
                output_right_step = layer_width / output_node_num
                output_right_pos = output_right_step / 2
                for node in layers[device_id][index]["out"]:
                    dot.node(node["name"] + str(index) + "o", node["name"] + ":V" + str(node["version"]) + ":" + node["address"], color=node["color"], penwidth="3" if node["color"] == "red" else "1", pos=str(output_right_pos) + "," + str(layer_top) + "!")
                    output_right_pos += output_right_step
                layer_top -= 150
            if input_node_num:
                for node in layers[device_id][index]["in"]:
                    dot.edge(node["name"] + str(index) + "i", kernel_node["index"])
            if output_node_num:
                for node in layers[device_id][index]["out"]:
                    dot.edge(kernel_node["index"], node["name"] + str(index) + "o")
    dot.node("Data Flow Graph showing kernel execution and comparing the data node between CUDA and SYCL", shape="plaintext", pos=str(total_width / 2) + ",150!", style='bold', fontsize='20')
# 4. Render Graph
    dot.render(filename=GRAPH_FILENAME, format='pdf', directory=CODEPIN_GRAPH_FILE_PATH, cleanup=True)

def main():
    global EPSILON_FILE
    match_checkpoint_num = 0
    dismatch_checkpoint_num = 0
    checkpoint_size = 0
    parser = argparse.ArgumentParser(
        description="CodePin report functionality of the compatibility tool.\n",
        add_help=False,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "--instrumented-cuda-log",
        metavar="<file path>",
        required=True,
        help="Specify the execution log file generated by the instrumented CUDA code.",
    )
    parser.add_argument(
        "--instrumented-sycl-log",
        metavar="<file path>",
        required=True,
        help="Specify the execution log file generated by the instrumented SYCL code.",
    )

    parser.add_argument(
        "--floating-point-comparison-epsilon",
        metavar="<file path>",
        required=False,
        help="Specify the relative and absolute tolerance epsilon JSON file for floating point data comparison. The JSON file contains the key-value pairs, where the key is a specific float type, and the value is the corresponding epsilon. For example:\n"
        "{\n"
        '"rel_tol": 1e-3,               # relative tolerance for all float types, it is a ratio value in the range [0, 1].\n\n'
        '"bf16_abs_tol": 7.81e-3,       # absolute tolerance for bfloat16 type.\n\n'
        '"fp16_abs_tol": 9.77e-4,       # absolute tolerance for float16 type.\n\n'
        '"float_abs_tol": 1.19e-7,      # absolute tolerance for float type.\n\n'
        '"double_abs_tol": 2.22e-16,    # absolute tolerance for double type.\n'
        "}\n"
        "When both rel_tol (relative tolerance) and abs_tol (absolute tolerance) are provided, both tolerances are taken into account.\n"
        "The tolerance values are passed to the Python math.isclose() function.\n"
        "If rel_tol is 0, then abs_tol is used as the tolerance. Conversely, if abs_tol is 0, then rel_tol is used. If both tolerances are 0, the floating point data must be exactly the same when compared.",
    )

    parser.add_argument(
        "--generate-data-flow-graph",
        required=False,
        default=False,
        action="store_true",
        help="Generate the data flow graph for the execution and comparison result."
    )

    args = parser.parse_args()
    EPSILON_FILE = args.floating_point_comparison_epsilon
    (
        cuda_prolog_checkpoint_list,
        cuda_epilog_checkpoint_list,
        mem_used_cuda,
        time_cuda,
        device_stream_dic_cuda,
        ordered_pro_id_cuda,
        ordered_epi_id_cuda
    ) = get_checkpoint_list_from_json_file(args.instrumented_cuda_log)
    (
        sycl_prolog_checkpoint_list,
        sycl_epilog_checkpoint_list,
        mem_used_sycl,
        time_sycl,
        device_stream_dic_sycl,
        ordered_pro_id_sycl,
        ordered_epi_id_sycl
    ) = get_checkpoint_list_from_json_file(args.instrumented_sycl_log)

    bottleneck_cuda = get_bottleneck(time_cuda)
    bottleneck_sycl = get_bottleneck(time_sycl)
    max_device_memory_cuda = get_memory_used(mem_used_cuda)
    max_device_memory_sycl = get_memory_used(mem_used_sycl)

    match_checkpoint_num, dismatch_checkpoint_num, checkpoint_size, failed_log = (
        compare_checkpoint_list(
            ordered_pro_id_cuda,
            ordered_epi_id_cuda,
            ordered_pro_id_sycl,
            ordered_epi_id_sycl,
            cuda_prolog_checkpoint_list,
            cuda_epilog_checkpoint_list,
            sycl_prolog_checkpoint_list,
            sycl_epilog_checkpoint_list,
            match_checkpoint_num,
            dismatch_checkpoint_num,
            checkpoint_size,
        )
    )
    # Need ordered checkpoint IDs to generate the data flow graph based on kernel execution order.
    if args.generate_data_flow_graph:
        generate_data_flow_graph(
            device_stream_dic_cuda,
            device_stream_dic_sycl,
            time_cuda,
            time_sycl,
            ordered_pro_id_cuda,
            ordered_epi_id_cuda,
            ordered_pro_id_sycl,
            ordered_epi_id_sycl,
            cuda_prolog_checkpoint_list,
            cuda_epilog_checkpoint_list,
            sycl_prolog_checkpoint_list,
            sycl_epilog_checkpoint_list
        )

    with open(CODEPIN_REPORT_FILE, "w") as f:
        f.write("CodePin Summary\n")
        f.write("Total API count, " + str(checkpoint_size) + "\n")
        f.write("CodePin Random Seed, " +
                str(CODEPIN_RANDOM_SEED_VALUE) + "\n")
        f.write("CodePin Sampling Threshold, " +
                str(CODEPIN_SAMPLING_THRESHOLD_VALUE) + "\n")
        f.write("CodePin Sampling Percent, " +
                str(CODEPIN_SAMPLING_PERCENT_VALUE) + "\n")
        f.write("Consistent API count, " + str(match_checkpoint_num) + "\n")
        f.write(
            "Most Time-consuming Kernel(CUDA), "
            + str(bottleneck_cuda[0])
            + ", time:"
            + str(bottleneck_cuda[1])
            + "\n"
        )
        f.write(
            "Most Time-consuming Kernel(SYCL), "
            + str(bottleneck_sycl[0])
            + ", time:"
            + str(bottleneck_sycl[1])
            + "\n"
        )
        f.write(
            "Peak Device Memory Used(CUDA), " + str(max_device_memory_cuda[1]) + "\n"
        )
        f.write(
            "Peak Device Memory Used(SYCL), " + str(max_device_memory_sycl[1]) + "\n"
        )
    if failed_log:
        with open(CODEPIN_REPORT_FILE, "a") as f:
            f.write(ERROR_CSV_PATTERN)
            f.write(failed_log)
    graph_file = ""
    if args.generate_data_flow_graph and os.path.isfile(os.path.join(CODEPIN_GRAPH_FILE_PATH, GRAPH_FILENAME + ".pdf")):
        graph_file = f" and '" + GRAPH_FILENAME + ".pdf' file"
    if dismatch_checkpoint_num != 0:
        print(
            f"Finished comparison of the two files and found differences. Please check 'CodePin_Report.csv' file" + graph_file + " located in your project directory.\n"
        )
        sys.exit(-1)
    print(
        f"Finished comparison of the two files and data is identical. Please check 'CodePin_Report.csv' file" + graph_file + " located in your project directory.\n"
    )
    sys.exit(0)


if __name__ == "__main__":
    freeze_support()
    main()
