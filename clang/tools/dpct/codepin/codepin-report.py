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
ERROR_MATCH_PATTERN = "Unable to find the corresponding serialization function"
CODEPIN_REPORT_FILE = os.path.join(os.getcwd(), "CodePin_Report.csv")
ERROR_CSV_PATTERN = "CUDA Meta Data ID, SYCL Meta Data ID, Type, Detail\n"
EPSILON_FILE = ""

# Reference: https://en.wikipedia.org/wiki/Machine_epsilon
# bfloat16 reference: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
default_epsilons = {
    "bf16_abs_tol": 7.81e-03,  # 2^-7
    "fp16_abs_tol": 9.77e-04,  # 2^-10
    "float_abs_tol": 1.19e-07,  # 2^-23
    "double_abs_tol": 2.22e-16,  # 2^-52
    "rel_tol": 1e-3,
}


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


def compare_checkpoint(cuda_checkpoint, sycl_checkpoint):
    error_messages = []
    for var_name, cuda_var in cuda_checkpoint.items():
        sycl_var = sycl_checkpoint.get(var_name)
        if cuda_var is not None and sycl_var is not None:
            try:
                compare_cp_var(cuda_var, sycl_var, var_name)
            except Exception as e:
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

    for id, cuda_epilog_checkpoint in cuda_epilog_checkpoint_list.items():
        checkpoint_size += 1
        if not sycl_epilog_checkpoint_list.get(id):
            failed_log += get_missing_key_log(id)
            dismatch_checkpoint_num += 1
            continue
        try:
            compare_checkpoint(
                cuda_epilog_checkpoint, sycl_epilog_checkpoint_list.get(id)
            )
            match_checkpoint_num += 1
        except comparison_error as e:
            failed_log += get_data_value_mismatch_log(id, e.message)
            failed_epilog.append(id)
            dismatch_checkpoint_num += 1
            continue
    for id, cuda_prolog_checkpoint in cuda_prolog_checkpoint_list.items():
        checkpoint_size += 1
        if not sycl_prolog_checkpoint_list.get(id):
            failed_log += get_missing_key_log(id)
            dismatch_checkpoint_num += 1
            continue
        try:
            compare_checkpoint(
                cuda_prolog_checkpoint, sycl_prolog_checkpoint_list.get(id)
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
        return parse_json(f.read())


def get_checkpoint_list_from_json_file(file_path):
    checkpoint_list = {}
    used_mem_dic = {}
    elapsed_time_dic = {}
    json_data_list = read_data_from_json_file(file_path)
    prolog_checkpoint_list = {}
    epilog_checkpoint_list = {}
    for item in json_data_list:
        id = item[UUID]
        if "prolog" in id:
            prolog_checkpoint_list[id] = item.get(CHECKPOINT, {})
        elif "epilog" in id:
            epilog_checkpoint_list[id] = item.get(CHECKPOINT, {})

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

    args = parser.parse_args()
    EPSILON_FILE = args.floating_point_comparison_epsilon
    (
        cuda_prolog_checkpoint_list,
        cuda_epilog_checkpoint_list,
        mem_used_cuda,
        time_cuda,
    ) = get_checkpoint_list_from_json_file(args.instrumented_cuda_log)
    (
        sycl_prolog_checkpoint_list,
        sycl_epilog_checkpoint_list,
        mem_used_sycl,
        time_sycl,
    ) = get_checkpoint_list_from_json_file(args.instrumented_sycl_log)

    bottleneck_cuda = get_bottleneck(time_cuda)
    bottleneck_sycl = get_bottleneck(time_sycl)
    max_device_memory_cuda = get_memory_used(mem_used_cuda)
    max_device_memory_sycl = get_memory_used(mem_used_sycl)

    match_checkpoint_num, dismatch_checkpoint_num, checkpoint_size, failed_log = (
        compare_checkpoint_list(
            cuda_prolog_checkpoint_list,
            cuda_epilog_checkpoint_list,
            sycl_prolog_checkpoint_list,
            sycl_epilog_checkpoint_list,
            match_checkpoint_num,
            dismatch_checkpoint_num,
            checkpoint_size,
        )
    )

    with open(CODEPIN_REPORT_FILE, "w") as f:
        f.write("CodePin Summary\n")
        f.write("Totally APIs count, " + str(checkpoint_size) + "\n")
        f.write("Consistently APIs count, " + str(match_checkpoint_num) + "\n")
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
    if dismatch_checkpoint_num != 0:
        print(
            f"Finished comparison of the two files and found differences. Please check 'CodePin_Report.csv' file located in your project directory.\n"
        )
        sys.exit(-1)
    print(
        f"Finished comparison of the two files and data is identical. Please check 'CodePin_Report.csv' file located in your project directory.\n"
    )
    sys.exit(0)


if __name__ == "__main__":
    freeze_support()
    main()
