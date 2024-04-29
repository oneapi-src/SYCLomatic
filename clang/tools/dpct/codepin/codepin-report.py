#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from multiprocessing import freeze_support
import argparse
import json
import os
import sys
from collections.abc import Container

UUID = "ID"
CHECKPOINT = "CheckPoint"
DATA = "Data"
TYPE = "Type"
FREE_MEM = "Free Device Memory"
TOTAL_MEM = "Total Device Memory"
TIME_ELAPSED = "Elapse Time(ms)"
ERROR_MATCH_PATTERN = "Unable to find the corresponding serialization function"
CODEPIN_REPORT_FILE = os.path.join(os.getcwd(), "CodePin_Report.csv")
match_checkpoint_num = 0
dismatch_checkpoint_num = 0
checkpoint_size = 0

ERROR_CSV_PATTERN = "CUDA Meta Data ID, SYCL Meta Data ID, Type, Detail\n"


# Raise the warning message when the data is not matched.
def data_value_dismatch_error(value1, value2):
    return comparison_error(
        f' and [ERROR: DATA VALUE MISMATCH] the CUDA value "{value1}" differs from the SYCL value "{value2}".'
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
    detail = f"[WARNING: METADATA MISMATCH] The pair of prolog data {id} are mismatched, and the corresponding pair of epilog data matches. This mismatch may be caused by the initialized memory or argument used in the API {id}.\n"
    return prepare_failed_log(id, id, "Data value", detail)


def get_missing_key_log(id):
    detail = f'[ERROR: METADATA MISSING] Cannot find the checkpoint: "{id}" in the execution log dataset of instrumented SYCL code.\n'
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


def compare_data_value(data1, data2):
    if data1 == ERROR_MATCH_PATTERN or data2 == ERROR_MATCH_PATTERN:
        raise no_serialization_function_error()
    if data1 != data2:
        raise data_value_dismatch_error(data1, data2)
    return True


def compare_list_value(cuda_list, sycl_list):
    for i in range(len(cuda_list)):
        try:
            if is_both_container(cuda_list[i], sycl_list[i]):
                compare_container_value(cuda_list[i], sycl_list[i])
                continue
            else:
                compare_data_value(cuda_list[i], sycl_list[i])
                continue
        except comparison_error as e:
            raise comparison_error(f"->[{i}]{e.message}")


def compare_dict_value(cuda_dict, sycl_dict):
    for name, data in cuda_dict.items():
        if name not in sycl_dict:
            raise data_missed_error(name)
        try:
            if is_both_container(data, sycl_dict[name]):
                compare_container_value(data, sycl_dict[name])
                continue
            else:
                if name == TYPE:  # Check the Data only, ignore the key is 'Type'
                    continue
                compare_data_value(data, sycl_dict[name])
        except comparison_error as e:
            raise comparison_error(f'->"{name}"{e.message}')


def compare_container_value(cuda_value, sycl_value):
    if len(cuda_value) != len(sycl_value):
        raise data_length_dismatch_error()
    if is_container_with_type(cuda_value, sycl_value, list):
        return compare_list_value(cuda_value, sycl_value)
    elif is_container_with_type(cuda_value, sycl_value, dict):
        return compare_dict_value(cuda_value, sycl_value)


def compare_checkpoint(cuda_checkpoint, sycl_checkpoint):
    for id, cuda_var in cuda_checkpoint.items():
        sycl_var = sycl_checkpoint.get(id)
        if cuda_var is not None and sycl_var is not None:
            try:
                compare_container_value(cuda_var, sycl_var)
                continue
            except comparison_error as e:
                raise comparison_error(f'"{id}"{e.message}')


def is_checkpoint_length_dismatch(cuda_list, sycl_list):
    if len(cuda_list) != len(sycl_list):
        print_checkpoint_length_dismatch_warning(cuda_list, sycl_list)


def compare_checkpoint_list(
    cuda_prolog_checkpoint_list,
    cuda_epilog_checkpoint_list,
    sycl_prolog_checkpoint_list,
    sycl_epilog_checkpoint_list,
):
    global match_checkpoint_num
    global dismatch_checkpoint_num
    global checkpoint_size
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
    return failed_log


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
    return prolog_checkpoint_list, epilog_checkpoint_list, used_mem_dic, elapsed_time_dic

def get_bottleneck(cp_list):
    bottleneck_id = ""
    max_time = 0.0
    for id, time in cp_list.items():
        if time > max_time:
            bottleneck_id = id
            max_time = time
    return (bottleneck_id, max_time)

def get_memory_used(cp_list):
    cp_id = ""
    max_mem = 0
    for id, used_mem in cp_list.items():
        if used_mem > max_mem:
            cp_id = id
            max_mem = used_mem
    return (cp_id, max_mem)

def main():
    global match_checkpoint_num
    global checkpoint_size
    parser = argparse.ArgumentParser(description="Codepin report tool.\n")
    parser.add_argument(
        "--instrumented-cuda-log",
        metavar="<file path>",
        required=True,
        help="Specifies the execution log file generated by instrumented CUDA code.",
    )
    parser.add_argument(
        "--instrumented-sycl-log",
        metavar="<file path>",
        required=True,
        help="Specifies the execution log file generated by instrumented SYCL code.",
    )
    args = parser.parse_args()

    cuda_prolog_checkpoint_list, cuda_epilog_checkpoint_list, mem_used_cuda, time_cuda = get_checkpoint_list_from_json_file(
        args.instrumented_cuda_log)
    sycl_prolog_checkpoint_list, sycl_epilog_checkpoint_list, mem_used_sycl, time_sycl = get_checkpoint_list_from_json_file(
        args.instrumented_sycl_log)
    
    bottleneck_cuda = get_bottleneck(time_cuda)
    bottleneck_sycl = get_bottleneck(time_sycl)
    max_device_memory_cuda = get_memory_used(mem_used_cuda)
    max_device_memory_sycl = get_memory_used(mem_used_sycl)

    failed_log = compare_checkpoint_list(
        cuda_prolog_checkpoint_list,
        cuda_epilog_checkpoint_list,
        sycl_prolog_checkpoint_list,
        sycl_epilog_checkpoint_list,
    )
    with(open(CODEPIN_REPORT_FILE, 'w')) as f:
        f.write("CodePin Summary\n")
        f.write("Totally APIs count, " + str(checkpoint_size) + "\n")
        f.write("Consistently APIs count, " + str(match_checkpoint_num) + "\n")
        f.write("Bottleneck Kernel(CUDA), " + str(bottleneck_cuda[0]) + ", time:" + str(bottleneck_cuda[1]) + "\n")
        f.write("Bottleneck Kernel(SYCL), " + str(bottleneck_sycl[0]) + ", time:" + str(bottleneck_sycl[1]) + "\n")
        f.write("Peak Device Memory Used(CUDA), " + str(max_device_memory_cuda[1]) + "\n")
        f.write("Peak Device Memory Used(SYCL), " + str(max_device_memory_sycl[1]) + "\n")
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
