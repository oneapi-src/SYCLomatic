#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exceptio

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
ERROR_MATCH_PATTERN = "Unable to find the corresponding serialization function"
CODEPIN_REPORT_FILE = os.path.join(os.getcwd(), 'CodePin_Report.csv')
passed_checkpoint_num = 0
checkpoint_size = 0

ERROR_CSV_PATTERN = "CUDA Meta Data ID, SYCL Meta Data ID, Type, Detail\n"
#Raise the warning message when the data is not matched.
def data_value_dismatch_error(value1, value2):
    return comparison_error(
            f" and [ERROR: DATA VALUE MISMATCH] the CUDA value \"{value1}\" differs from the SYCL value \"{value2}\".")
def no_serialization_function_error():
    return comparison_error(
            f" and [ERROR: NO SERIALIZATION FUNCTION]the CUDA or SYCL value cannot be dummped, lack of dump function. Please report it to the tool developer.")
def data_missed_error(name):
    return comparison_error(
            f" and [ERROR: DATA MISSED] Cannot find the {name} in SYCL Json.\n")
def data_length_dismatch_error():
    return comparison_error(
            f" and [ERROR: DATA LENGTH MISMATCH] The size of CUDA and SYCL data are mismatch.")

def print_checkpoint_length_dismatch_warning(cuda_list, sycl_list):
     print(
            f"[ERROR: CHECKPOINT LENGTH MISMATCH] \n CUDA CodePin list length is: {len(cuda_list)}. \n SYCL CodePin list length is: {len(sycl_list)}.\n")

def prepare_failed_log(cuda_id, sycl_id, log_type, detail):
    return ",".join([cuda_id, sycl_id, log_type, detail]) + "\n"

def get_missing_key_log(id):
    detail = f"[ERROR: METADATA MISSING] Cannot find the checkpoint: \"{id}\" in the execution log dataset of instrumented SYCL code.\n"
    return prepare_failed_log(id, "Missing", "Execution path", detail)

def get_data_value_mismatch_log(id, message):
    return prepare_failed_log(id, id, "Data value", "The location of failed ID " + message)

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
                if name == TYPE:                  # Check the Data only, ignore the key is 'Type'
                    continue
                compare_data_value(data, sycl_dict[name])
        except comparison_error as e:
            raise comparison_error(f"->\"{name}\"{e.message}")


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
                raise comparison_error(f"\"{id}\"{e.message}")

def compare_checkpoint_list(cuda_list, sycl_list):
    global passed_checkpoint_num
    global checkpoint_size
    failed_log  = ""
    if len(cuda_list) != len(sycl_list):
        print_checkpoint_length_dismatch_warning(cuda_list, sycl_list)

    for id, cuda_checkpoint in cuda_list.items():
        checkpoint_size += 1
        if not sycl_list.get(id):
            failed_log += get_missing_key_log(id)
            continue
        try:
            compare_checkpoint(cuda_checkpoint, sycl_list.get(id))
            passed_checkpoint_num += 1
        except comparison_error as e:
            failed_log += get_data_value_mismatch_log(id, e.message)
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
    json_data_list = read_data_from_json_file(file_path)
    for item in json_data_list:
        id = item[UUID]
        checkpoint_list[id] = item.get(CHECKPOINT, {})
    return checkpoint_list

def main():
    global passed_checkpoint_num
    global checkpoint_size
    parser = argparse.ArgumentParser(
        description='Codepin report tool.\n')
    parser.add_argument('--instrumented-cuda-log', metavar='<file path>',
                        required=True, help='Specifies the execution log file generated by instrumented CUDA code.')
    parser.add_argument('--instrumented-sycl-log', metavar='<file path>', required=True,
                        help='Specifies the execution log file generated by instrumented SYCL code.')
    args = parser.parse_args()

    cuda_checkpoint = get_checkpoint_list_from_json_file(args.instrumented_cuda_log)
    sycl_checkpoint = get_checkpoint_list_from_json_file(args.instrumented_sycl_log)
    
    failed_log = compare_checkpoint_list(cuda_checkpoint, sycl_checkpoint)
    with(open(CODEPIN_REPORT_FILE, 'w')) as f:
        f.write("CodePin Summary\n")
        f.write("Totally APIs count, " + str(checkpoint_size) + "\n")
        f.write("Consistently APIs count, " + str(passed_checkpoint_num) + "\n")
    if (failed_log):
        with(open(CODEPIN_REPORT_FILE, 'a')) as f:
            f.write(ERROR_CSV_PATTERN)
            f.write(failed_log)
        print(f"Comparison of the two files succeeded succeeded and found differences. Please check the status in the file 'CodePin Report.csv' located in your project directory.")
        sys.exit(-1)
    print(f"Comparison of the two files succeeded, no differences were found.")
    sys.exit(0)

if __name__ == '__main__':
    freeze_support()
    main()
