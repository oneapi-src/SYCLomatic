#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2021 Intel Corporation. All rights reserved.
#
# The information and source code contained herein is the exclusive
# property of Intel Corporation and may not be disclosed, examined
# or reproduced in whole or in part without explicit written authorization
# from the company.

import os
import sys
import re
import io

cur_file_dir = os.path.dirname(os.path.realpath(__file__))
content_files_list = ["atomic", "blas_utils", "device",
                      "dpct", "dpl_utils", "image", "kernel", "memory", "util"]
dpl_extras_content_files_list = [
    "algorithm", "functional", "iterators", "memory", "numeric", "vector"]

content_files_name_list = ["Atomic", "BlasUtils", "Device",
                      "Dpct", "DplUtils", "Image", "Kernel", "Memory", "Util"]
dpl_extras_content_files_name_list = [
    "Algorithm", "Functional", "Iterators", "Memory", "Numeric", "Vector"]

is_os_win = False

def exit_script():
    sys.exit()

def get_file_pathes(cont_file, inc_files_dir, runtime_files_dir, is_dpl_extras):
    if (is_dpl_extras):
        return [os.path.join(cur_file_dir, "dpl_extras/", cont_file + ".h.inc"), os.path.join(runtime_files_dir, "dpl_extras/", cont_file + ".h"), os.path.join(inc_files_dir, "dpl_extras/", cont_file + ".inc"), os.path.join(inc_files_dir, "dpl_extras/", cont_file + ".all.inc")]
    else:
        return [os.path.join(cur_file_dir, cont_file + ".hpp.inc"), os.path.join(runtime_files_dir, cont_file + ".hpp"), os.path.join(inc_files_dir, cont_file + ".inc"), os.path.join(inc_files_dir, cont_file + ".all.inc")]

def create_dir(inc_files_dir, runtime_files_dir):
    if (not os.path.exists(os.path.join(runtime_files_dir))):
        os.makedirs(os.path.join(runtime_files_dir))
    if (not os.path.exists(os.path.join(runtime_files_dir, "dpl_extras/"))):
        os.makedirs(os.path.join(runtime_files_dir, "dpl_extras/"))
    if (not os.path.exists(os.path.join(inc_files_dir))):
        os.makedirs(os.path.join(inc_files_dir))
    if (not os.path.exists(os.path.join(inc_files_dir, "dpl_extras/"))):
        os.makedirs(os.path.join(inc_files_dir, "dpl_extras/"))

def convert_line_end(line):
    if (is_os_win):
        line = line.replace(UNIX_LINE_ENDING, WINDOWS_LINE_ENDING)
    return line

def process_a_file(cont_file, inc_files_dir, runtime_files_dir, is_dpl_extras, file_dict):
    file_names = get_file_pathes(cont_file, inc_files_dir, runtime_files_dir, is_dpl_extras)
    cont_file_handle = io.open(file_names[0], "rb")
    runtime_file_handle = io.open(file_names[1], "w+b")
    inc_file_handle = io.open(file_names[2], "w+b")
    inc_all_file_handle = io.open(file_names[3], "w+b")

    inc_file_lines = []
    inc_all_file_lines = []
    runtime_file_lines = []

    helper_file_enum_name = file_dict[cont_file]
    if (is_dpl_extras):
        helper_file_enum_name = "DplExtras" + file_dict[cont_file]

    Idx = 0
    # All dependency must into single line.
    # generated code is like: DPCT_DEPENDENCY({clang::dpct::HelperFileEnum::Memory, "dpct_memcpy_detail"},{clang::dpct::HelperFileEnum::Util, "DataType"},)
    dependency_line = bytes("", 'utf-8')
    is_dependency = False
    is_code = False
    for line in cont_file_handle:
        if (line.startswith(bytes("// DPCT_LABEL_BEGIN", 'utf-8'))):
            line = line.replace(bytes('//', 'utf-8'), bytes('', 'utf-8'))
            line = line.strip()
            splited = line.split(bytes('|', 'utf-8'))
            content_begin_line = bytes("DPCT_CONTENT_BEGIN(" + helper_file_enum_name + ", \"", 'utf-8') + \
                                 splited[1] + bytes("\", \"", 'utf-8') + splited[2] + bytes("\", " + \
                                 str(Idx) + ")\n", 'utf-8')
            inc_file_lines.append(content_begin_line)
            Idx = Idx + 1
        elif (line.startswith(bytes("// DPCT_LABEL_END", 'utf-8'))):
            inc_file_lines.append(bytes("DPCT_CONTENT_END\n", 'utf-8'))
            is_code = False
        elif (line.startswith(bytes("// DPCT_DEPENDENCY_BEGIN", 'utf-8'))):
            dependency_line = bytes("DPCT_DEPENDENCY(", 'utf-8')
            is_dependency = True
        elif (line.startswith(bytes("// DPCT_DEPENDENCY_END", 'utf-8'))):
            dependency_line = dependency_line + bytes(")\n", 'utf-8')
            inc_file_lines.append(dependency_line)
            is_dependency = False
            dependency_line = bytes("", 'utf-8')
        elif (line.startswith(bytes("// DPCT_DEPENDENCY_EMPTY", 'utf-8'))):
            inc_file_lines.append(bytes("DPCT_DEPENDENCY()\n", 'utf-8'))
        elif (line.startswith(bytes("// DPCT_CODE", 'utf-8'))):
            is_code = True
        elif (line.startswith(bytes("// DPCT_COMMENT", 'utf-8'))):
            continue
        else:
            if (is_dependency):
                line = line.replace(bytes('//', 'utf-8'), bytes('', 'utf-8'))
                line = line.strip()
                splited = line.split(bytes('|', 'utf-8'))
                dependency_line = dependency_line + bytes("{clang::dpct::HelperFileEnum::", 'utf-8') + splited[0] + bytes(", \"", 'utf-8') + \
                             splited[1] + bytes("\"},", 'utf-8')
            else:
                if (is_code):
                    inc_file_lines.append(bytes("R\"Delimiter(", 'utf-8') + convert_line_end(line) + bytes(")Delimiter\"\n", 'utf-8'))
                runtime_file_lines.append(convert_line_end(line))
                inc_all_file_lines.append(bytes("R\"Delimiter(", 'utf-8') + convert_line_end(line) + bytes(")Delimiter\"\n", 'utf-8'))

    inc_file_str = bytes("", 'utf-8')
    inc_all_file_str = bytes("", 'utf-8')
    runtime_file_str = bytes("", 'utf-8')
    for line in inc_file_lines:
        inc_file_str = inc_file_str + line
    for line in runtime_file_lines:
        runtime_file_str = runtime_file_str + line
    for line in inc_all_file_lines:
        inc_all_file_str = inc_all_file_str + line

    runtime_file_handle.write(runtime_file_str)
    inc_file_handle.write(inc_file_str)
    inc_all_file_handle.write(inc_all_file_str)

    cont_file_handle.close()
    runtime_file_handle.close()
    inc_file_handle.close()
    inc_all_file_handle.close()

def check_files():
    file_handle = io.open(os.path.join(cur_file_dir, "HelperFileNames.inc"), "rb")
    files_in_inc = []
    for line in file_handle:
        if (line.startswith(bytes("HELPERFILE(", 'utf-8'))):
            line = line.replace(bytes('HELPERFILE(', 'utf-8'), bytes('', 'utf-8'))
            line = line.replace(bytes(')', 'utf-8'), bytes('', 'utf-8'))
            splited = line.split(bytes(',', 'utf-8'))
            abs_file_path = os.path.join(bytes(cur_file_dir, 'utf-8'), splited[0])
            files_in_inc.append(os.path.abspath(abs_file_path.decode('utf-8')))
    file_handle.close()

    files_in_current_dir = []
    for searching_dir, dir_list, file_list in os.walk(cur_file_dir):
        for file_name in file_list:
            if (file_name.endswith(".h.inc") or file_name.endswith(".hpp.inc")):
                abs_file_path = os.path.join(searching_dir, file_name)
                files_in_current_dir.append(abs_file_path)

    set_of_files_in_inc = set(files_in_inc)
    set_of_files_in_current_dir = set(files_in_current_dir)
    if (set_of_files_in_inc == set_of_files_in_current_dir):
        return True

    print("Files defined in HelperFileNames.inc and files in current folder are not same. Please update the HelperFileNames.inc file.\n")
    print("Files defined in HelperFileNames.inc:")
    print(set_of_files_in_inc)
    print("Files in current folder:")
    print(set_of_files_in_current_dir)
    return False

def main(build_dir):
    if (not check_files()):
        exit_script()

    content_files_dict = dict(zip(content_files_list, content_files_name_list))
    dpl_extras_content_files_dict = dict(zip(dpl_extras_content_files_list, dpl_extras_content_files_name_list))

    inc_files_dir = os.path.join(build_dir, "tools/clang/include/clang/DPCT")
    runtime_files_dir = os.path.join(build_dir, "tools/clang/runtime/dpct-rt/include")

    create_dir(inc_files_dir, runtime_files_dir)

    if (sys.platform == "win32"):
        is_os_win = True

    for cont_file in content_files_list:
        process_a_file(cont_file, inc_files_dir, runtime_files_dir, False, content_files_dict)
    for cont_file in dpl_extras_content_files_list:
        process_a_file(cont_file, inc_files_dir, runtime_files_dir, True, dpl_extras_content_files_dict)

if __name__ == "__main__":
    main(sys.argv[1])