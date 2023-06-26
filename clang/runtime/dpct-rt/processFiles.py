#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#===--------------- processFiles.py ----------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

import os
import sys
import re
import io
import argparse

cur_file_dir = os.path.dirname(os.path.realpath(__file__))
content_files_list = ["atomic", "blas_utils", "device",
                      "dpct", "dpl_utils", "image", "kernel", "math", "memory", "util", "rng_utils", "lib_common_utils",
                      "dnnl_utils", "ccl_utils", "fft_utils", "lapack_utils", "sparse_utils"]
dpl_extras_content_files_list = [
    "algorithm", "functional", "iterators", "memory", "numeric", "vector", "dpcpp_extensions"]

input_files_dir = os.path.join(cur_file_dir, "include")

def exit_script():
    sys.exit()

def get_file_paths(cont_file, inc_files_dir, runtime_files_dir, is_dpl_extras):
    if (is_dpl_extras):
        cont_file_result = os.path.join(
            input_files_dir, "dpl_extras/", cont_file + ".h")
        runtime_files_dir_result = os.path.join(
            runtime_files_dir, "dpl_extras/", cont_file + ".h")
        inc_files_all_dir_result = os.path.join(
            inc_files_dir, "dpl_extras/", cont_file + ".all.inc")
        return [cont_file_result, runtime_files_dir_result, inc_files_all_dir_result]
    else:
        cont_file_result = os.path.join(input_files_dir, cont_file + ".hpp")
        runtime_files_dir_result = os.path.join(
            runtime_files_dir, cont_file + ".hpp")
        inc_files_all_dir_result = os.path.join(
            inc_files_dir, cont_file + ".all.inc")
        return [cont_file_result, runtime_files_dir_result, inc_files_all_dir_result]


def create_dir(inc_files_dir, runtime_files_dir):
    if (not os.path.exists(os.path.join(runtime_files_dir))):
        os.makedirs(os.path.join(runtime_files_dir))
    if (not os.path.exists(os.path.join(runtime_files_dir, "dpl_extras/"))):
        os.makedirs(os.path.join(runtime_files_dir, "dpl_extras/"))
    if (not os.path.exists(os.path.join(inc_files_dir))):
        os.makedirs(os.path.join(inc_files_dir))
    if (not os.path.exists(os.path.join(inc_files_dir, "dpl_extras/"))):
        os.makedirs(os.path.join(inc_files_dir, "dpl_extras/"))

def convert_to_cxx_code(line_code):
    return bytes("R\"Delimiter(", 'utf-8') + line_code + bytes(")Delimiter\"\n", 'utf-8')

def process_a_file(cont_file, inc_files_dir, runtime_files_dir, is_dpl_extras):
    file_names = get_file_paths(
        cont_file, inc_files_dir, runtime_files_dir, is_dpl_extras)
    cont_file_handle = io.open(file_names[0], "rb")
    runtime_file_handle = io.open(file_names[1], "w+b")
    inc_all_file_handle = io.open(file_names[2], "w+b")

    inc_all_file_lines = []
    runtime_file_lines = []

    for line in cont_file_handle:
        runtime_file_lines.append(line)
        inc_all_file_lines.append(convert_to_cxx_code(line))

    runtime_file_str = bytes("", 'utf-8')
    for line in runtime_file_lines:
        runtime_file_str = runtime_file_str + line

    # cl.exe will emit error if a literal string length >= 65535
    # so we need convert the generated code from style
    # {code}
    # std::string code = "a very very long string";
    # {code}
    # to style
    # {code}
    # std::string code = std::string("a long string") + ... + std::string("a long string");
    # {code}
    # The simplest way is treat one line a std::string instance and then use
    # "+" to connect them, but if the number of std::string instances is too large (like 7000),
    # clang will crash.
    # So we should generate code like
    # {code}
    # std::string code = std::string("as long as possible but < 65535") + ... + std::string("as long as possible but < 65535");
    # {code}
    string_literal_length_limit = 65535
    string_literal_length_counter = 0
    inc_all_file_str = bytes("std::string(", 'utf-8')
    for line in inc_all_file_lines:
        if (string_literal_length_counter + len(line) < string_literal_length_limit):
            inc_all_file_str = inc_all_file_str + line
            string_literal_length_counter = string_literal_length_counter + len(line)
        else:
            inc_all_file_str = inc_all_file_str + bytes(")+std::string(", 'utf-8')
            inc_all_file_str = inc_all_file_str + line
            string_literal_length_counter = len(line)
    inc_all_file_str = inc_all_file_str + bytes(")", 'utf-8')

    runtime_file_handle.write(runtime_file_str)
    inc_all_file_handle.write(inc_all_file_str)

    cont_file_handle.close()
    runtime_file_handle.close()
    inc_all_file_handle.close()

def main():
    if (sys.version_info.major < 3):
        print("Please use python3.")
        exit_script()

    parser = argparse.ArgumentParser(description="Processing files.")
    parser.add_argument('--build-dir', action='store', default='',
                        help='The build directory.')
    parser.add_argument('--inc-output-dir', action='store', default='',
                        help='The path of the output inc files. These inc files will be included by src code. Ignored when --build-dir specified.')
    parser.add_argument('--helper-header-output-dir', action='store', default='',
                        help='The path of the output runtime files. These files are the final full set helper header files. Ignored when --build-dir specified.')
    parser.add_argument('--input-inc-dir', action='store', default='',
                        help='The path of the input *.inc files. Default value is dir where this script at.')
    args = parser.parse_args()

    inc_files_dir = str()
    runtime_files_dir = str()

    if (args.build_dir):
        inc_files_dir = os.path.join(
            args.build_dir, "tools/clang/include/clang/DPCT")
        runtime_files_dir = os.path.join(
            args.build_dir, "tools/clang/runtime/dpct-rt/include")
    else:
        if (args.inc_output_dir and args.helper_header_output_dir):
            inc_files_dir = args.inc_output_dir
            runtime_files_dir = args.helper_header_output_dir
        else:
            print("Error: inc-output-dir or helper-header-output-dir is empty.")
            exit_script()

    if (args.input_inc_dir):
        input_files_dir = args.input_inc_dir

    create_dir(inc_files_dir, runtime_files_dir)

    for cont_file in content_files_list:
        process_a_file(cont_file, inc_files_dir,
                       runtime_files_dir, False)
    for cont_file in dpl_extras_content_files_list:
        process_a_file(cont_file, inc_files_dir, runtime_files_dir,
                       True)

    print("[Note] DPCT *.inc files are generated successfully!")


if __name__ == "__main__":
    main()
