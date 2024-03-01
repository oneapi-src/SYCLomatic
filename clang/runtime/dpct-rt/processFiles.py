#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#===--------------- processFiles.py ----------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

# This script converts header files to literal `std::string`s.
# Each header file is converted to a `std::string`.
# Simply put, the work is add "R\"Delimiter(" and ")Delimiter\"\n" around each header file.
# But on Windows, there are some length limitation about literal string,
# so this script will split a long string into some shorter ones.
# 
# Usage:
# python3 processFiles.py build-folder
# Note: This file (processFiles.py) can only be moved to other locations.

import os
import sys
import re
import io
import argparse

cur_file_dir = os.path.dirname(os.path.realpath(__file__))

runtime_header_src_files_dir = os.path.join(cur_file_dir, "include", "dpct")

def exit_script():
    sys.exit()

def get_file_paths(runtime_src_file, inc_files_dir):
    split_str = os.path.join("include", "dpct", "")
    inc_files_all_dir_result = os.path.join(inc_files_dir, runtime_src_file.split(split_str)[1]) + ".inc"
    return [runtime_src_file, inc_files_all_dir_result]

def convert_to_cxx_code(line_code):
    return bytes("R\"Delimiter(", 'utf-8') + line_code + bytes(")Delimiter\"\n", 'utf-8')

def process_a_file(runtime_src_file, inc_files_dir):
    file_names = get_file_paths(runtime_src_file, inc_files_dir)
    cont_file_handle = io.open(file_names[0], "rb")

    inc_all_file_lines = []

    for line in cont_file_handle:
        inc_all_file_lines.append(convert_to_cxx_code(line))

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
    os.makedirs(os.path.dirname(file_names[1]), exist_ok=True)
    with open(file_names[1], "w+b") as f:
        f.write(inc_all_file_str)

    cont_file_handle.close()

def get_header_files(path):
    header_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            header_files.append(os.path.join(root, file))
    return header_files

def main():
    if (sys.version_info.major < 3):
        print("Please use python3.")
        exit_script()

    parser = argparse.ArgumentParser(description="Processing files.")
    parser.add_argument('--build-dir', action='store', default='', help='The build directory.')
    args = parser.parse_args()

    inc_files_dir = str()

    if (args.build_dir):
        inc_files_dir = os.path.join(
            args.build_dir, "tools/clang/include/clang/DPCT")
    else:
        print("Error: build-dir is empty.")
        exit_script()

    helper_files_list = get_header_files(runtime_header_src_files_dir)

    for runtime_src_file in helper_files_list:
        process_a_file(runtime_src_file, inc_files_dir)

    print("[Note] DPCT *.inc files are generated successfully!")

if __name__ == "__main__":
    main()
