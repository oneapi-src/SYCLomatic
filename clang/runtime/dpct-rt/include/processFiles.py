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
from enum import Enum

cur_file_dir = os.path.dirname(os.path.realpath(__file__))
content_files_list = ["atomic", "blas_utils", "device",
                      "dpct", "dpl_utils", "image", "kernel", "memory", "util", "rng_utils", "lib_common_utils", 
                      "dnnl_utils", "ccl_utils", "fft_utils", "lapack_utils"]
dpl_extras_content_files_list = [
    "algorithm", "functional", "iterators", "memory", "numeric", "vector", "dpcpp_extensions"]

content_files_name_list = ["Atomic", "BlasUtils", "Device",
                           "Dpct", "DplUtils", "Image", "Kernel", "Memory", "Util", "RngUtils", "LibCommonUtils", 
                           "DnnlUtils", "CclUtils", "FftUtils", "LapackUtils"]
dpl_extras_content_files_name_list = [
    "Algorithm", "Functional", "Iterators", "Memory", "Numeric", "Vector", "DpcppExtensions"]

is_os_win = False

features_enum_list = []
features_enum_pair_list = []

features_enum_referenced_list = []

stack = []
input_files_dir = cur_file_dir

class Usm_status_enum(Enum):
    USM = 1
    NON_USM = 2
    COMMON = 3


def is_if_DPCT_USM_LEVEL_NONE(line):
    line = line.strip()
    if (not (line.startswith(bytes("#ifdef", 'utf-8')) or \
             line.startswith(bytes("#if", 'utf-8')))):
        return False

    if (line.startswith(bytes("#ifdef", 'utf-8'))):
        line = line.replace(bytes("#ifdef", 'utf-8'), bytes("", 'utf-8'))
    elif (line.startswith(bytes("#if", 'utf-8'))):
        line = line.replace(bytes("#if", 'utf-8'), bytes("", 'utf-8'))

    line = line.strip()
    if (line == bytes("DPCT_USM_LEVEL_NONE", 'utf-8')):
        return True
    return False


def is_ifn_DPCT_USM_LEVEL_NONE(line):
    line = line.strip()
    if (not line.startswith(bytes("#ifndef", 'utf-8'))):
        return False

    line = line.replace(bytes("#ifndef", 'utf-8'), bytes("", 'utf-8'))

    line = line.strip()
    if (line == bytes("DPCT_USM_LEVEL_NONE", 'utf-8')):
        return True
    return False


def is_pp_if(line):
    line = line.strip()
    if (line.startswith(bytes("#ifndef", 'utf-8')) or \
        line.startswith(bytes("#ifdef", 'utf-8')) or \
        line.startswith(bytes("#if", 'utf-8'))):
        return True
    return False


def is_pp_else(line):
    line = line.strip()
    if (line.startswith(bytes("#else", 'utf-8'))):
        return True
    return False


def is_pp_endif(line):
    line = line.strip()
    if (line.startswith(bytes("#endif", 'utf-8'))):
        return True
    return False


def get_current_usm_status():
    for i in reversed(stack):
        if (i == Usm_status_enum.NON_USM):
            return Usm_status_enum.NON_USM
        elif (i == Usm_status_enum.USM):
            return Usm_status_enum.USM
    return Usm_status_enum.COMMON


def update_usm_status(line):
    need_skip = True
    if (is_pp_if(line)):
        if (is_if_DPCT_USM_LEVEL_NONE(line)):
            stack.append(Usm_status_enum.NON_USM)
        elif (is_ifn_DPCT_USM_LEVEL_NONE(line)):
            stack.append(Usm_status_enum.USM)
        else:
            stack.append(Usm_status_enum.COMMON)
            need_skip = False
    elif (is_pp_else(line)):
        if (stack[-1] == Usm_status_enum.NON_USM):
            stack.pop()
            stack.append(Usm_status_enum.USM)
        elif (stack[-1] == Usm_status_enum.USM):
            stack.pop()
            stack.append(Usm_status_enum.NON_USM)
        else:
            need_skip = False
    elif (is_pp_endif(line)):
        if (stack[-1] == Usm_status_enum.COMMON):
            need_skip = False
        stack.pop()
    else:
        need_skip = False
    return (get_current_usm_status(), need_skip)


def exit_script():
    sys.exit()


def get_file_pathes(cont_file, inc_files_dir, runtime_files_dir, is_dpl_extras):
    if (is_dpl_extras):
        cont_file_result = os.path.join(
            input_files_dir, "dpl_extras/", cont_file + ".h.inc")
        runtime_files_dir_result = os.path.join(
            runtime_files_dir, "dpl_extras/", cont_file + ".h")
        inc_files_dir_result = os.path.join(
            inc_files_dir, "dpl_extras/", cont_file + ".inc")
        inc_files_all_dir_result = os.path.join(
            inc_files_dir, "dpl_extras/", cont_file + ".all.inc")
        return [cont_file_result, runtime_files_dir_result, inc_files_dir_result, inc_files_all_dir_result]
    else:
        cont_file_result = os.path.join(input_files_dir, cont_file + ".hpp.inc")
        runtime_files_dir_result = os.path.join(
            runtime_files_dir, cont_file + ".hpp")
        inc_files_dir_result = os.path.join(inc_files_dir, cont_file + ".inc")
        inc_files_all_dir_result = os.path.join(
            inc_files_dir, cont_file + ".all.inc")
        return [cont_file_result, runtime_files_dir_result, inc_files_dir_result, inc_files_all_dir_result]


def create_dir(inc_files_dir, runtime_files_dir):
    if (not os.path.exists(os.path.join(runtime_files_dir))):
        os.makedirs(os.path.join(runtime_files_dir))
    if (not os.path.exists(os.path.join(runtime_files_dir, "dpl_extras/"))):
        os.makedirs(os.path.join(runtime_files_dir, "dpl_extras/"))
    if (not os.path.exists(os.path.join(inc_files_dir))):
        os.makedirs(os.path.join(inc_files_dir))
    if (not os.path.exists(os.path.join(inc_files_dir, "dpl_extras/"))):
        os.makedirs(os.path.join(inc_files_dir, "dpl_extras/"))


def append_lines(dist_list, src_list):
    for item in src_list:
        dist_list.append(item)

def convert_to_cxx_code(line_code):
    return bytes("R\"Delimiter(", 'utf-8') + line_code + bytes(")Delimiter\"\n", 'utf-8')

def process_a_file(cont_file, inc_files_dir, runtime_files_dir, is_dpl_extras, file_dict):
    file_names = get_file_pathes(
        cont_file, inc_files_dir, runtime_files_dir, is_dpl_extras)
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
    # generated code is like:
    # DPCT_DEPENDENCY({clang::dpct::HelperFileEnum::Memory, "dpct_memcpy_detail"},{clang::dpct::HelperFileEnum::Util, "DataType"},)
    dependency_line = bytes("", 'utf-8')
    is_dependency = False
    is_code = False
    have_usm_code = False
    usm_line = []
    non_usm_line = []
    has_code = False
    usm_status = Usm_status_enum.COMMON
    parent_feature = bytes("", 'utf-8')
    for line in cont_file_handle:
        if (line.startswith(bytes("// DPCT_LABEL_BEGIN", 'utf-8'))):
            line = line.replace(bytes('//', 'utf-8'), bytes('', 'utf-8'))
            line = line.strip()
            splited = line.split(bytes('|', 'utf-8'))
            content_begin_line = bytes("DPCT_CONTENT_BEGIN(" + helper_file_enum_name + ", \"", 'utf-8') + \
                splited[1] + bytes("\", \"", 'utf-8') + splited[2] + bytes("\", " + str(Idx) + ")\n", 'utf-8')
            inc_file_lines.append(content_begin_line)
            feature_enum_name = bytes(helper_file_enum_name + "_", 'utf-8') + splited[1]
            features_enum_list.append(feature_enum_name)
            feature_pair_name = bytes("{clang::dpct::HelperFileEnum::" + helper_file_enum_name + ", \"", 'utf-8') +\
                                splited[1] + bytes("\"}", 'utf-8')
            features_enum_pair_list.append(bytes("{clang::dpct::HelperFeatureEnum::", 'utf-8') + feature_enum_name +\
                                           bytes(", ", 'utf-8') + feature_pair_name + bytes("}", 'utf-8'))
            Idx = Idx + 1
        elif (line.startswith(bytes("// DPCT_LABEL_END", 'utf-8'))):
            if (not has_code):
                inc_file_lines.append(bytes("\"\"\n", 'utf-8'))
            if (not have_usm_code):
                inc_file_lines.append(bytes(", \"\", \"\"\n", 'utf-8'))
            else:
                inc_file_lines.append(bytes(",\n", 'utf-8'))
                inc_file_lines.append(
                    bytes("// =====below is usm code=================\n", 'utf-8'))
                if (not usm_line):
                    usm_line.append(bytes("\"\"\n", 'utf-8'))
                append_lines(inc_file_lines, usm_line)
                inc_file_lines.append(bytes(",\n", 'utf-8'))
                inc_file_lines.append(
                    bytes("// =====below is none-usm code============\n", 'utf-8'))
                if (not non_usm_line):
                    non_usm_line.append(bytes("\"\"\n", 'utf-8'))
                append_lines(inc_file_lines, non_usm_line)
            
            if (parent_feature == bytes("", 'utf-8')):
                inc_file_lines.append(bytes("DPCT_PARENT_FEATURE(Unknown, \"\")\n", 'utf-8'))
            else:
                inc_file_lines.append(bytes("DPCT_PARENT_FEATURE(" + helper_file_enum_name + ", \"", 'utf-8') + \
                                      parent_feature + bytes("\")\n", 'utf-8'))
            inc_file_lines.append(bytes("DPCT_CONTENT_END\n", 'utf-8'))
            is_code = False
            have_usm_code = False
            has_code = False
            parent_feature = bytes("", 'utf-8')
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
            have_usm_code = False
            usm_line.clear()
            non_usm_line.clear()
        elif (line.startswith(bytes("// DPCT_COMMENT", 'utf-8'))):
            continue
        elif (line.startswith(bytes("// DPCT_PARENT_FEATURE", 'utf-8'))):
            line = line.replace(bytes('//', 'utf-8'), bytes('', 'utf-8'))
            line = line.strip()
            splited = line.split(bytes('|', 'utf-8'))
            parent_feature = splited[1]
        else:
            if (is_dependency):
                line = line.replace(bytes('//', 'utf-8'), bytes('', 'utf-8'))
                line = line.strip()
                splited = line.split(bytes('|', 'utf-8'))
                if ((len(splited) <= 2) or (splited[2] == bytes("", 'utf-8'))):
                    dependency_line = dependency_line + bytes("{{clang::dpct::HelperFileEnum::", 'utf-8') + \
                                      splited[0] + bytes(", \"", 'utf-8') + splited[1] + \
                                      bytes("\"}, clang::dpct::HelperFeatureDependencyKind::HFDK_Both},", 'utf-8')
                else:
                    dependency_line = dependency_line + bytes("{{clang::dpct::HelperFileEnum::", 'utf-8') + \
                                      splited[0] + bytes(", \"", 'utf-8') + splited[1] + \
                                      bytes("\"}, clang::dpct::HelperFeatureDependencyKind::HFDK_", 'utf-8') +\
                                      splited[2] + bytes("},", 'utf-8')
                features_enum_referenced_list.append(splited[0] + bytes("_", 'utf-8')  + splited[1])
            else:
                runtime_file_lines.append(line)
                inc_all_file_lines.append(convert_to_cxx_code(line))
                if (is_code):
                    has_code = True
                    inc_file_lines.append(convert_to_cxx_code(line))
                    usm_status, need_skip = update_usm_status(line)
                    if (need_skip):
                        continue
                    if (usm_status == Usm_status_enum.COMMON):
                        usm_line.append(convert_to_cxx_code(line))
                        non_usm_line.append(convert_to_cxx_code(line))
                    elif (usm_status == Usm_status_enum.NON_USM):
                        have_usm_code = True
                        non_usm_line.append(convert_to_cxx_code(line))
                    elif (usm_status == Usm_status_enum.USM):
                        have_usm_code = True
                        usm_line.append(convert_to_cxx_code(line))

    inc_file_str = bytes("", 'utf-8')
    runtime_file_str = bytes("", 'utf-8')
    for line in inc_file_lines:
        inc_file_str = inc_file_str + line
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
    inc_file_handle.write(inc_file_str)
    inc_all_file_handle.write(inc_all_file_str)

    cont_file_handle.close()
    runtime_file_handle.close()
    inc_file_handle.close()
    inc_all_file_handle.close()


def check_files():
    file_handle = io.open(os.path.join(
        input_files_dir, "HelperFileAndFeatureNames.inc"), "rb")
    files_in_inc = []
    for line in file_handle:
        if (line.startswith(bytes("HELPERFILE(", 'utf-8'))):
            line = line.replace(
                bytes('HELPERFILE(', 'utf-8'), bytes('', 'utf-8'))
            line = line.replace(bytes(')', 'utf-8'), bytes('', 'utf-8'))
            splited = line.split(bytes(',', 'utf-8'))
            abs_file_path = os.path.join(
                bytes(input_files_dir, 'utf-8'), splited[0])
            files_in_inc.append(os.path.abspath(abs_file_path.decode('utf-8')))
    file_handle.close()

    files_in_current_dir = []
    for searching_dir, dir_list, file_list in os.walk(input_files_dir):
        for file_name in file_list:
            if (file_name.endswith(".h.inc") or file_name.endswith(".hpp.inc")):
                abs_file_path = os.path.join(searching_dir, file_name)
                files_in_current_dir.append(abs_file_path)

    set_of_files_in_inc = set(files_in_inc)
    set_of_files_in_current_dir = set(files_in_current_dir)
    if (set_of_files_in_inc == set_of_files_in_current_dir):
        return True

    print("Error: Files defined in HelperFileAndFeatureNames.inc and files in current folder are not same.")
    print("Please update the HelperFileAndFeatureNames.inc file.\n")

    files_in_inc_but_not_in_dir = set_of_files_in_inc - set_of_files_in_current_dir
    if (len(files_in_inc_but_not_in_dir) != 0):
        print("File(s) defined in HelperFileAndFeatureNames.inc but does not occur in the current folder:")
        print(files_in_inc_but_not_in_dir)

    files_in_dir_but_not_in_inc = set_of_files_in_current_dir - set_of_files_in_inc
    if (len(files_in_dir_but_not_in_inc) != 0):
        print("File(s) occurs in the current folder but not defined in HelperFileAndFeatureNames.inc:")
        print(files_in_dir_but_not_in_inc)

    return False

def check_dependency():
    set_of_features_enum = set(features_enum_list)
    set_of_features_enum_referenced = set(features_enum_referenced_list)
    features_referenced_but_not_defined = set_of_features_enum_referenced - set_of_features_enum
    if (len(features_referenced_but_not_defined) != 0):
        print("Error: Feature(s) is used in dependency but does not defined:")
        print(features_referenced_but_not_defined)
        return False
    return True

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

    content_files_dict = dict(zip(content_files_list, content_files_name_list))
    dpl_extras_content_files_dict = dict(
        zip(dpl_extras_content_files_list, dpl_extras_content_files_name_list))

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

    if (not check_files()):
        exit_script()

    create_dir(inc_files_dir, runtime_files_dir)

    if (sys.platform == "win32"):
        is_os_win = True

    features_enum_file_name = os.path.join(inc_files_dir, "HelperFeatureEnum.inc")
    if (os.path.exists(features_enum_file_name)):
        os.remove(features_enum_file_name)

    for cont_file in content_files_list:
        process_a_file(cont_file, inc_files_dir,
                       runtime_files_dir, False, content_files_dict)
    for cont_file in dpl_extras_content_files_list:
        process_a_file(cont_file, inc_files_dir, runtime_files_dir,
                       True, dpl_extras_content_files_dict)

    if (not check_dependency()):
        exit_script()

    features_enum_file_handle = io.open(features_enum_file_name, "wb")
    features_enum_str = bytes("", 'utf-8')
    features_enum_str = features_enum_str + bytes("#ifdef DPCT_FEATURE_ENUM\n", 'utf-8')
    for element in features_enum_list:
        features_enum_str = features_enum_str + element + bytes(",\n", 'utf-8')
    features_enum_str = features_enum_str + bytes("#endif // DPCT_FEATURE_ENUM\n", 'utf-8')
    features_enum_str = features_enum_str + bytes("#ifdef DPCT_FEATURE_ENUM_FEATURE_PAIR_MAP\n", 'utf-8')
    for element in features_enum_pair_list:
        features_enum_str = features_enum_str + element + bytes(",\n", 'utf-8')
    features_enum_str = features_enum_str + bytes("#endif // DPCT_FEATURE_ENUM_FEATURE_PAIR_MAP\n", 'utf-8')

    features_enum_file_handle.write(features_enum_str)
    features_enum_file_handle.close()

    print("[Note] DPCT *.inc files are generated successfully!")


if __name__ == "__main__":
    main()
