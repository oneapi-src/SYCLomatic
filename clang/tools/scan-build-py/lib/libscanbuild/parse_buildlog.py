# -*- coding: utf-8 -*-
#===--------------- parse_buildlog.py --------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

""" This module is responsible for to parse a build log file to generate compilation
    database entries"""

import sys
import re
import os
import logging
from libscanbuild.compilation import split_command
from libscanbuild.shell import encode,decode

__all__ = ['parse_build_log']

def create_compilation_DB_entry(file, command, directory):
    """create one entry of compilation database"""
    abs_dir = os.path.abspath(directory)
    file_path = os.path.join(abs_dir, file)

    if(not os.path.exists(file_path)):
        print('Error: option --work-directory is not set correctly, please specify correct --work-directory.')
        sys.exit(-1)

    fullname = file if os.path.isabs(file) else file_path
    logging.debug('file entry in compilation database: %s', fullname)
    return {'file' : fullname, 'command' : command, 'directory' : abs_dir}

def parse_build_log(file, directory):
    """Parse a build log file to generate compilation database entries"""

    # Currently it covers ninja compile command pattern and make compile command pattern.
    # TODO: Support bazel compile command pattern in the future. 
    ninja_pattern = re.compile(r'\[\d+\/\d+\]')
    warning_pattern = re.compile(r' warning :')
    compiler_dict = {
        'c' : 'cc',
        'c++' : 'c++',
        'cuda' : 'nvcc',
        'mpich' : 'c++'
    }

    entries = []
    with open(file, 'r') as log_file:
        for (ln, line) in enumerate(log_file.readlines()):
            # Skip compiler warning messages
            match = warning_pattern.search(line)
            if match: continue

            pattern_space = re.compile("\s+")
            arg_split = [x for x in pattern_space.split(line) if x]
            match = ninja_pattern.search(line)

            # Skip nijia build process rate like "[2/30]"
            if match: arg_split.pop(0)

            compilation = split_command(arg_split)
            if not compilation: continue

            if compilation.compiler in compiler_dict:
                compiler = compiler_dict[compilation.compiler]
                for source in compilation.files:
                    command = [compiler, '-c'] + compilation.flags + [source]
                    logging.debug('formated as: %s', command)
                    entries.append(create_compilation_DB_entry(source, encode(command), directory))
    return entries
