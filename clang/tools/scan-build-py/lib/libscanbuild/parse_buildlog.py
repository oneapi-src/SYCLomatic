# -*- coding: utf-8 -*-
# Copyright (C) Intel Corporation. All rights reserved.
#
# The information and source code contained herein is the exclusive
# property of Intel Corporation and may not be disclosed, examined
# or reproduced in whole or in part without explicit written authorization
# from the company.

""" This module is responsible for to parse a build log file to generate compilation
    database entries"""

import re
import os
import logging
from libscanbuild.compilation import split_command
from libscanbuild.shell import encode,decode

__all__ = ['parse_build_log']

def create_compilation_DB_entry(file, command, directory):
    """create one entry of compilation database"""
    abs_dir = os.path.abspath(directory)
    fullname = file if os.path.isabs(file) else os.path.join(abs_dir, file)
    return {'file' : fullname, 'command' : command, 'directory' : abs_dir}

def parse_build_log(file):
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
                    entries.append(create_compilation_DB_entry(source, encode(command), file))
    return entries
