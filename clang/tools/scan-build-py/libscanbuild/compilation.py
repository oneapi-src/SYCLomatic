# -*- coding: utf-8 -*-
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
""" This module is responsible for to parse a compiler invocation. """

import re
import os
import collections

__all__ = ['split_command', 'classify_source', 'compiler_language']

# Ignored compiler options map for compilation database creation.
# The map is used in `split_command` method. (Which does ignore and classify
# parameters.) Please note, that these are not the only parameters which
# might be ignored.
#
# Keys are the option name, value number of options to skip
IGNORED_FLAGS = {
    # compiling only flag, ignored because the creator of compilation
    # database will explicitly set it.
    '-c': 0,
    # preprocessor macros, ignored because would cause duplicate entries in
    # the output (the only difference would be these flags). this is actual
    # finding from users, who suffered longer execution time caused by the
    # duplicates.
    '-MD': 0,
    '-MMD': 0,
    '-MG': 0,
    '-MP': 0,
    '-MF': 1,
    '-MT': 1,
    '-MQ': 1,
    # linker options, ignored because for compilation database will contain
    # compilation commands only. so, the compiler would ignore these flags
    # anyway. the benefit to get rid of them is to make the output more
    # readable.
    '-static': 0,
    '-shared': 0,
    '-s': 0,
    '-rdynamic': 0,
    '-l': 1,
    '-L': 1,
    '-u': 1,
    '-z': 1,
    '-T': 1,
    '-Xlinker': 1,
    # All of the following options are ignored, as they are not related to syclct tool
    '-gencode': 1,
    '-ccbin': 1,
    '-ptx': 0,
    '-Xcompiler': 0,
    '-cuda': 0,
    '-cubin': 0,
    '-fatbin': 0,
    '-gpu': 0,
    '-dc': 0,
    '-dw': 0,
    '-dlink': 0,
    '-link': 0,
    '-lib': 0,
    '-run': 0,
    '-Xarchive': 0,
    '-Xptxas': 0,
    '-Xnvlink': 0,
    '-noprof': 0,
    '-dryrun': 0,
    '-keep': 0,
    '-keep-dir': 0,
    '-clean': 0,
    '-code': 1,
    '-rdc': 1,
    '-e': 1,
    '-maxrregcount': 1,
    '-use_fast_math': 0,
    '-ftz': 1,
    '-prec-div': 1,
    '-prec-sqrt': 1,
    '-fmad': 1,
    '-default-stream': 1,
    '-keep-device-functions': 0,
    '-src-in-ptx': 0,
    '-restrict': 0,
    '-Wno-deprecated-gpu-targets': 0,
    '-res-usage': 0,
    '-V': 0,
    '-optf': 0,
}

# Known C/C++ compiler executable name patterns
COMPILER_PATTERNS = frozenset([
    re.compile(r'^(intercept-|analyze-|)c(c|\+\+)$'),
    re.compile(r'^([^-]*-)*[mg](cc|\+\+)(-\d+(\.\d+){0,2})?$'),
    re.compile(r'^([^-]*-)*clang(\+\+)?(-\d+(\.\d+){0,2})?$'),
    re.compile(r'^llvm-g(cc|\+\+)$'),
])


def split_command(command):
    """ Returns a value when the command is a compilation, None otherwise.

    The value on success is a named tuple with the following attributes:

        files:    list of source files
        flags:    list of compile options
        compiler: string value of 'c', 'c++' or 'cuda' """

    # the result of this method
    result = collections.namedtuple('Compilation',
                                    ['compiler', 'flags', 'files'])
    result.compiler = compiler_language(command)
    result.flags = []
    result.files = []
    # quit right now, if the program was not a C/C++ compiler
    if not result.compiler:
        return None
    # iterate on the compile options
    args = iter(command[1:])
    for arg in args:
        # quit when compilation pass is not involved
        if arg in {'-E', '-S', '-cc1', '-M', '-MM', '-###'}:
            return None
        # ignore some flags
        elif arg in IGNORED_FLAGS:
            count = IGNORED_FLAGS[arg]
            for _ in range(count):
                next(args)
        elif arg in {'-lmpichcxx', '-lmpich', '-lmpi_cxx', '-lmpi'}:
            result.compiler = 'mpich'
            result.flags.append(arg)
        elif re.match(r'^-(l|L|Wl,).+', arg):
            pass
        # some parameters could look like filename, take as compile option
        elif arg in {'-D', '-I'}:
            result.flags.extend([arg, next(args)])
        # parameter which looks source file is taken...
        elif re.match(r'^[^-].+', arg) and classify_source(arg):
            # nvcc compiler compiles source files with suffix cuda(.cu) and
            # cpp(.cc,.c++,.cpp) should be added into compilation database.
            #
            # while other compiler takes all type of sources.
            #
            # ====================================================
            # | Compiler |          Accept Source Type           |
            # ====================================================
            # |  nvcc    |  cuda(.cu), c++(.cc, .cpp, .c++, ...) |
            # ----------------------------------------------------
            # |  Others  |  All                                  |
            # ----------------------------------------------------
            if result.compiler == 'cuda' and classify_source(arg) not in ['cuda', 'c++']:
                return None
            else:
                result.files.append(arg)
        # ignore -code=xx option.
        elif re.match(r'^-code=',arg):
            pass
        # and consider everything else as compile option.
        else:
            result.flags.append(arg)
    #Append buildin cuda options for migration tool to identy right code path
    if result.compiler == 'cuda':
        result.flags.append("-D__CUDA_ARCH__=400")
        result.flags.append("-D__CUDACC__=1")
    # do extra check on number of source files
    if result.files:
        return result
    # linker command should be added into compilation database
    elif len(result.files) == 0 and result.compiler == 'cuda':
        return result
    # linker command should be added into compilation database
    elif len(result.files) == 0 and result.compiler == 'mpich':
        return result
    else:
        return None


def classify_source(filename, c_compiler=True):
    """ Return the language from file name extension. """

    mapping = {
        '.c': 'c' if c_compiler else 'c++',
        '.i': 'c-cpp-output' if c_compiler else 'c++-cpp-output',
        '.ii': 'c++-cpp-output',
        '.m': 'objective-c',
        '.mi': 'objective-c-cpp-output',
        '.mm': 'objective-c++',
        '.mii': 'objective-c++-cpp-output',
        '.C': 'c++',
        '.cc': 'c++',
        '.CC': 'c++',
        '.cp': 'c++',
        '.cpp': 'c++',
        '.cxx': 'c++',
        '.c++': 'c++',
        '.C++': 'c++',
        '.txx': 'c++',
        '.cu' : 'cuda'
    }

    __, extension = os.path.splitext(os.path.basename(filename))
    return mapping.get(extension)


def compiler_language(command):
    """ A predicate to decide the command is a compiler call or not.

    Returns 'c', 'c++' or 'cuda' when it match. None otherwise. """

    cplusplus = re.compile(r'^(.+)(\+\+)(-.+|)$')

    if command:
        executable = os.path.basename(command[0])
        if any(pattern.match(executable) for pattern in COMPILER_PATTERNS):
            return 'c++' if cplusplus.match(executable) else 'c'
        if executable == 'nvcc':
            return 'cuda'
    return None
