# -*- coding: utf-8 -*-
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
""" This module is responsible for to parse a compiler invocation. """

import re
import os
import collections

__all__ = ["split_command", "classify_source", "compiler_language"]

# Ignored compiler options map for compilation database creation.
# The map is used in `split_command` method. (Which does ignore and classify
# parameters.) Please note, that these are not the only parameters which
# might be ignored.
#
# Keys are the option name, value number of options to skip
IGNORED_FLAGS = {
    # compiling only flag, ignored because the creator of compilation
    # database will explicitly set it.
    "-c": 0,
    "--compile": 0,
    # preprocessor macros, ignored because would cause duplicate entries in
    # the output (the only difference would be these flags). this is actual
    # finding from users, who suffered longer execution time caused by the
    # duplicates.
    "-MD": 0,
    "-MMD": 0,
    "-MG": 0,
    "-MP": 0,
    "-MF": 1,
    "--dependency-output": 1,
    "-MT": 1,
    "-MQ": 1,
    # linker options, ignored because for compilation database will contain
    # compilation commands only. so, the compiler would ignore these flags
    # anyway. the benefit to get rid of them is to make the output more
    # readable.
    "-static": 0,
    #"-shared": 0,
    "-s": 0,
    "-rdynamic": 0,
    "-l": 1,
    "-L": 1,
    "-u": 1,
    "-z": 1,
    "-T": 1,

    # All of the following options are ignored, as they are not related to dpct tool
    "-march": 1,
    "--cuda": 0,
    "-cuda": 0,
    "--threads": 1,
    "-threads": 1,

    # --cubin/-cubin, --ptx/-ptx are kept in compilation database, which are
    # used for driver API migration.
    #"--cubin": 0,
    #"-cubin": 0,
    #"--ptx" : 0,
    #"-ptx" : 0,

    "--device-c": 0,
    "-dc": 0,
    "--device-w": 0,
    "-dw": 0,
    "--device-link": 0,
    "-dlink": 0,
    "--link": 0,
    "-link": 0,
    "--lib": 0,
    "-lib": 0,
    "--run": 0,
    "-run": 0,
    "--pre-include": 1,
    "--library": 1,
    "-l": 1,
    "--library-path": 1,
    "-L": 1,
    "--output-directory": 1,
    "-odir" : 1,
    "--compiler-bindir": 1,
    "-ccbin": 1,
    "-cudart": 1,
    "--cudart": 1,
    "--libdevice-directory": 1,
    "-ldir": 1,
    "--use-local-env": 0,
    "--profile": 0,
    "-pg": 0,
    "--debug": 0,
    "-g": 0,
    "--generate-line-info": 0,
    "-lineinfo": 0,
    #"--shared": 0,
    #"-shared" : 0,
    "--x": 1,
    "-x": 1,
    "--no-host-device-initializer-list": 0,
    "-nohdinitlist": 0,
    "--no-host-device-move-forward": 0,
    "-nohdmoveforward": 0,
    "--expt-relaxed-constexpr": 0,
    "-expt-relaxed-constexpr": 0,
    "--expt-extended-lambda": 0,
    "-expt-extended-lambda" : 0,
    "-Xcompiler": 1,
    "--compiler-options": 1,
    "--compiler-options": 1,
    "-Xcompiler": 1,
    "--linker-options": 1,
    "-Xlinker": 1,
    "--archive-options": 1,
    "-Xarchive": 1,
    "--ptxas-options": 1,
    "-Xptxas": 1,
    "--nvlink-options": 1,
    "-Xnvlink": 1,
    "-noprof": 0,
    "--dont-use-profile": 0,
    "-dryrun": 0,
    "--dryrun": 0,
    "--verbose": 0,
    "-v": 0,
    "--keep": 0,
    "-keep": 0,
    "--keep-dir": 1,
    "-keep-dir": 1,
    "--save-temps": 0,
    "-save-temps": 0,
    "--clean-targets": 0,
    "-clean": 0,
    "--run-args": 1,
    "-run-args": 1,
    "--input-drive-prefix": 1,
    "-idp": 1,
    "--dependency-drive-prefix": 1,
    "-ddp":  1,
    "--drive-prefix": 1,
    "-dp": 1,
    "--dependency-target-name": 1,
    "-MT": 1,
    "--no-align-double": 0,
    "--no-device-link": 0,
    "-nodlink": 0,
    "--gpu-code": 1,
    "-code": 1,
    "-gencode": 1,
    "--generate-code": 1,
    "--relocatable-device-code": 1,
    "-rdc": 1,
    "--entries": 1,
    "-e": 1,
    "--maxrregcount": 1,
    "-maxrregcount": 1,
    "--use_fast_math": 0,
    "-use_fast_math": 0,
    "--ftz": 1,
    "-ftz": 1,
    "--prec-div": 1,
    "-prec-div": 1,
    "--prec-sqrt": 1,
    "-prec-sqrt": 1,
    "--fmad": 1,
    "-fmad": 1,
    "--default-stream" : 1,
    "-default-stream" : 1,
    "--keep-device-functions" : 0,
    "-keep-device-functions" : 0,
    "--source-in-ptx" : 0,
    "-src-in-ptx" : 0,
    "--restrict" : 0,
    "-restrict" : 0,
    "--Wreorder" : 0,
    "-Wreorder" : 0,
    "--Wno-deprecated-declarations" : 0,
    "-Wno-deprecated-declarations" : 0,
    "--Wno-deprecated-gpu-targets" : 0,
    "-Wno-deprecated-gpu-targets" : 0,
    "--Werror" : 1,
    "-Werror" : 1,
    "--resource-usage" : 0,
    "-res-usage" : 0,
    "--extensible-whole-program" : 0,
    "-ewp" : 0,
    "--no-compress" : 0,
    "-no-compress" : 0,
    "--help" : 0,
    "-h" : 0,
    "--version" : 0,
    "-V" : 0,
    #"-fopenmp": 0,
    "-forward-unknown-to-host-compiler" : 0,
    "-Xllc" : 0,
    "--Xllc" : 0,
    "-Xcicc" : 1,
    "sed" : 1,
    "2>&1" : 0,
    "|" : 0
}

MAP_FLAGS = {
    "--fatbin": "-Xcuda-fatbinary",
    "-fatbin": "-Xcuda-fatbinary",
    "-G": "--cuda-noopt-device-debug",
    "--device-debug": "--cuda-noopt-device-debug",
    "--machine" : "-m",
    "-m" : "-m",
    "--gpu-architecture" : "--cuda-gpu-arch=",
    "-arch" : "--cuda-gpu-arch=",
    "--disable-warnings" : "--no-warnings",
    "-w" : "--no-warnings",
}

# Clang option --cuda-gpu-arch do not support the argument like "compute_30"
def gpu_virtual_arch_to_arch(virtual_arch):
    pattern_underline = re.compile("_")
    virtual_arch_split = [x for x in pattern_underline.split(virtual_arch) if x]
    virtual_arch_ver = virtual_arch_split[1]
    return 'sm_' + virtual_arch_ver

def sub_arg_split(arg, separator):
    arg_split = []
    pattern_space = re.compile("\s+")
    pattern_comma = re.compile(",")

    arg_value = [x for x in arg.split(separator) if x]

    # To handle combined optons like ' -DXXX -O3 -w -march=native '"
    if re.search(r'\s+', arg_value[0]):
        arg_split = [x for x in pattern_space.split(arg_value[0]) if x]
    # To handle combined optons like ',"-Wall","-O2","-Wextra","-g"'
    elif re.search(r',', arg_value[0]):
        arg_split = [x.strip('"') for x in pattern_comma.split(arg_value[0]) if x]
    return arg_split

# Known C/C++ compiler executable name patterns
COMPILER_PATTERNS = frozenset(
    [
        re.compile(r"^(intercept-|analyze-|)c(c|\+\+)$"),
        re.compile(r"^([^-]*-)*[mg](cc|\+\+)(-\d+(\.\d+){0,2})?$"),
        re.compile(r"^([^-]*-)*clang(\+\+)?(-\d+(\.\d+){0,2})?$"),
        re.compile(r"^llvm-g(cc|\+\+)$"),
        re.compile(r"^ic(c|pc|px|x)$"),
        re.compile(r"^mpi(cc|cxx|gcc|gxx|icc|icpc)$"),
    ]
)

def parse_option_file(file):
    """Parse command line options from specified file"""
    options = []
    with open(file, 'r') as option_file:
        for (ln, line) in enumerate(option_file.readlines()):
            pattern_space = re.compile("\s+")
            arg_split = [x for x in pattern_space.split(line) if x]
            flags = parse_args(iter(arg_split))
            options.extend(flags[0])
    return options

def abspath(cwd, name):
    """Create normalized absolute path from input filename."""
    fullname = name if os.path.isabs(name) else os.path.join(cwd, name)
    return os.path.normpath(fullname)

def parse_args(args, directory='.'):
    flags = []
    compiler = ''
    files = []
    is_processing_cmd = False
    preprocess_output_files = []
    for arg in args:
        # Remove comment options started with "#" in command line, like "# -gencode arch=compute_61,code=sm_61"
        if re.match(r'^\#', arg):
            arg_next = next(args, None)
            while(arg_next != None):
                 arg_next = next(args, None)
        elif arg == '-o':
            flags.append(arg)
            arg_next = next(args, None)
            if(arg_next != None):
                flags.append(arg_next)
                preprocess_output_files.append(arg_next)
            pass
        # quit when compilation pass is not involved
        elif arg in {'-E', '-S', '-cc1', '-M', '-MM', '-###'}:
            return None
        # map nvcc flags
        elif arg in MAP_FLAGS:
            new_flag = MAP_FLAGS[arg]
            if arg == '--machine' or arg == '-m':
                # '-m 32' -> '-m32' or '--machine 32' -> '-m32'
                arg_next = next(args)
                new_flag += arg_next
            elif arg == '--gpu-architecture' or arg == '-arch':
                arg_next = next(args)
                if arg_next == "all": # Skip option "-arch all" and "--gpu-architecture all"
                    continue
                arg_next = gpu_virtual_arch_to_arch(arg_next)
                # clang cuda parser does not support CUDA gpu architecture: sm_13
                if(arg_next == "sm_13"):
                    continue
                new_flag += arg_next
            flags.append(new_flag)
        # ignore some flags
        elif arg in IGNORED_FLAGS:
            count = IGNORED_FLAGS[arg]
            if arg == '-cuda' or arg == '--cuda':
                is_processing_cmd = True
            for _ in range(count):
                if arg == '-Xcompiler' or arg == '--compiler-options'  \
                   or arg == '-Xarchive' or arg == '--archive-options' \
                   or arg == '-Xptxas' or arg == '--ptxas-options'     \
                   or arg == '-Xnvlink' or arg == '--nvlink-options'   \
                   or arg == '-run-args' or arg == '--run-args':
                    # for '-Xcompiler', it may be with arg like "'-Xcompiler' ' -DXXX -O3 -w -march=native '"
                    arg_next = next(args, None)
                    if(arg_next != None):
                        arg_next = arg_next.strip()
                        arg_split = []
                        pattern_space = re.compile("\s+")
                        pattern_comma = re.compile(",")

                        # To handle combined optons like ' -DXXX -O3 -w -march=native '"
                        if re.search(r'\s+', arg_next):
                            arg_split = [x for x in pattern_space.split(arg_next) if x]
                        # To handle combined optons like ',"-Wall","-O2","-Wextra","-g"'
                        elif re.search(r',', arg_next):
                            arg_split = [x.strip('"') for x in pattern_comma.split(arg_next) if x]

                        # In the case of else condition, it is difficult to tell whether arg_split[0] is
                        # the value of option '-Xcompiler' or an independent argument, so just treat it as an independent argument,
                        # it will be processed in the next outer loop:
                        # E.g: [..., '-Xcompiler', '"', '-g', '-O3', '-Wall', '"',...]
                        if re.search(r'\s+', arg_next) or re.search(r',', arg_next):
                            xcompiler_flags = parse_args(iter(arg_split))
                            flags.extend(xcompiler_flags[0])
                        # To handle -Xcompiler -fopenmp
                        elif arg == '-Xcompiler':
                            flags.extend([arg_next.strip('"')])
                else:
                    next(args)
        elif arg in {'-lmpichcxx', '-lmpich', '-lmpi_cxx', '-lmpi'}:
            compiler = 'mpich'
            flags.append(arg)
        elif re.match(r'^-(l|L|Wl,).+', arg):
            pass
        # some parameters could look like filename, take as compile option
        elif arg in {'-D', '-I'}:
            flags.extend([arg, next(args)])
        # some command line options are passed from file specified by
        elif arg in {'-optf', '--options-file'}:
            value = next(args)
            option_file = abspath(directory, value)
            flags.extend(parse_option_file(option_file))
            pass
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
            if compiler == 'cuda' and classify_source(arg) not in ['cuda', 'c++']:
                return None
            else:
                files.append(arg)
        # ignore -fmad=xx option.
        elif re.match(r'^-fmad=', arg):
            pass
        # ignore -x=xx option.
        elif re.match(r'^-x=', arg):
            pass
        # ignore -Xcompiler=xx option.
        elif re.match(r'^-Xcompiler=', arg):
            arg_split = sub_arg_split(arg, '-Xcompiler=')
            xcompiler_flags = parse_args(iter(arg_split))
            flags.extend(xcompiler_flags[0])
            pass
        elif re.match(r'^--compiler-options=', arg):
            arg_split = sub_arg_split(arg, '--compiler-options=')
            xcompiler_flags = parse_args(iter(arg_split))
            flags.extend(xcompiler_flags[0])
            pass
        elif re.match(r'^-Xlinke=', arg):
            arg_split = sub_arg_split(arg, '-Xlinkes=')
            xcompiler_flags = parse_args(iter(arg_split))
            flags.extend(xcompiler_flags[0])
            pass
        elif re.match(r'^--linker-options=', arg):
            arg_split = sub_arg_split(arg, '--linker-options=')
            xcompiler_flags = parse_args(iter(arg_split))
            flags.extend(xcompiler_flags[0])
            pass
        elif re.match(r'^-Xarchive=', arg):
            arg_split = sub_arg_split(arg, '-Xarchive=')
            xcompiler_flags = parse_args(iter(arg_split))
            flags.extend(xcompiler_flags[0])
            pass
        elif re.match(r'^--archive-options=', arg):
            arg_split = sub_arg_split(arg, '--archive-options=')
            xcompiler_flags = parse_args(iter(arg_split))
            flags.extend(xcompiler_flags[0])
            pass
        elif re.match(r'^-Xptxa=', arg):
            arg_split = sub_arg_split(arg, '-Xptxa=')
            xcompiler_flags = parse_args(iter(arg_split))
            flags.extend(xcompiler_flags[0])
            pass
        elif re.match(r'^--ptxas-options=', arg):
            arg_split = sub_arg_split(arg, '--ptxas-options=')
            xcompiler_flags = parse_args(iter(arg_split))
            flags.extend(xcompiler_flags[0])
            pass
        elif re.match(r'^-Xnvlink=', arg):
            arg_split = sub_arg_split(arg, '-Xnvlink=')
            xcompiler_flags = parse_args(iter(arg_split))
            flags.extend(xcompiler_flags[0])
            pass
        elif re.match(r'^--nvlink-options=', arg):
            arg_split = sub_arg_split(arg, '--nvlink-options=')
            xcompiler_flags = parse_args(iter(arg_split))
            flags.extend(xcompiler_flags[0])
            pass
        elif re.match(r'^-run-args=', arg):
            arg_split = sub_arg_split(arg, '-run-args=')
            xcompiler_flags = parse_args(iter(arg_split))
            flags.extend(xcompiler_flags[0])
            pass
        elif re.match(r'^--run-args=', arg):
            arg_split = sub_arg_split(arg, '--run-args=')
            xcompiler_flags = parse_args(iter(arg_split))
            flags.extend(xcompiler_flags[0])
            pass
        # ignore -march=xx option, .e.g -march=native
        elif re.match(r'^-march=', arg):
            pass
        elif re.match(r'^--gpu-architecture=', arg) or re.match(r'^-arch=', arg):
            #split by '=' and strip whitespace
            result = [x.strip() for x in arg.split('=')]

            if result[1] == 'all': # Skip option "-arch=all" and "--gpu-architecture=all"
                continue

            new_opt = MAP_FLAGS[result[0]] + gpu_virtual_arch_to_arch(result[1])
            flags.append(new_opt)
            pass
        elif re.match(r'^--options-file=', arg) or re.match(r'^-optf=', arg):
            #split by '=' and strip whitespace
            result = [x.strip() for x in arg.split('=')]
            value = result[1]
            option_file = abspath(directory, value)
            flags.extend(parse_option_file(option_file))
            pass
        elif re.match(r'^-isystem=', arg):
            #"-isystem=<directory>"" is not a effective option in clang command line,
            #So replace "-isystem=<directory>" with "-isystem<directory>".
            new_opt = arg.replace("-isystem=", "-isystem", 8)
            flags.append(new_opt)
            pass
        elif re.match(r'^--pre-include=', arg):
            pass
        elif re.match(r'^--library=', arg):
            pass
        elif re.match(r'^-l=', arg):
            pass
        elif re.match(r'^--library-path=', arg):
            pass
        elif re.match(r'^-L=', arg):
            pass
        elif re.match(r'^--output-directory=', arg):
            pass
        elif re.match(r'^-odir=', arg):
            pass
        elif re.match(r'^--compiler-bindir=', arg):
            pass
        elif re.match(r'^-ccbin=', arg):
            pass
        elif re.match(r'^--cudart=', arg):
            pass
        elif re.match(r'^-cudart=', arg):
            pass
        elif re.match(r'^--libdevice-directory=', arg):
            pass
        elif re.match(r'^-ldir=', arg):
            pass
        elif re.match(r'^--x=', arg):
            pass
        elif re.match(r'^-x=', arg):
            pass
        elif re.match(r'^--keep-dir=', arg):
            pass
        elif re.match(r'^--input-drive-prefix=', arg):
            pass
        elif re.match(r'^-idp=', arg):
            pass
        elif re.match(r'^--dependency-drive-prefix=', arg):
            pass
        elif re.match(r'^-ddp=', arg):
            pass
        elif re.match(r'^--drive-prefix=', arg):
            pass
        elif re.match(r'^-dp=', arg):
            pass
        elif re.match(r'^--dependency-target-name=', arg):
            pass
        elif re.match(r'^-MT=', arg):
            pass
        elif re.match(r'^--gpu-code=', arg):
            pass
        elif re.match(r'^-code=', arg):
            pass
        elif re.match(r'^--generate-code=', arg):
            pass
        elif re.match(r'^-gencode=', arg):
            pass
        elif re.match(r'^--relocatable-device-code=', arg):
            pass
        elif re.match(r'^-rdc=', arg):
            pass
        elif re.match(r'^--entries=', arg):
            pass
        elif re.match(r'^-e=', arg):
            pass
        elif re.match(r'^--maxrregcount=', arg):
            pass
        elif re.match(r'^-maxrregcount=', arg):
            pass
        elif re.match(r'^--use_fast_math=', arg):
            pass
        elif re.match(r'^-use_fast_math=', arg):
            pass
        elif re.match(r'^--ftz=', arg):
            pass
        elif re.match(r'^-ftz=', arg):
            pass
        elif re.match(r'^--prec-div=', arg):
            pass
        elif re.match(r'^-prec-div=', arg):
            pass
        elif re.match(r'^--prec-sqrt=', arg):
            pass
        elif re.match(r'^-prec-sqrt=', arg):
            pass
        elif re.match(r'^--fmad=', arg):
            pass
        elif re.match(r'^-fmad=', arg):
            pass
        elif re.match(r'^--default-stream=', arg):
            pass
        elif re.match(r'^-default-stream=', arg):
            pass
        elif re.match(r'^--Werror=', arg):
            pass
        elif re.match(r'^-Werror=', arg):
            pass
        # E.g \" is imported by option like
        # [..., '-Xcompiler', '"', '-g', '-O3', '-Wall', '"',...]
        elif arg in {'\"'}:
            pass
        elif re.match(r'^-dag-vectorize-ops=', arg):
            pass
        # Remove double quotes from including path like -I"/path/"
        elif re.match(r'^-I\"', arg):
            arg = arg.replace('"', '')
            flags.append(arg)
            pass
        # and consider everything else as compile option.
        else:
           flags.append(arg)

    if not is_processing_cmd:
        preprocess_output_files = []

    return [flags, compiler, files, preprocess_output_files]

def split_command(command, directory='.'):
    """Returns a value when the command is a compilation, None otherwise.

    The value on success is a named tuple with the following attributes:

        files:    list of source files
        flags:    list of compile options
        compiler: string value of 'c', 'c++' or 'cuda'"""

    # the result of this method
    result = collections.namedtuple("Compilation", ["compiler", "flags", "files", "preprocess_output_files"])
    result.compiler = compiler_language(command)
    result.flags = []
    result.files = []
    # quit right now, if the program was not a C/C++ compiler
    if not result.compiler:
        return None
    # iterate on the compile options
    args = iter(command[1:])

    ret = parse_args(args, directory)
    if ret == None:
        return None
    else:
        result.flags = ret[0]
        if ret[1] != "":
            result.compiler = ret[1]
        result.files = ret[2]
        result.preprocess_output_files = ret[3]

    #Append buildin cuda options for migration tool to identy right code path
    if result.compiler == "cuda":
        result.flags.append("-D__CUDACC__=1")
    # do extra check on number of source files
    if result.files:
        return result
    # linker command should be added into compilation database
    elif len(result.files) == 0 and result.compiler == 'ld':
        return result
    # linker command should be added into compilation database
    elif len(result.files) == 0 and result.compiler == 'cuda':
        return result
    # linker command should be added into compilation database
    elif len(result.files) == 0 and result.compiler == 'mpich':
        return result
    elif len(result.files) == 0 and result.compiler == 'ar':
        return result
    else:
        return None


def classify_source(filename, c_compiler=True):
    """Return the language from file name extension."""

    mapping = {
        ".c": "c" if c_compiler else "c++",
        ".i": "c-cpp-output" if c_compiler else "c++-cpp-output",
        ".ii": "c++-cpp-output",
        ".m": "objective-c",
        ".mi": "objective-c-cpp-output",
        ".mm": "objective-c++",
        ".mii": "objective-c++-cpp-output",
        ".C": "c++",
        ".cc": "c++",
        ".CC": "c++",
        ".cp": "c++",
        ".cpp": "c++",
        ".cxx": "c++",
        ".c++": "c++",
        ".C++": "c++",
        ".txx": "c++",
        ".cu" : "cuda",
    }

    __, extension = os.path.splitext(os.path.basename(filename))
    return mapping.get(extension)


def compiler_language(command):
    """A predicate to decide the command is a compiler call or not.

    Returns 'c', 'c++' or 'cuda' when it match. None otherwise."""

    cplusplus = re.compile(r"^(.+)(\+\+)(-.+|)$")

    if command:
        executable = os.path.basename(command[0])
        if any(pattern.match(executable) for pattern in COMPILER_PATTERNS):
            return "c++" if cplusplus.match(executable) else "c"
        if executable == "nvcc":
            return "cuda"
        if executable == "intercept-stub":
            return "cuda"
        if executable == "ld":
            return "ld"
        if executable == "ar":
            return "ar"
    return None
