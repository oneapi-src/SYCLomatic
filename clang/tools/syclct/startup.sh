#!/bin/bash
################################################################################
#
# Copyright (C) 2018 Intel Corporation. All rights reserved.
#
# The information and source code contained herein is the exclusive property of
# Intel Corporation and may not be disclosed, examined or reproduced in whole or
# in part without explicit written authorization from the company.
#
################################################################################

export SYCLCT_BUNDLE_ROOT=$(realpath $(dirname "${BASH_SOURCE[0]}"))

# Binary check for SYCL CT
if [[ ! -e $SYCLCT_BUNDLE_ROOT/bin/syclct || \
      ! -e $SYCLCT_BUNDLE_ROOT/bin/intercept-build ]]; then
    printf "[\033[0;31mERROR\033[0m] Cannot find neccessary syclct binary\n"
    return 1
fi

# SYCL CT bundle
export PATH=$SYCLCT_BUNDLE_ROOT/bin:$PATH
export CPATH=$SYCLCT_BUNDLE_ROOT/include:$CPATH

# SYCL compiler - ComputeCpp
if [[ -n "$COMPUTECPP_ROOT" ]]; then
    if [ ! -f "$COMPUTECPP_ROOT/bin/compute++" ]; then
        printf "[\033[0;31mERROR\033[0m] Cannot find compute++\n"
        return 1
    fi
    if [ ! -d "$COMPUTECPP_ROOT/include" ]; then
        printf "[\033[0;31mERROR\033[0m] Cannot find ComputeCpp include directory\n"
        return 1
    fi
    if [ ! -d "$COMPUTECPP_ROOT/lib" ]; then
        printf "[\033[0;31mERROR\033[0m] Cannot find ComputeCpp lib directory\n"
        return 1
    fi
    printf "[\033[0;32mINFO\033[0m] SYCL compiler: $COMPUTECPP_ROOT/bin/"
    printf "compute++\n"
    export PATH=$COMPUTECPP_ROOT/bin:$PATH
    export CPATH=$COMPUTECPP_ROOT/include:$CPATH
    export LD_LIBRARY_PATH=$COMPUTECPP_ROOT/lib:$LD_LIBRARY_PATH

    # OpenCL headers if needed
    if [ -n "$OPENCL_INCLUDE_DIRECTORY" ]; then
        export CPATH=$OPENCL_INCLUDE_DIRECTORY:$CPATH
    else
        printf "[\033[0;33mWARNING\033[0m] OPENCL_INCLUDE_DIRECTORY is not set."
        printf " Please make sure OpenCL include directory is in CPATH or set "
        printf "OPENCL_INCLUDE_DIRECTORY and re-source this script, otherwise "
        printf "ComputeCpp will not work well.\n"
    fi

    # OpenCL libraries if needed
    if [ -n "$OPENCL_LIBRARY_DIRECTORY" ]; then
        export LD_LIBRARY_PATH=$OPENCL_LIBRARY_DIRECTORY:$LD_LIBRARY_PATH
        # If the directory 'oclcpu' exists, then it should be appended
        [ -d "$OPENCL_LIBRARY_DIRECTORY/oclcpu" ] && \
        export LD_LIBRARY_PATH=$OPENCL_LIBRARY_DIRECTORY/oclcpu:$LD_LIBRARY_PATH
        # If the directory 'oclgpu' exists, then it should be appended
        [ -d "$OPENCL_LIBRARY_DIRECTORY/oclgpu" ] && \
        export LD_LIBRARY_PATH=$OPENCL_LIBRARY_DIRECTORY/oclgpu:$LD_LIBRARY_PATH
    else
        printf "[\033[0;33mWARNING\033[0m] OPENCL_LIBRARY_DIRECTORY is not set."
        printf " Please make sure OpenCL library directory is in LD_LIBRARY_PATH"
        printf " or set OPENCL_LIBRARY_DIRECTORY and re-source this script, "
        printf "otherwise ComputeCpp will not work well.\n"
    fi
fi

export LIBRARY_PATH=$LD_LIBRARY_PATH

# Default CUDA SDK found by syclct
printf "[\033[0;32mINFO\033[0m] CUDA SDK selected by syclct: \
$(syclct -- -v 2>&1 | grep -oP '(?<=Found CUDA installation: ).*(?=,)') \n"
