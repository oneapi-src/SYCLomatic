################################################################################
#
# Copyright (C) 2019 - 2020 Intel Corporation. All rights reserved.
#
# The information and source code contained herein is the exclusive property of
# Intel Corporation and may not be disclosed, examined or reproduced in whole or
# in part without explicit written authorization from the company.
#
################################################################################
#!/usr/bin/env bash

set -e -x

# Pass environment variable CUDA_PATH to lit test framework, test
# different version of CUDA by modifing CUDA_PATH

# Default: Assume CUDA is installed in /usr/local/cuda
#   ninja check-clang-dpct

# Test with cuda 8.0:
CUDA_PATH=/usr/local/cuda-8.0 ninja check-clang-dpct
# Test with cuda 9.2:
CUDA_PATH=/usr/local/cuda-9.2 ninja check-clang-dpct
# Test with cuda 10.0:
CUDA_PATH=/usr/local/cuda-10.0 ninja check-clang-dpct
