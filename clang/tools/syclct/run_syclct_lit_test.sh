#!/usr/bin/env bash

set -e -x

# Pass environment variable CUDA_PATH to lit test framework, test
# different version of CUDA by modifing CUDA_PATH

# Default: Assume CUDA is installed in /usr/local/cuda
#   ninja check-clang-syclct

# Test with cuda 8.0:
CUDA_PATH=/usr/local/cuda-8.0 ninja check-clang-syclct
# Test with cuda 9.2:
CUDA_PATH=/usr/local/cuda-9.2 ninja check-clang-syclct
# Test with cuda 10.0:
CUDA_PATH=/usr/local/cuda-10.0 ninja check-clang-syclct
