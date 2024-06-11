cmake_minimum_required(VERSION 3.24)
project(AProject)
cuda_add_library(target_one MODULE nosrc1.cu)

# Test C++ 99 standard
set_target_properties(target_one
                      PROPERTIES
                      CXX_STANDARD 99)

# Test C++ 11 standard
set_target_properties(target_one
                      PROPERTIES
                      CXX_STANDARD 11)

# Test C++ 14 standard
set_target_properties(target_one
                      PROPERTIES
                      CXX_STANDARD 14)

# No change unless property matches exactly
set_target_properties(target_one
                      PROPERTIES
                      CXX_Standard 14)

set_target_properties(target_one
                      PROPERTIES
                      CXX_STANDARD 20)

# Test CUDA_SEPARABLE_COMPILATION
set_target_properties(target_one
                      PROPERTIES
                      CUDA_SEPARABLE_COMPILATION Off)

# No change unless property matches exactly
set_target_properties(target_one
                      PROPERTIES
                      Cuda_SEPARABLE_COMPILATION Off)

# Test CUDA_ARCHITECHTURES
set_target_properties(target_one
                      PROPERTIES
                      CUDA_ARCHITECHTURES Auto)

# No change unless property matches exactly
set_target_properties(target_one
                      PROPERTIES
                      cuda_architechtures All)

cuda_add_library(target_two MODULE nosrc2.cu)

# Test two target specs
set_target_properties(target_one target_two
                      PROPERTIES
                      CUDA_ARCHITECHTURES Kepler+Tesla)

# Make sure other properties are intact
set_target_properties(target_one
                      PROPERTIES
                      CUDA_ARCHITECHTURES Kepler+Tesla
                      OTHER_PROPERTY 1)

