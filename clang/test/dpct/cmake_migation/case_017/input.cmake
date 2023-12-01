# In the following cases check for changes after running the pattern-rewritter
set (CMAKE_CXX_STANDARD g++)
set (CMAKE_CXX_STANDARD clang++)
set (CMAKE_CXX_STANDARD /usr/bin/clang++)
set (CMAKE_CXX_STANDARD no_a_cpp_compiler)
# Using magic number to be diverse

# In the following cases check for NO changes after running the pattern-rewritter
set (CMAKE_C_STANDARD icx)
set (CMAKE_C_STANDARD /path/to/icx)

set (CMAKE_CXX_STANDARD icpx)
set (CMAKE_CXX_STANDARD /path/to/icpx)

# No change to non-cmake standard
set (Cmake_CXX_Standard clang++)
