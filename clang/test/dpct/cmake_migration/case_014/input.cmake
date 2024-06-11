set(CPP_VER_98 98)
set(CPP_VER_11 11)
set(CPP_VER_14 14)
set(CPP_VER_17 17)
set(CPP_VER_20 20)
set(CPP_VER_23 23)
set(CPP_VER_26 26)

# In the following cases check for changes after running the
# pattern-rewritter
set (CMAKE_CXX_STANDARD CPP_VER_98)
set (CMAKE_CXX_STANDARD CPP_VER_11)
set (CMAKE_CXX_STANDARD CPP_VER_14)
# Using magic number to be diverse
set (CMAKE_CXX_STANDARD 14)

set (CMAKE_CXX_STANDARD CPP_VER_17)
set (CMAKE_CXX_STANDARD CPP_VER_20)

# In the following cases no change is expected but due to limitations of
# pattern-rewriter the C++ standard is updated to C++20. This will be fixed in
# future.
set (CMAKE_CXX_STANDARD CPP_VER_23)
set (CMAKE_CXX_STANDARD CPP_VER_26)

set (CMake_CXX_Standard CPP_VER_11)

# We should not touch C standard setting;
# Using magic number to be diverse
set (CMAKE_C_STANDARD 11)
