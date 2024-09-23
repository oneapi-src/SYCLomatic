add_library(target
            foo.cpp
            layer.cudnn.cpp
            foo.h
            )
            
add_library(bar bar.cpp bar.h)

target_compile_features(${TARGET} PUBLIC cxx_std_14)
set(CMAKE_CXX_STANDARD 14)
target_compile_features(culib PRIVATE cxx_std_14)
set_target_properties(target_one PROPERTIES CXX_STANDARD 17)
add_compile_options(-std=c++17)

add_library(xxhash OBJECT deps/xxhash/xxhash.c deps/xxhash/xxhash.h)
add_library(xxhash OBJECT deps/xxhash/xxhash.cc deps/xxhash/xxhash.h)
