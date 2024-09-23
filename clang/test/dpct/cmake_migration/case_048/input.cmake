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

add_library(chash OBJECT deps/chash/chash.c deps/chash/chash.h)
add_library(cchash OBJECT deps/cchash/cchash.cc deps/cchash/cchash.h)
add_library(cxxhash OBJECT deps/cxxhash/cxxhash.cxx deps/cxxhash/cxxhash.h)
add_library(cpphash OBJECT deps/cpphash/cpphash.cpp deps/cpphash/cpphash.h)
