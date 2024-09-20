add_compile_options(-std=c++11)
add_compile_options(-std=c++14)
add_compile_options(-std=c++17)

#Currently we use Yaml based migartion rule to migarte -std=c++22 down to -std=c++17.
#We will implement an implicit migration rule to fix this issue in future.
#add_compile_options(-std=c++22)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++22")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
add_link_options("--add-arch=sm_70")
list(APPEND PMEMD_NVCC_FLAGS --std c++11)

string(APPEND CMAKE_C_FLAGS " -prec_div")
string(APPEND CMAKE_CXX_FLAGS " -parallel")
string(APPEND CMAKE_CXX_FLAGS " -wd188,186,144,913,556,858,597,177,1292,167,279,592,94,2722,3199")
