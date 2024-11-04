add_compile_options(-std=c++11)
add_compile_options(-std=c++14)
add_compile_options(-std=c++17)

#Currently we use Yaml based migration rule to migrate -std=c++22 down to -std=c++17.
#We will implement an implicit migration rule to fix this issue in future.
#add_compile_options(-std=c++22)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++22")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
add_link_options("--add-arch=sm_70")
list(APPEND PMEMD_NVCC_FLAGS --std c++11)
