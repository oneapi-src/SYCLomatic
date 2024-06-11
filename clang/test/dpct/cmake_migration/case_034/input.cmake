target_compile_features(foo PUBLIC cuda_std_03)

target_compile_features(foo PUBLIC cuda_std_14)

target_compile_features(foo PUBLIC cuda_std_26)

set_property(TARGET cuda_project PROPERTY CUDA_STANDARD 03)

set_property(TARGET cuda_project PROPERTY CUDA_STANDARD 14)

set_property(TARGET cuda_project PROPERTY CUDA_STANDARD 26)

set_target_properties(${TARGETS} PROPERTIES CUDA_STANDARD 03)

set_target_properties(${TARGETS} PROPERTIES CUDA_STANDARD 03 INCLUDE_DIRECTORIES ${INC_DIR})

set_target_properties(${TARGETS} PROPERTIES INCLUDE_DIRECTORIES ${INC_DIR} CUDA_STANDARD 03)

set_target_properties(${TARGETS} PROPERTIES INCLUDE_DIRECTORIES ${INC_DIR} CUDA_STANDARD 26 COMPILE_FLAGS ${TARGET_COMPILE_FLAGS})

set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD_REQUIRED OFF)
