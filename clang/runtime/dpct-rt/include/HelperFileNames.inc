//===---HelperFileNames.inc-------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

HELPERFILE(dpct.hpp.inc, Dpct)
HELPERFILE(device.hpp.inc, Device)
HELPERFILE(kernel.hpp.inc, Kernel)
HELPERFILE(memory.hpp.inc, Memory)
HELPERFILE(util.hpp.inc, Util)
HELPERFILE(dpl_utils.hpp.inc, DplUtils)
HELPERFILE(blas_utils.hpp.inc, BlasUtils)
HELPERFILE(image.hpp.inc, Image)
HELPERFILE(atomic.hpp.inc, Atomic)
HELPERFILE(dpl_extras/algorithm.h.inc, DplExtrasAlgorithm)
HELPERFILE(dpl_extras/functional.h.inc, DplExtrasFunctional)
HELPERFILE(dpl_extras/iterators.h.inc, DplExtrasIterators)
HELPERFILE(dpl_extras/memory.h.inc, DplExtrasMemory)
HELPERFILE(dpl_extras/numeric.h.inc, DplExtrasNumeric)
HELPERFILE(dpl_extras/vector.h.inc, DplExtrasVector)

// The message emitted will use the API name. If a feature does not have
// related item in the map, use the feature name in message.
// The 1st parameter is the file enum value.
// The 2nd parameter is the feature name.
// The 3rd parameter is the API name which can be found in source code.
HELPER_FEATURE_MAP_TO_APINAME(Dpct, "dpct_align_and_inline", "__dpct_align__(n) and __dpct_inline__")
HELPER_FEATURE_MAP_TO_APINAME(Dpct, "dpct_named_lambda", "dpct_kernel_name and dpct_kernel_scalar")
HELPER_FEATURE_MAP_TO_APINAME(Dpct, "dpct_compatibility_temp", "DPCT_COMPATIBILITY_TEMP")
HELPER_FEATURE_MAP_TO_APINAME(Dpct, "dpct_pi_f", "DPCT_PI_F")
HELPER_FEATURE_MAP_TO_APINAME(Dpct, "dpct_pi", "DPCT_PI")

HELPER_FEATURE_MAP_TO_APINAME(Image, "image_channel_forward_decl", "image_channel")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_wrapper_base_forward_decl", "image_wrapper_base")
HELPER_FEATURE_MAP_TO_APINAME(Image, "create_image_wrapper_detail_forward_decl", "create_image_wrapper")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_creator_forward_decl", "image_creator")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_matrix_p_alias", "image_matrix_p")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_wrapper_base_p_alias", "image_wrapper_base_p")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_accessor_ext_forward_decl", "image_accessor_ext")
HELPER_FEATURE_MAP_TO_APINAME(Image, "create_image_wrapper_detail", "create_image_wrapper")

HELPER_FEATURE_MAP_TO_APINAME(Memory, "typedef_byte_t", "byte_t")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "typedef_buffer_t", "buffer_t")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "dpct_malloc_detail", "dpct_malloc")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "dpct_malloc_pitch_detail", "dpct_malloc")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "dpct_memset_detail", "dpct_memset")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "dpct_memset_3d_detail", "dpct_memset")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "dpct_memset_2d_detail", "dpct_memset")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "dpct_memcpy_detail", "dpct_memcpy")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "dpct_memcpy_3d_detail", "dpct_memcpy")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "dpct_memcpy_2d_3d_pitch_detail", "dpct_memcpy")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "dpct_memcpy_2d_pitch_detail", "dpct_memcpy")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "get_buffer_T", "get_buffer")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "get_buffer_byte_t", "get_buffer")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "dpct_malloc_3d", "dpct_malloc")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "dpct_malloc_2d", "dpct_malloc")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "dpct_memcpy_2d", "dpct_memcpy")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "async_dpct_memcpy_2d", "async_dpct_memcpy")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "dpct_memcpy_3d", "dpct_memcpy")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "async_dpct_memcpy_3d", "async_dpct_memcpy")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "dpct_memset_2d", "dpct_memset")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "async_dpct_memset_2d", "async_dpct_memset")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "dpct_memset_3d", "dpct_memset")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "async_dpct_memset_3d", "async_dpct_memset")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "dpct_accessor", "accessor")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "global_memory_alias", "global_memory")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "constant_memory_alias", "constant_memory")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "shared_memory_alias", "shared_memory")

HELPER_FEATURE_MAP_TO_APINAME(Util, "matrix_mem_copy_T", "matrix_mem_copy")
HELPER_FEATURE_MAP_TO_APINAME(Util, "vectorized_isgreater_T", "vectorized_isgreater")
HELPER_FEATURE_MAP_TO_APINAME(Util, "vectorized_isgreater_unsigned", "vectorized_isgreater")

HELPER_FEATURE_MAP_TO_APINAME(DplExtrasIterators, "using_std_advance", "std::advance")
HELPER_FEATURE_MAP_TO_APINAME(DplExtrasIterators, "using_std_distance", "std::distance")

HELPER_FEATURE_MAP_TO_APINAME(DplExtrasMemory, "cl_sycl_alias", "namespace sycl")
HELPER_FEATURE_MAP_TO_APINAME(DplExtrasMemory, "device_pointer_forward_decl", "device_pointer")
HELPER_FEATURE_MAP_TO_APINAME(DplExtrasMemory, "device_iterator_forward_decl_device_pointer", "device_pointer")

HELPER_FEATURE_MAP_TO_APINAME(DplExtrasVector, "cl_sycl_alias", "namespace sycl")
HELPER_FEATURE_MAP_TO_APINAME(DplExtrasVector, "using_std_distance", "std::distance")
HELPER_FEATURE_MAP_TO_APINAME(DplExtrasVector, "using_std_distance", "std::distance")