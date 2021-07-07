//===---HelperFileAndFeatureNames.inc---------------------------------===//
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

// dpct.hpp
// features
HELPER_FEATURE_MAP_TO_APINAME(Dpct, "dpct_align_and_inline", "__dpct_align__(n) and __dpct_inline__")
HELPER_FEATURE_MAP_TO_APINAME(Dpct, "dpct_named_lambda", "dpct_kernel_name and dpct_kernel_scalar")
HELPER_FEATURE_MAP_TO_APINAME(Dpct, "dpct_compatibility_temp", "DPCT_COMPATIBILITY_TEMP")
HELPER_FEATURE_MAP_TO_APINAME(Dpct, "dpct_pi_f", "DPCT_PI_F")
HELPER_FEATURE_MAP_TO_APINAME(Dpct, "dpct_pi", "DPCT_PI")

// image.hpp
// features
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_channel_forward_decl", "image_channel")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_wrapper_base_forward_decl", "image_wrapper_base")
HELPER_FEATURE_MAP_TO_APINAME(Image, "create_image_wrapper_detail_forward_decl", "create_image_wrapper")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_creator_forward_decl", "image_creator")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_matrix_p_alias", "image_matrix_p")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_wrapper_base_p_alias", "image_wrapper_base_p")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_accessor_ext_forward_decl", "image_accessor_ext")
HELPER_FEATURE_MAP_TO_APINAME(Image, "create_image_wrapper_detail", "create_image_wrapper")
// sub-features
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_channel_1", "image_channel")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_channel_2", "image_channel")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_channel_3", "image_channel")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_matrix_get_range_T", "image_matrix::get_range")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_matrix_1", "image_matrix")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_matrix_create_image_with_channel", "image_matrix::create_image")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_matrix_2", "image_matrix")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_data_1", "image_data")
HELPER_FEATURE_MAP_TO_APINAME(Image, "sampling_info_1", "sampling_info")
HELPER_FEATURE_MAP_TO_APINAME(Image, "sampling_info_set_addressing_mode", "sampling_info::set")
HELPER_FEATURE_MAP_TO_APINAME(Image, "sampling_info_set_filtering_mode", "sampling_info::set")
HELPER_FEATURE_MAP_TO_APINAME(Image, "sampling_info_set_coordinate_normalization_mode_enum", "sampling_info::set")
HELPER_FEATURE_MAP_TO_APINAME(Image, "sampling_info_set_addressing_mode_filtering_mode_coordinate_normalization_mode", "sampling_info::set")
HELPER_FEATURE_MAP_TO_APINAME(Image, "sampling_info_set_addressing_mode_filtering_mode_is_normalized", "sampling_info::set")
HELPER_FEATURE_MAP_TO_APINAME(Image, "sampling_info_2", "sampling_info")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_wrapper_base_set_addressing_mode", "image_wrapper_base::set")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_wrapper_base_set_filtering_mode", "image_wrapper_base::set")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_wrapper_base_set_coordinate_normalization_mode_enum", "image_wrapper_base::set")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_wrapper_base_set_addressing_mode_filtering_mode_coordinate_normalization_mode", "image_wrapper_base::set")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_wrapper_base_set_addressing_mode_filtering_mode_is_normalized", "image_wrapper_base::set")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_wrapper_base_1", "image_wrapper_base")
HELPER_FEATURE_MAP_TO_APINAME(Image, "image_wrapper_1", "image_wrapper")

// memory.hpp
// features
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
// sub-features
HELPER_FEATURE_MAP_TO_APINAME(Memory, "pitched_data_1", "pitched_data")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "mem_mgr_1", "mem_mgr")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "mem_mgr_2", "mem_mgr")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "device_memory_accessor_t_alias", "device_memory::accessor_t")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "device_memory_value_t_alias", "device_memory::value_t")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "device_memory_dpct_accessor_t_alias", "device_memory::dpct_accessor_t")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "device_memory_1", "device_memory")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "device_memory_init_q", "device_memory::init")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "device_memory_get_ptr_q", "device_memory::get_ptr")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "device_memory_2", "device_memory")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "device_memory_3", "device_memory")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "device_memory_4", "device_memory")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "device_memory_accessor_t_alias_1", "device_memory::accessor_t")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "device_memory_5", "device_memory")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "device_memory_get_access_sp", "device_memory::get_access")
HELPER_FEATURE_MAP_TO_APINAME(Memory, "device_memory_6", "device_memory")

// util.hpp
// features
HELPER_FEATURE_MAP_TO_APINAME(Util, "matrix_mem_copy_T", "matrix_mem_copy")
HELPER_FEATURE_MAP_TO_APINAME(Util, "vectorized_isgreater_T", "vectorized_isgreater")
HELPER_FEATURE_MAP_TO_APINAME(Util, "vectorized_isgreater_unsigned", "vectorized_isgreater")

// device.hpp
// sub-features
HELPER_FEATURE_MAP_TO_APINAME(Device, "device_info_1", "device_info")
HELPER_FEATURE_MAP_TO_APINAME(Device, "device_ext_get_device_info_return_void", "device_ext::get_device_info")
HELPER_FEATURE_MAP_TO_APINAME(Device, "device_ext_get_device_info_return_info", "device_ext::get_device_info")
HELPER_FEATURE_MAP_TO_APINAME(Device, "device_ext_friend_async_dpct_free_decl", "async_dpct_free")
HELPER_FEATURE_MAP_TO_APINAME(Device, "device_ext_1", "device_ext")
HELPER_FEATURE_MAP_TO_APINAME(Device, "device_ext_2", "device_ext")
HELPER_FEATURE_MAP_TO_APINAME(Device, "device_mgr_1", "device_mgr")
HELPER_FEATURE_MAP_TO_APINAME(Device, "device_mgr_2", "device_mgr")
HELPER_FEATURE_MAP_TO_APINAME(Device, "device_mgr_3", "device_mgr")
HELPER_FEATURE_MAP_TO_APINAME(Device, "device_mgr_4", "device_mgr")

// dpl_extras/iterators.h
// features
HELPER_FEATURE_MAP_TO_APINAME(DplExtrasIterators, "using_std_advance", "std::advance")
HELPER_FEATURE_MAP_TO_APINAME(DplExtrasIterators, "using_std_distance", "std::distance")

// dpl_extras/memory.h
// features
HELPER_FEATURE_MAP_TO_APINAME(DplExtrasMemory, "cl_sycl_alias", "namespace sycl")
HELPER_FEATURE_MAP_TO_APINAME(DplExtrasMemory, "device_pointer_forward_decl", "device_pointer")
HELPER_FEATURE_MAP_TO_APINAME(DplExtrasMemory, "device_iterator_forward_decl_device_pointer", "device_pointer")

// dpl_extras/vector.h
// features
HELPER_FEATURE_MAP_TO_APINAME(DplExtrasVector, "cl_sycl_alias", "namespace sycl")
HELPER_FEATURE_MAP_TO_APINAME(DplExtrasVector, "using_std_distance", "std::distance")
HELPER_FEATURE_MAP_TO_APINAME(DplExtrasVector, "using_std_distance", "std::distance")
