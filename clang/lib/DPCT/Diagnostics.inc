//===---Diagnostics.inc-----------------------------------------------===//
//
// Copyright (C) 2018 - 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
// TODO: Make the format for comments and warnings/errors the same
// The format for arguments is %0 for warnings/errors and {0} for comments.
// A warning, note and error may not have the same ID, but comment can have the same ID with warning or note or error.
// A comment and diagnostics with the same ID have the same index space for format arguments:
// warning may use %0 and comment may use {1} and in that way they are using different parts of parameter pack


// FORMAT: {NAME, ID, MESSAGE}
DEF_WARNING(IFSTMT_NOT_REMOVED, 1000, "Error handling if-stmt was detected but could not be rewritten.")
DEF_COMMENT(IFSTMT_NOT_REMOVED, 1000, "Error handling if-stmt was detected but could not be rewritten.")
DEF_WARNING(STMT_NOT_REMOVED, 1001, "Statement couldn't be removed.")
DEF_COMMENT(STMT_NOT_REMOVED, 1001, "Statement couldn't be removed.")
DEF_WARNING(IFSTMT_SPECIAL_CASE, 1002, "Special case error handling if-stmt was detected. You may need to rewrite this code.")
DEF_COMMENT(IFSTMT_SPECIAL_CASE, 1002, "Special case error handling if-stmt was detected. You may need to rewrite this code.")
DEF_WARNING(NOERROR_RETURN_COMMA_OP, 1003, "Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.")
DEF_COMMENT(NOERROR_RETURN_COMMA_OP, 1003, "Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.")
DEF_WARNING(NOTSUPPORTED, 1004, "Could not generate replacement.")
DEF_COMMENT(NOTSUPPORTED, 1004, "Could not generate replacement.")
DEF_WARNING(VERSION_COMMENT, 1005, "The device version is different. You need to rewrite this code.")
DEF_COMMENT(VERSION_COMMENT, 1005, "The device version is different. You need to rewrite this code.")
DEF_WARNING(NOT_SUPPORT_API_INTEGRATEDORNOT, 1006, "DPC++ does not provide a standard API to differentiate between integrated/ discrete GPU devices.")
DEF_COMMENT(NOT_SUPPORT_API_INTEGRATEDORNOT, 1006, "DPC++ does not provide a standard API to differentiate between integrated/ discrete GPU devices.")
DEF_WARNING(API_NOT_MIGRATED, 1007, "Migration of this CUDA API is not supported by the Intel(R) DPC++ Compatibility Tool.")
DEF_COMMENT(API_NOT_MIGRATED, 1007, "Migration of this CUDA API is not supported by the Intel(R) DPC++ Compatibility Tool.")
DEF_WARNING(API_NOT_MIGRATED_SYCL_UNDEF, 1008, "clock function is not defined in the DPC++. This is a hardware-specific feature. Consult with your hardware vendor to find a replacement.")
DEF_COMMENT(API_NOT_MIGRATED_SYCL_UNDEF, 1008, "clock function is not defined in the DPC++. This is a hardware-specific feature. Consult with your hardware vendor to find a replacement.")
DEF_WARNING(TRNA_WARNING_ERROR_HANDLING_API_COMMENTED, 1009, "SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.")
DEF_COMMENT(TRNA_WARNING_ERROR_HANDLING_API_COMMENTED, 1009, "SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.")
DEF_WARNING(TRNA_WARNING_ERROR_HANDLING_API_REPLACED_0, 1010, "SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.")
DEF_COMMENT(TRNA_WARNING_ERROR_HANDLING_API_REPLACED_0, 1010, "SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.")
DEF_WARNING(TRNA_WARNING_OVERLOADED_API_FOUND, 1011, "The tool detected overloaded operators for built-in vector types, which may conflict with the SYCL 1.2.1 standard operators (see 4.10.2.1 Vec interface). The tool inserted a namespace to avoid the conflict. Use SYCL 1.2.1 standard operators instead.")
DEF_COMMENT(TRNA_WARNING_OVERLOADED_API_FOUND, 1011, "The tool detected overloaded operators for built-in vector types, which may conflict with the SYCL 1.2.1 standard operators (see 4.10.2.1 Vec interface). The tool inserted a namespace to avoid the conflict. Use SYCL 1.2.1 standard operators instead.")
DEF_WARNING(TIME_MEASUREMENT_FOUND, 1012, "Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.")
DEF_COMMENT(TIME_MEASUREMENT_FOUND, 1012, "Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.")
DEF_WARNING(ROUNDING_MODE_UNSUPPORTED, 1013, "The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.")
DEF_COMMENT(ROUNDING_MODE_UNSUPPORTED, 1013, "The rounding mode could not be specified and the generated code may have different precision then the original code. Verify the correctness. SYCL math built-ins rounding mode is aligned with OpenCL C 1.2 standard.")
DEF_WARNING(STREAM_FLAG_PRIORITY_NOT_SUPPORTED, 1014, "The flag/priority options are not supported for SYCL queues; the output parameter(s) are set to 0.")
DEF_COMMENT(STREAM_FLAG_PRIORITY_NOT_SUPPORTED, 1014, "The flag/priority options are not supported for SYCL queues; the output parameter(s) are set to 0.")
DEF_WARNING(NOT_SUPPORTED_PARAMETERS_VALUE, 1016, "The %0 was not migrated, because %1. Rewrite this code manually.")
DEF_COMMENT(NOT_SUPPORTED_PARAMETERS_VALUE, 1016, "The {0} was not migrated, because {1}. Rewrite this code manually.")
DEF_WARNING(MATH_EMULATION, 1017, "The %0 is simulated by %1. You need to check the potential precision and/or performance issues of generated code.")
DEF_COMMENT(MATH_EMULATION, 1017, "The {0} is simulated by {1}. You need to check the potential precision and/or performance issues of generated code.")
DEF_WARNING(POTENTIAL_PERFORMACE_ISSUE, 1018, "The %0 was migrated, but due to %1, the generated code performance may be sub-optimal.")
DEF_COMMENT(POTENTIAL_PERFORMACE_ISSUE, 1018, "The {0} was migrated, but due to {1}, the generated code performance may be sub-optimal.")
DEF_WARNING(PRINTF_FUNC_MIGRATION_WARNING, 1015, "Adjust the code manually.")
DEF_COMMENT(PRINTF_FUNC_MIGRATION_WARNING, 1015, "Adjust the code manually.")
DEF_WARNING(LOCAL_MEM_SIZE, 1019, "local_mem_size in SYCL is not a complete equivalent of sharedMemPerBlock in CUDA. You may need to adjust the code.")
DEF_COMMENT(LOCAL_MEM_SIZE, 1019, "local_mem_size in SYCL is not a complete equivalent of sharedMemPerBlock in CUDA. You may need to adjust the code.")
DEF_WARNING(FUNCTION_CALL_IN_DEVICE, 1020, "Migration of %0, if it is called from __global__ or __device__ function is not supported. You may need to redesign the code to use the host-side %1 instead, which submits this call to the SYCL queue automatically.")
DEF_COMMENT(FUNCTION_CALL_IN_DEVICE, 1020, "Migration of {0} if it is called from __global__ or __device__ function is not supported. You may need to redesign the code to use host-side {1} instead, which will submit this call to SYCL queue automatically.")
DEF_WARNING(HANDLE_IN_DEVICE, 1021, "Migration of cublasHandle_t in __global__ or __device__ function is not supported. You may need to redesign the code.")
DEF_COMMENT(HANDLE_IN_DEVICE, 1021, "Migration of cublasHandle_t in __global__ or __device__ function is not supported. You may need to redesign the code.")
DEF_WARNING(MAX_GRID_SIZE, 1022, "There is no exact match between the maxGridSize and the max_nd_range size. Verify the correctness of the code.")
DEF_COMMENT(MAX_GRID_SIZE, 1022, "There is no exact match between the maxGridSize and the max_nd_range size. Verify the correctness of the code.")
DEF_WARNING(MASK_UNSUPPORTED, 1023, "The DPC++ sub-group does not support mask options for %0.")
DEF_COMMENT(MASK_UNSUPPORTED, 1023, "The DPC++ sub-group does not support mask options for {0}.")
DEF_WARNING(NOERROR_RETURN_ZERO, 1024, "The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.")
DEF_COMMENT(NOERROR_RETURN_ZERO, 1024, "The original code returned the error code that was further consumed by the program logic. This original code was replaced with 0. You may need to rewrite the program logic consuming the error code.")
DEF_WARNING(QUEUE_CREATED_IGNORING_OPTIONS, 1025, "The SYCL queue is created ignoring the flag/priority options.")
DEF_COMMENT(QUEUE_CREATED_IGNORING_OPTIONS, 1025, "The SYCL queue is created ignoring the flag/priority options.")
DEF_WARNING(FUNC_CALL_REMOVED, 1026, "The call to %0 was removed. %1")
DEF_COMMENT(FUNC_CALL_REMOVED, 1026, "The call to {0} was removed. {1}")
DEF_WARNING(FUNC_CALL_REMOVED_0, 1027, "The call to %0 was replaced with 0. %1")
DEF_COMMENT(FUNC_CALL_REMOVED_0, 1027, "The call to {0} was replaced with 0. {1}")
