// TODO: Make the format for comments and warnings/errors the same
// The format for arguments is %0 for warnings/errors and {0} for comments.
// A warning, note and error may not have the same ID, but comment can have the same ID with warning or note or error.
// A comment and diagnostics with the same ID have the same index space for format arguments:
// warning may use %0 and comment may use {1} and in that way they are using different parts of parameter pack


// FORMAT: {NAME, ID, MESSAGE}
DEF_WARNING(IFSTMT_NOT_REMOVED, 1000, "Error handling if-stmt was detected but couldn't be rewritten, see details in the resulting file comments")
DEF_COMMENT(IFSTMT_NOT_REMOVED, 1000, "Error handling if-stmt was detected but couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>")
DEF_WARNING(STMT_NOT_REMOVED, 1001, "Statement couldn't be removed. See details in the resulting file comments")
DEF_COMMENT(STMT_NOT_REMOVED, 1001, "Below statement couldn't be removed from error handling if-stmt. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>")
DEF_WARNING(IFSTMT_SPECIAL_CASE, 1002, "Special case error handling if-stmt was detected. It couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>")
DEF_COMMENT(IFSTMT_SPECIAL_CASE, 1002, "Special case error handling if-stmt was detected. It couldn't be rewritten. SYCL error handling is based on exceptions, so you might need to rewrite this code. More details: <Error handling article link placeholder>")
DEF_WARNING(NOERROR_RETURN_COMMA_OP, 1003, "Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code")
DEF_COMMENT(NOERROR_RETURN_COMMA_OP, 1003, "Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code")
DEF_WARNING(NOTSUPPORTED, 1004, "%0 is not supported in Sycl")
DEF_COMMENT(NOTSUPPORTED, 1004, "{0} is not supported in Sycl")
DEF_WARNING(VERSION_COMMENT, 1005, "The device version is different. You may want to rewrite this code")
DEF_COMMENT(VERSION_COMMENT, 1005, "The device version is different. You may want to rewrite this code")
DEF_WARNING(NOT_SUPPORT_API_INTEGRATEDORNOT, 1006, "SYCL doesn't provide standard API to differentiate between integrated/discrete GPU devices. Consider to re-implement the code which depends on this field")
DEF_COMMENT(NOT_SUPPORT_API_INTEGRATEDORNOT, 1006, "SYCL doesn't provide standard API to differentiate between integrated/discrete GPU devices. Consider to re-implement the code which depends on this field")
DEF_COMMENT(API_NOT_MIGRATED, 1007, "{0}: not support API, need manual porting. ")
DEF_WARNING(API_NOT_MIGRATED_SYCL_UNDEF, 1008, "Function clock is not defined in the SYCL specification. This is a hardware-specific feature. Consider consulting with hardware vendor to find a replacement.")
DEF_COMMENT(API_NOT_MIGRATED_SYCL_UNDEF, 1008, "Function clock is not defined in the SYCL specification. This is a hardware-specific feature. Consider consulting with hardware vendor to find a replacement.")
DEF_WARNING(TRNA_WARNING_ERROR_HANDLING_API_COMMENTED, 1009, "SYCL API uses exceptions to report errors and doesn't use the error codes. Hence, {0} is commented out and a warning string is inserted. You may need to rewrite this code.")
DEF_COMMENT(TRNA_WARNING_ERROR_HANDLING_API_COMMENTED, 1009, "SYCL API uses exceptions to report errors and doesn't use the error codes. Hence, {0} is commented out and a warning string is inserted. You may need to rewrite this code.")
DEF_WARNING(TRNA_WARNING_ERROR_HANDLING_API_REPLACED_0, 1010, "SYCL API uses exceptions to report errors and doesn't use the error codes. Hence, {0} was replaced with 0. You may need to rewrite this code.")
DEF_COMMENT(TRNA_WARNING_ERROR_HANDLING_API_REPLACED_0, 1010, "SYCL API uses exceptions to report errors and doesn't use the error codes. Hence, {0} was replaced with 0. You may need to rewrite this code.")
DEF_WARNING(TRNA_WARNING_OVERLOADED_API_FOUND, 1011, "The tool detected overloaded operators for built-in vector types, which may conflict with SYCL 1.2.1 standard operators (see 4.10.2.1 Vec interface). The tool inserted namespace to avoid the conflict. Consider using SYCL 1.2.1 standard operators instead.")
DEF_COMMENT(TRNA_WARNING_OVERLOADED_API_FOUND, 1011, "The tool detected overloaded operators for built-in vector types, which may conflict with SYCL 1.2.1 standard operators (see 4.10.2.1 Vec interface). The tool inserted namespace to avoid the conflict. Consider using SYCL 1.2.1 standard operators instead.")
DEF_WARNING(TIME_MEASUREMENT_FOUND, 1012, "The tool detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. Depending on your goals you may want to change the way the time is measured.")
DEF_COMMENT(TIME_MEASUREMENT_FOUND, 1012, "The tool detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. Depending on your goals you may want to change the way the time is measured.")
DEF_WARNING(ROUNDING_MODE_UNSUPPORTED, 1013, "The rounding mode of %0 is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.")
DEF_COMMENT(ROUNDING_MODE_UNSUPPORTED, 1013, "The rounding mode of {0} is not defined in SYCL 1.2.1 standard. Please, verify the correctness of generated code.")
DEF_WARNING(STREAM_FLAG_PRIORITY_NOT_SUPPORTED, 1014, "Flag and priority options are not supported in SYCL queue. You may want to rewrite this code.")
DEF_COMMENT(STREAM_FLAG_PRIORITY_NOT_SUPPORTED, 1014, "Flag and priority options are not supported in SYCL queue. You may want to rewrite this code.")
DEF_WARNING(CALLBACK_FOR_QUEUE_NOT_SUPPORTED, 1015, "DPC++ doesn't have callback mechanism for queues. cudaStreamAddCallback was replaced with a blocking wait and host function call. Consider changing the algorithm to avoid waiting and calling of the function on the host.")
DEF_COMMENT(CALLBACK_FOR_QUEUE_NOT_SUPPORTED, 1015, "DPC++ doesn't have callback mechanism for queues. cudaStreamAddCallback was replaced with a blocking wait and host function call. Consider changing the algorithm to avoid waiting and calling of the function on the host.")
DEF_WARNING(NOTSUPPORTEDCASE, 1016, "parameter(s) value of %1 in %0 is not supported in Sycl. You may need to migrate this code manually.")
DEF_COMMENT(NOTSUPPORTEDCASE, 1016, "parameter(s) value of {1} in {0} is not supported in Sycl. You may need to migrate this code manually.")
DEF_WARNING(MATH_SIMULATION, 1017, "%0 is simulated by %1 in DPC++. Please check the potential precision and/or performance issues of generated code.")
DEF_COMMENT(MATH_SIMULATION, 1017, "{0} is simulated by {1} in DPC++. Please check the potential precision and/or performance issues of generated code.")
