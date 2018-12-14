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
DEF_NOTE(IFSTMT_SPECIAL_CASE, 1002, "Special case error handling if-stmt was detected:'%0'. You may need to rewrite this code")
DEF_WARNING(NOERROR_RETURN_COMMA_OP, 1003, "Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code")
DEF_COMMENT(NOERROR_RETURN_COMMA_OP, 1003, "Translated api does not return error code. (*, 0) is inserted. You may want to rewrite this code")
DEF_WARNING(NOTSUPPORTED, 1004, "{0} is not supported in Sycl")
DEF_COMMENT(NOTSUPPORTED, 1004, "{0} is not supported in Sycl")
DEF_WARNING(VERSION_COMMENT, 1005, "The device version is different. You may want to rewrite this code")
DEF_COMMENT(VERSION_COMMENT, 1005, "The device version is different. You may want to rewrite this code")
DEF_WARNING(NOT_SUPPORT_API_INTEGRATEDORNOT, 1006, "SYCL doesn't provide standard API to differentiate between integrated/discrete GPU devices. Consider to re-implement the code which depends on this field")
DEF_COMMENT(NOT_SUPPORT_API_INTEGRATEDORNOT, 1006, "SYCL doesn't provide standard API to differentiate between integrated/discrete GPU devices. Consider to re-implement the code which depends on this field")
