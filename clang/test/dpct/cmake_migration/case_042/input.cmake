# simplest case
set_source_files_properties(${src_files} PROPERTIES LANGUAGE CUDA)

# add space test
set_source_files_properties (${src_files} PROPERTIES LANGUAGE CUDA)

# with more properties after LANGUAGE
set_source_files_properties(${src_files} PROPERTIES LANGUAGE CUDA COMPILE_DEFINITIONS SOME_DEFINE)

# add properties on right
set_source_files_properties(${src_files} PROPERTIES COMPILE_FLAGS "--some-flag" LANGUAGE CUDA COMPILE_DEFINITIONS)
