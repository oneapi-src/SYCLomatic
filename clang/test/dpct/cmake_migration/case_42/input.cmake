# simplest case
set_source_files_properties(${src_files} PROPERTIES LANGUAGE CUDA)

# with more properties after LANGUAGE
set_source_files_properties(${src_files} PROPERTIES LANGUAGE CUDA COMPILE_DEFINITIONS SOME_DEFINE)

set_source_files_properties(${src_files} PROPERTIES COMPILE_FLAGS "--some-flag" LANGUAGE CUDA COMPILE_DEFINITIONS)

