add_executable(foo src_file.cu)

add_executable(foo src_file1.cu src_file2.cu src_file3.cu)

add_executable(foo ${GENERATED_CONFIG_H} ${A_FILES} ${B_FILES} src_file3.cu)
