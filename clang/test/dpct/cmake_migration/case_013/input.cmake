cuda_compile_cubin(cubinfile_1 ${cufile})
cuda_compile_cubin(
  cubinfile_2 
  ${cufile}
)

cuda_compile_cubin(cubinfiles_1 file1.cu file2.cu)
cuda_compile_cubin(cubinfiles_2 
  file1.cu 
  file2.cu)

cuda_compile_cubin(cubinfile_1 file1.cu OPTIONS
  -O3 -Xptxas -v --use_fast_math)
