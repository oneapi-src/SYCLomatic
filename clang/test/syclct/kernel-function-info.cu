// RUN: syclct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/kernel-function-info.sycl.cpp --match-full-lines %s

//CHECK: template<class T>
//CHECK-NEXT: void testTemplateKernel(T *data)
template<class T>
__global__ void testTemplateKernel(T *data)
{
}

//CHECK: void testKernel(void* data)
__global__ void testKernel(void* data)
{
}

template<class T>
void getTemplateFuncAttrs()
{
  //CHECK: sycl_kernel_function_info attrs;
  cudaFuncAttributes attrs;

  //CHECK: (get_kernel_function_info(&attrs, (const void *)testTemplateKernel<T>), 0);
  cudaFuncGetAttributes(&attrs, testTemplateKernel<T>);

  //CHECK: int threadPerBlock = attrs.max_work_group_size;
  int threadPerBlock = attrs.maxThreadsPerBlock;
}

void getFuncAttrs()
{
  //CHECK: sycl_kernel_function_info attrs;
  cudaFuncAttributes attrs;

  //CHECK: (get_kernel_function_info(&attrs, (const void *)testKernel), 0);
  cudaFuncGetAttributes(&attrs, testKernel);

  //CHECK: int threadPerBlock = attrs.max_work_group_size;
  int threadPerBlock = attrs.maxThreadsPerBlock;

}

int main()
{
  getTemplateFuncAttrs<int>();
  getFuncAttrs();
}
