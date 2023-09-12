// RUN: dpct --format-range=none -out-root %T/kernel-function-info %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/kernel-function-info/kernel-function-info.dp.cpp --match-full-lines %s

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
  //CHECK: dpct::kernel_function_info attrs;
  cudaFuncAttributes attrs;

  //CHECK: dpct::get_kernel_function_info(&attrs, (const void *)testTemplateKernel<T>);
  cudaFuncGetAttributes(&attrs, testTemplateKernel<T>);

  //CHECK: int threadPerBlock = attrs.max_work_group_size;
  int threadPerBlock = attrs.maxThreadsPerBlock;
}

void getFuncAttrs()
{
  //CHECK: dpct::kernel_function_info *attrs;
  cudaFuncAttributes *attrs;

  //CHECK: dpct::get_kernel_function_info(attrs, (const void *)testKernel);
  cudaFuncGetAttributes(attrs, testKernel);

  //CHECK: int threadPerBlock = attrs->max_work_group_size;
  int threadPerBlock = attrs->maxThreadsPerBlock;
}

int main()
{
  getTemplateFuncAttrs<int>();
  getFuncAttrs();
}

