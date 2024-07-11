// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct -out-root %T/kernel-function-unsupported-cuda-8 %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/kernel-function-unsupported-cuda-8/kernel-function-unsupported-cuda-8.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl %T/kernel-function-unsupported-cuda-8/kernel-function-unsupported-cuda-8.dp.cpp -o %T/kernel-function-unsupported-cuda-8/kernel-function-unsupported-cuda-8.dp.o %}

void foo(void) {
  CUfunction f;
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuFuncSetAttribute was removed because SYCL currently does not support setting kernel function attributes
  // CHECK-NEXT: */
  cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 1024);

  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuFuncSetAttribute was removed because SYCL currently does not support setting kernel function attributes
  // CHECK-NEXT: */
  // CHECK-NEXT: int result = 0;
  CUresult result = cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 1024);
  if (result == CUDA_SUCCESS) {;}
}
