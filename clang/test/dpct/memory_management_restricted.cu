// FIXME
// UNSUPPORTED: -windows-
// RUN: dpct --usm-level=restricted -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --match-full-lines --input-file %T/memory_management_restricted.dp.cpp %s

#include <cuda_runtime.h>

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {}

#define DATAMACRO 32*32

int main(){
    float **data = NULL;

    //CHECK:  /*
    //CHECK-NEXT:  DPCT1003:0: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT:  */
    //CHECK-NEXT:  checkCudaErrors((*((void **)data) = cl::sycl::malloc_device(DATAMACRO, dpct::get_device_manager().current_device(), dpct::get_default_queue().get_context()), 0));
    checkCudaErrors(cudaMalloc((void **)data, DATAMACRO));
}