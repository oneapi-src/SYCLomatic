// RUN: dpct --format-range=none   --use-custom-helper=api -out-root %T/Memory/api_test46_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Memory/api_test46_out/MainSourceFiles.yaml | wc -l > %T/Memory/api_test46_out/count.txt
// RUN: FileCheck --input-file %T/Memory/api_test46_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Memory/api_test46_out

// CHECK: 20
// TEST_FEATURE: Memory_usm_device_allocator_alias

#include <thrust/device_allocator.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/iterator/iterator_traits.h>

namespace cusp
{
template <typename T, typename MemorySpace>
struct default_memory_allocator
    : thrust::device_malloc_allocator<T> 
{};
}

int main(int argc, char *argv[]) {
  return 0;
}
