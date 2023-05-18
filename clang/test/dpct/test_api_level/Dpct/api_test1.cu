// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Dpct/api_test1_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Dpct/api_test1_out/MainSourceFiles.yaml | wc -l > %T/Dpct/api_test1_out/count.txt
// RUN: FileCheck --input-file %T/Dpct/api_test1_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Dpct/api_test1_out

// CHECK: 30

// TEST_FEATURE: Dpct_non_local_include_dependency
// TEST_FEATURE: Dpct_dpct_align_and_inline
// TEST_FEATURE: Dpct_dpct_noinline
// TEST_FEATURE: Dpct_dpct_check_error

class __align__(8) T1 {
    unsigned int l, a;
};

__forceinline__ void foo(){}

__noinline__ cudaError_t foo2(){
  void **buffer;
  size_t size;
  return cudaMalloc(buffer, size);
}

int main() {
  return 0;
}
