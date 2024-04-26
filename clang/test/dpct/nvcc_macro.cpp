// RUN: dpct --format-range=none --out-root %T/nvcc_macro %s --cuda-include-path="%cuda-path/include" --extra-arg="-xc++" || true
// RUN: FileCheck %s --match-full-lines --input-file %T/nvcc_macro/nvcc_macro.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/nvcc_macro/nvcc_macro.cpp -o %T/nvcc_macro/nvcc_macro.o %}

//      CHECK: #if defined(__NVCC__)
// CHECK-NEXT: #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 8)
// CHECK-NEXT:   #define COMP_NVCC 1
// CHECK-NEXT: #else
// CHECK-NEXT:   #error "Unknown version."
// CHECK-NEXT: #endif
// CHECK-NEXT: #else
// CHECK-NEXT:   #define COMP_NVCC 0
// CHECK-NEXT: #endif
#if defined(__NVCC__)
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 8)
  #define COMP_NVCC 1
#else
  #error "Unknown version."
#endif
#else
  #define COMP_NVCC 0
#endif

int main() {
  return 0;
}
