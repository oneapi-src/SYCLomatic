// RUN: dpct --format-range=none -out-root %T/fix_internal_error %s --cuda-include-path="%cuda-path/include" > %T/fix_internal_error_output.txt 2>&1
// RUN: grep "dpct internal error" %T/fix_internal_error_output.txt | wc -l > %T/fix_internal_error_wc_output.txt || true
// RUN: FileCheck %S/check_no_internal_error.txt --match-full-lines --input-file %T/fix_internal_error_wc_output.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/fix_internal_error/fix_internal_error.dp.cpp

// Test description:
// This test is to cover un-instantiate member call.
// The 1st instantiated will construct a DeviceFunctionInfo instance.
// When analyze foo in the template decl, tool cannot find the default parameter.
// The analysis is skipped to avoid internal error.
template<class T>
struct TEST {
  __device__ void foo(int a = 0) {}
};

template<class T>
__global__ void bar();

template __global__ void bar<int>();

template<class T>
__global__ void bar() {
  TEST<T> test;
  test.foo();
}

// Test description:
// This test is to cover device function which has template parameter pack.
// The bar1() will instantiate foo() with 0 parameter while bar2() will instantiate foo() with 1 parameter.
// Tool cannot differentiate above 2 cases. The analysis is skipped to avoid internal error.
template <template <bool, typename...> class f, typename... Args> __host__ __device__ bool foo(Args &&...args)
{
  return true;
}

template <bool is_device> struct FOO {
  constexpr bool operator()() { return false; }
};


__device__ __host__ inline bool bar1() { return foo<FOO>(); }

__device__ __host__ inline bool bar2() { return foo<FOO>(false); }


#include <exception>
#include <string>

class MY_EXCEPTION : public std::exception {
public:
  MY_EXCEPTION(uint32_t, std::string){};
};

//CHECK:#define MACRO_DDD(ARG)                                                         \
//CHECK-NEXT:  do {                                                                         \
//CHECK-NEXT:    int e = ARG;                                                       \
//CHECK-NEXT:    if (e != 0) {                                                    \
//CHECK-NEXT:      throw MY_EXCEPTION(uint32_t(__LINE__), "cudaGetErrorString is not supported"/*cudaGetErrorString(e)*/);           \
//CHECK-NEXT:    }                                                                          \
//CHECK-NEXT:  } while (0)

#define MACRO_DDD(ARG)                                                         \
  do {                                                                         \
    cudaError_t e = ARG;                                                       \
    if (e != cudaSuccess) {                                                    \
      throw MY_EXCEPTION(uint32_t(__LINE__), cudaGetErrorString(e));           \
    }                                                                          \
  } while (0)

//CHECK:#define MACRO_CCC() MACRO_DDD(0)

#define MACRO_CCC() MACRO_DDD(cudaGetLastError())

#define MACRO_BBB(ARG)                                                         \
  { MACRO_CCC(); }

#define MACRO_AAA(ARG) ARG();

void foo2() {
  MACRO_AAA([&] {
    MACRO_BBB(1)
    MACRO_BBB(1)
  });
}

#undef MACRO_AAA
#undef MACRO_BBB
#undef MACRO_CCC
#undef MACRO_DDD
