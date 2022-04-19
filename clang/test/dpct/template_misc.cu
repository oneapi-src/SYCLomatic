// RUN: dpct --format-range=none -out-root %T/template_misc %s --cuda-include-path="%cuda-path/include" > %T/output.txt
// RUN: grep "dpct internal error" %T/output.txt | wc -l > %T/wc_output.txt || true
// RUN: FileCheck %s --match-full-lines --input-file %T/wc_output.txt

// CHECK: 0

// Test description:
// This test is to cover un-instantiate member call.
// The 1st instantiated will construct a DeviceFunctionInfo instance.
// When analyze foo in the template decl, tool cannot found the default parameter.
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