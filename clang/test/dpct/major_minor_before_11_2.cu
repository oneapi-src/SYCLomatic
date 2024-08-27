// UNSUPPORTED: cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8, cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6
// UNSUPPORTED: v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v11.8, v12.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6
// RUN: dpct --format-range=none --out-root %T/major_minor_before_11_2 %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/major_minor_before_11_2/major_minor_before_11_2.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/major_minor_before_11_2/major_minor_before_11_2.dp.cpp -o %T/major_minor_before_11_2/major_minor_before_11_2.dp.o %}

// CHECK: #define DPCT_COMPAT_RT_MAJOR_VERSION {{8|9|10|11}}
// CHECK-NEXT: #define DPCT_COMPAT_RT_MINOR_VERSION {{[0-2]}}
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>

// CHECK: #if ((DPCT_COMPAT_RT_MAJOR_VERSION < 11) || (DPCT_COMPAT_RT_MAJOR_VERSION == 11 && DPCT_COMPAT_RT_MINOR_VERSION < 2))
// CHECK-NEXT: sycl::float2 f2;
// CHECK-NEXT: #else
// CHECK-NEXT: float3 f3;
// CHECK-NEXT: #endif
#if ((__CUDACC_VER_MAJOR__ < 11) || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ < 2))
float2 f2;
#else
float3 f3;
#endif
