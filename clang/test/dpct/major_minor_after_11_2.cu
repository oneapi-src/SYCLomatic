// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1
// RUN: dpct --format-range=none --out-root %T/major_minor_after_11_2 %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --input-file %T/major_minor_after_11_2/major_minor_after_11_2.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/major_minor_after_11_2/major_minor_after_11_2.dp.cpp -o %T/major_minor_after_11_2/major_minor_after_11_2.dp.o %}

// CHECK: #define DPCT_COMPAT_RT_MAJOR_VERSION {{11|12}}
// CHECK-NEXT: #define DPCT_COMPAT_RT_MINOR_VERSION {{[0-8]}}
// CHECK-NEXT: #include <sycl/sycl.hpp>
// CHECK-NEXT: #include <dpct/dpct.hpp>

// CHECK: #if ((DPCT_COMPAT_RT_MAJOR_VERSION < 11) || (DPCT_COMPAT_RT_MAJOR_VERSION == 11 && DPCT_COMPAT_RT_MINOR_VERSION < 2))
// CHECK-NEXT: float2 f2;
// CHECK-NEXT: #else
// CHECK-NEXT: sycl::float3 f3;
// CHECK-NEXT: #endif
#if ((__CUDACC_VER_MAJOR__ < 11) || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ < 2))
float2 f2;
#else
float3 f3;
#endif
