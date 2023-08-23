// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0

/// Half2 Arithmetic Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hcmadd | FileCheck %s -check-prefix=HCMADD
// HCMADD: CUDA API:
// HCMADD-NEXT:   __hcmadd(h1 /*__half2*/, h2 /*__half2*/, h3 /*__half2*/);
// HCMADD-NEXT:   __hcmadd(b1 /*__nv_bfloat162*/, b2 /*__nv_bfloat162*/, b3 /*__nv_bfloat162*/);
// HCMADD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// HCMADD-NEXT:   sycl::ext::intel::math::hcmadd(h1, h2, h3);
// HCMADD-NEXT:   dpct::complex_mul_add(b1, b2, b3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=abs | FileCheck %s -check-prefix=ABS
// ABS: CUDA API:
// ABS-NEXT:   abs(i /*int*/);
// ABS-NEXT: Is migrated to:
// ABS-NEXT:   sycl::abs(i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=llmax | FileCheck %s -check-prefix=LLMAX
// LLMAX: CUDA API:
// LLMAX-NEXT:   llmax(ll1 /*long long*/, ll2 /*long long*/);
// LLMAX-NEXT: Is migrated to:
// LLMAX-NEXT:   sycl::max<long long>(ll1, ll2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=llmin | FileCheck %s -check-prefix=LLMIN
// LLMIN: CUDA API:
// LLMIN-NEXT:   llmin(ll1 /*long long*/, ll2 /*long long*/);
// LLMIN-NEXT: Is migrated to:
// LLMIN-NEXT:   sycl::min<long long>(ll1, ll2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ullmax | FileCheck %s -check-prefix=ULLMAX
// ULLMAX: CUDA API:
// ULLMAX-NEXT:   ullmax(ull1 /*unsigned long long*/, ull2 /*unsigned long long*/);
// ULLMAX-NEXT: Is migrated to:
// ULLMAX-NEXT:   sycl::max<unsigned long long>(ull1, ull2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ullmin | FileCheck %s -check-prefix=ULLMIN
// ULLMIN: CUDA API:
// ULLMIN-NEXT:   ullmin(ull1 /*unsigned long long*/, ull2 /*unsigned long long*/);
// ULLMIN-NEXT: Is migrated to:
// ULLMIN-NEXT:   sycl::min<unsigned long long>(ull1, ull2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=umax | FileCheck %s -check-prefix=UMAX
// UMAX: CUDA API:
// UMAX-NEXT:   umax(u1 /*unsigned int*/, u2 /*unsigned int*/);
// UMAX-NEXT: Is migrated to:
// UMAX-NEXT:   sycl::max<unsigned int>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=umin | FileCheck %s -check-prefix=UMIN
// UMIN: CUDA API:
// UMIN-NEXT:   umin(u1 /*unsigned int*/, u2 /*unsigned int*/);
// UMIN-NEXT: Is migrated to:
// UMIN-NEXT:   sycl::min<unsigned int>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=max | FileCheck %s -check-prefix=MAX
// MAX: CUDA API:
// MAX-NEXT:   max(ull /*const unsigned long long int*/, ll /*const long long int*/);
// MAX-NEXT:   max(ll /*const long long int*/, ull /*const unsigned long long int*/);
// MAX-NEXT:   max(ull /*const unsigned long long int*/,
// MAX-NEXT:       ull /*const unsigned long long int*/);
// MAX-NEXT:   max(ll /*const long long int*/, ll /*const long long int*/);
// MAX-NEXT:   max(ul /*const unsigned long int*/, l /*const long int*/);
// MAX-NEXT:   max(l /*const long int*/, ul /*const unsigned long int*/);
// MAX-NEXT:   max(ul /*const unsigned long int*/, ul /*const unsigned long int*/);
// MAX-NEXT:   max(l /*const long int*/, l /*const long int*/);
// MAX-NEXT:   max(u /*const unsigned int*/, i /*const int*/);
// MAX-NEXT:   max(i /*const int*/, u /*const unsigned int*/);
// MAX-NEXT:   max(u /*const unsigned int*/, u /*const unsigned int*/);
// MAX-NEXT:   max(i /*const int*/, i /*const int*/);
// MAX-NEXT:   max(f /*const float*/, f /*const float*/);
// MAX-NEXT:   max(d /*const double*/, f /*const float*/);
// MAX-NEXT:   max(f /*const float*/, d /*const double*/);
// MAX-NEXT:   max(d /*const double*/, d /*const double*/);
// MAX-NEXT: Is migrated to:
// MAX-NEXT:   dpct::max(ull, ll);
// MAX-NEXT:   dpct::max(ll, ull);
// MAX-NEXT:   sycl::max(ull, ull);
// MAX-NEXT:   sycl::max(ll, ll);
// MAX-NEXT:   dpct::max(ul, l);
// MAX-NEXT:   dpct::max(l, ul);
// MAX-NEXT:   sycl::max(ul, ul);
// MAX-NEXT:   sycl::max(l, l);
// MAX-NEXT:   dpct::max(u, i);
// MAX-NEXT:   dpct::max(i, u);
// MAX-NEXT:   sycl::max(u, u);
// MAX-NEXT:   sycl::max(i, i);
// MAX-NEXT:   sycl::max(f, f);
// MAX-NEXT:   dpct::max(d, f);
// MAX-NEXT:   dpct::max(f, d);
// MAX-NEXT:   sycl::max(d, d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=min | FileCheck %s -check-prefix=MIN
// MIN: CUDA API:
// MIN-NEXT:   min(ull /*const unsigned long long int*/, ll /*const long long int*/);
// MIN-NEXT:   min(ll /*const long long int*/, ull /*const unsigned long long int*/);
// MIN-NEXT:   min(ull /*const unsigned long long int*/,
// MIN-NEXT:       ull /*const unsigned long long int*/);
// MIN-NEXT:   min(ll /*const long long int*/, ll /*const long long int*/);
// MIN-NEXT:   min(ul /*const unsigned long int*/, l /*const long int*/);
// MIN-NEXT:   min(l /*const long int*/, ul /*const unsigned long int*/);
// MIN-NEXT:   min(ul /*const unsigned long int*/, ul /*const unsigned long int*/);
// MIN-NEXT:   min(l /*const long int*/, l /*const long int*/);
// MIN-NEXT:   min(u /*const unsigned int*/, i /*const int*/);
// MIN-NEXT:   min(i /*const int*/, u /*const unsigned int*/);
// MIN-NEXT:   min(u /*const unsigned int*/, u /*const unsigned int*/);
// MIN-NEXT:   min(i /*const int*/, i /*const int*/);
// MIN-NEXT:   min(f /*const float*/, f /*const float*/);
// MIN-NEXT:   min(d /*const double*/, f /*const float*/);
// MIN-NEXT:   min(f /*const float*/, d /*const double*/);
// MIN-NEXT:   min(d /*const double*/, d /*const double*/);
// MIN-NEXT: Is migrated to:
// MIN-NEXT:   dpct::min(ull, ll);
// MIN-NEXT:   dpct::min(ll, ull);
// MIN-NEXT:   sycl::min(ull, ull);
// MIN-NEXT:   sycl::min(ll, ll);
// MIN-NEXT:   dpct::min(ul, l);
// MIN-NEXT:   dpct::min(l, ul);
// MIN-NEXT:   sycl::min(ul, ul);
// MIN-NEXT:   sycl::min(l, l);
// MIN-NEXT:   dpct::min(u, i);
// MIN-NEXT:   dpct::min(i, u);
// MIN-NEXT:   sycl::min(u, u);
// MIN-NEXT:   sycl::min(i, i);
// MIN-NEXT:   sycl::min(f, f);
// MIN-NEXT:   dpct::min(d, f);
// MIN-NEXT:   dpct::min(f, d);
// MIN-NEXT:   sycl::min(d, d);
