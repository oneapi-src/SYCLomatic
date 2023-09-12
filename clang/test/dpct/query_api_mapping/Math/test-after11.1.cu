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
// LLMAX-NEXT:   dpct::max(ll1, ll2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=llmin | FileCheck %s -check-prefix=LLMIN
// LLMIN: CUDA API:
// LLMIN-NEXT:   llmin(ll1 /*long long*/, ll2 /*long long*/);
// LLMIN-NEXT: Is migrated to:
// LLMIN-NEXT:   dpct::min(ll1, ll2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ullmax | FileCheck %s -check-prefix=ULLMAX
// ULLMAX: CUDA API:
// ULLMAX-NEXT:   ullmax(ull1 /*unsigned long long*/, ull2 /*unsigned long long*/);
// ULLMAX-NEXT: Is migrated to:
// ULLMAX-NEXT:   dpct::max(ull1, ull2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ullmin | FileCheck %s -check-prefix=ULLMIN
// ULLMIN: CUDA API:
// ULLMIN-NEXT:   ullmin(ull1 /*unsigned long long*/, ull2 /*unsigned long long*/);
// ULLMIN-NEXT: Is migrated to:
// ULLMIN-NEXT:   dpct::min(ull1, ull2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=umax | FileCheck %s -check-prefix=UMAX
// UMAX: CUDA API:
// UMAX-NEXT:   umax(u1 /*unsigned int*/, u2 /*unsigned int*/);
// UMAX-NEXT: Is migrated to:
// UMAX-NEXT:   dpct::max(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=umin | FileCheck %s -check-prefix=UMIN
// UMIN: CUDA API:
// UMIN-NEXT:   umin(u1 /*unsigned int*/, u2 /*unsigned int*/);
// UMIN-NEXT: Is migrated to:
// UMIN-NEXT:   dpct::min(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=max | FileCheck %s -check-prefix=MAX
// MAX: CUDA API:
// MAX-NEXT:   /* 1 */ max(ull /*const unsigned long long int*/, ll /*const long long int*/);
// MAX-NEXT:   /* 2 */ max(ll /*const long long int*/, ull /*const unsigned long long int*/);
// MAX-NEXT:   /* 3 */ max(ull /*const unsigned long long int*/,
// MAX-NEXT:               ull /*const unsigned long long int*/);
// MAX-NEXT:   /* 4 */ max(ll /*const long long int*/, ll /*const long long int*/);
// MAX-NEXT:   /* 5 */ max(ul /*const unsigned long int*/, l /*const long int*/);
// MAX-NEXT:   /* 6 */ max(l /*const long int*/, ul /*const unsigned long int*/);
// MAX-NEXT:   /* 7 */ max(ul /*const unsigned long int*/, ul /*const unsigned long int*/);
// MAX-NEXT:   /* 8 */ max(l /*const long int*/, l /*const long int*/);
// MAX-NEXT:   /* 9 */ max(u /*const unsigned int*/, i /*const int*/);
// MAX-NEXT:   /* 10 */ max(i /*const int*/, u /*const unsigned int*/);
// MAX-NEXT:   /* 11 */ max(u /*const unsigned int*/, u /*const unsigned int*/);
// MAX-NEXT:   /* 12 */ max(i /*const int*/, i /*const int*/);
// MAX-NEXT:   /* 13 */ max(f /*const float*/, f /*const float*/);
// MAX-NEXT:   /* 14 */ max(d /*const double*/, f /*const float*/);
// MAX-NEXT:   /* 15 */ max(f /*const float*/, d /*const double*/);
// MAX-NEXT:   /* 16 */ max(d /*const double*/, d /*const double*/);
// MAX-NEXT: Is migrated to:
// MAX-NEXT:   /* 1 */ dpct::max(ull, ll);
// MAX-NEXT:   /* 2 */ dpct::max(ll, ull);
// MAX-NEXT:   /* 3 */ sycl::max(ull, ull);
// MAX-NEXT:   /* 4 */ sycl::max(ll, ll);
// MAX-NEXT:   /* 5 */ dpct::max(ul, l);
// MAX-NEXT:   /* 6 */ dpct::max(l, ul);
// MAX-NEXT:   /* 7 */ sycl::max(ul, ul);
// MAX-NEXT:   /* 8 */ sycl::max(l, l);
// MAX-NEXT:   /* 9 */ dpct::max(u, i);
// MAX-NEXT:   /* 10 */ dpct::max(i, u);
// MAX-NEXT:   /* 11 */ sycl::max(u, u);
// MAX-NEXT:   /* 12 */ sycl::max(i, i);
// MAX-NEXT:   /* 13 */ sycl::max(f, f);
// MAX-NEXT:   /* 14 */ dpct::max(d, f);
// MAX-NEXT:   /* 15 */ dpct::max(f, d);
// MAX-NEXT:   /* 16 */ sycl::max(d, d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=min | FileCheck %s -check-prefix=MIN
// MIN: CUDA API:
// MIN-NEXT:   /* 1 */ min(ull /*const unsigned long long int*/, ll /*const long long int*/);
// MIN-NEXT:   /* 2 */ min(ll /*const long long int*/, ull /*const unsigned long long int*/);
// MIN-NEXT:   /* 3 */ min(ull /*const unsigned long long int*/,
// MIN-NEXT:               ull /*const unsigned long long int*/);
// MIN-NEXT:   /* 4 */ min(ll /*const long long int*/, ll /*const long long int*/);
// MIN-NEXT:   /* 5 */ min(ul /*const unsigned long int*/, l /*const long int*/);
// MIN-NEXT:   /* 6 */ min(l /*const long int*/, ul /*const unsigned long int*/);
// MIN-NEXT:   /* 7 */ min(ul /*const unsigned long int*/, ul /*const unsigned long int*/);
// MIN-NEXT:   /* 8 */ min(l /*const long int*/, l /*const long int*/);
// MIN-NEXT:   /* 9 */ min(u /*const unsigned int*/, i /*const int*/);
// MIN-NEXT:   /* 10 */ min(i /*const int*/, u /*const unsigned int*/);
// MIN-NEXT:   /* 11 */ min(u /*const unsigned int*/, u /*const unsigned int*/);
// MIN-NEXT:   /* 12 */ min(i /*const int*/, i /*const int*/);
// MIN-NEXT:   /* 13 */ min(f /*const float*/, f /*const float*/);
// MIN-NEXT:   /* 14 */ min(d /*const double*/, f /*const float*/);
// MIN-NEXT:   /* 15 */ min(f /*const float*/, d /*const double*/);
// MIN-NEXT:   /* 16 */ min(d /*const double*/, d /*const double*/);
// MIN-NEXT: Is migrated to:
// MIN-NEXT:   /* 1 */ dpct::min(ull, ll);
// MIN-NEXT:   /* 2 */ dpct::min(ll, ull);
// MIN-NEXT:   /* 3 */ sycl::min(ull, ull);
// MIN-NEXT:   /* 4 */ sycl::min(ll, ll);
// MIN-NEXT:   /* 5 */ dpct::min(ul, l);
// MIN-NEXT:   /* 6 */ dpct::min(l, ul);
// MIN-NEXT:   /* 7 */ sycl::min(ul, ul);
// MIN-NEXT:   /* 8 */ sycl::min(l, l);
// MIN-NEXT:   /* 9 */ dpct::min(u, i);
// MIN-NEXT:   /* 10 */ dpct::min(i, u);
// MIN-NEXT:   /* 11 */ sycl::min(u, u);
// MIN-NEXT:   /* 12 */ sycl::min(i, i);
// MIN-NEXT:   /* 13 */ sycl::min(f, f);
// MIN-NEXT:   /* 14 */ dpct::min(d, f);
// MIN-NEXT:   /* 15 */ dpct::min(f, d);
// MIN-NEXT:   /* 16 */ sycl::min(d, d);
