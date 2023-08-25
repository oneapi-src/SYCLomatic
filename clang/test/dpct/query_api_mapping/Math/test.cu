/// Half Precision Conversion And Data Movement

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__high2float | FileCheck %s -check-prefix=HIGH2FLOAT
// HIGH2FLOAT: CUDA API:
// HIGH2FLOAT-NEXT:   __high2float(h /*__half2*/);
// HIGH2FLOAT-NEXT: Is migrated to:
// HIGH2FLOAT-NEXT:   h[1];

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__low2float | FileCheck %s -check-prefix=LOW2FLOAT
// LOW2FLOAT: CUDA API:
// LOW2FLOAT-NEXT:   __low2float(h /*__half2*/);
// LOW2FLOAT-NEXT: Is migrated to:
// LOW2FLOAT-NEXT:   h[0];

/// Single Precision Intrinsics

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__expf | FileCheck %s -check-prefix=_EXPF
// _EXPF: CUDA API:
// _EXPF-NEXT:   __expf(f /*float*/);
// _EXPF-NEXT: Is migrated to:
// _EXPF-NEXT:   sycl::native::exp(f);
