// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__assert_fail | FileCheck %s -check-prefix=__ASSERT_FAIL
// __ASSERT_FAIL: CUDA API:
// __ASSERT_FAIL-NEXT:   __assert_fail(msg /*const char **/, file /*const char **/, line /*unsigned*/,
// __ASSERT_FAIL-NEXT:                 func /*const char **/);
// __ASSERT_FAIL-NEXT: Is migrated to:
// __ASSERT_FAIL-NEXT:   assert(0);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__assertfail | FileCheck %s -check-prefix=__ASSERTFAIL
// __ASSERTFAIL: CUDA API:
// __ASSERTFAIL-NEXT:   __assertfail(msg /*const char **/, file /*const char **/, line /*unsigned*/,
// __ASSERTFAIL-NEXT:                func /*const char **/, charSize /*size_t*/);
// __ASSERTFAIL-NEXT: Is migrated to:
// __ASSERTFAIL-NEXT:   assert(0);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCabs | FileCheck %s -check-prefix=CUCABS
// CUCABS: CUDA API:
// CUCABS-NEXT:   cuCabs(c /*cuDoubleComplex*/);
// CUCABS-NEXT: Is migrated to:
// CUCABS-NEXT:   dpct::cabs<double>(c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCabsf | FileCheck %s -check-prefix=CUCABSF
// CUCABSF: CUDA API:
// CUCABSF-NEXT:   cuCabsf(c /*cuFloatComplex*/);
// CUCABSF-NEXT: Is migrated to:
// CUCABSF-NEXT:   dpct::cabs<float>(c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCadd | FileCheck %s -check-prefix=CUCADD
// CUCADD: CUDA API:
// CUCADD-NEXT:   cuCadd(c1 /*cuDoubleComplex*/, c2 /*cuDoubleComplex*/);
// CUCADD-NEXT: Is migrated to:
// CUCADD-NEXT:   c1 + c2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCaddf | FileCheck %s -check-prefix=CUCADDF
// CUCADDF: CUDA API:
// CUCADDF-NEXT:   cuCaddf(c1 /*cuFloatComplex*/, c2 /*cuFloatComplex*/);
// CUCADDF-NEXT: Is migrated to:
// CUCADDF-NEXT:   c1 + c2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCdiv | FileCheck %s -check-prefix=CUCDIV
// CUCDIV: CUDA API:
// CUCDIV-NEXT:   cuCdiv(c1 /*cuDoubleComplex*/, c2 /*cuDoubleComplex*/);
// CUCDIV-NEXT: Is migrated to:
// CUCDIV-NEXT:   dpct::cdiv<double>(c1, c2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCdivf | FileCheck %s -check-prefix=CUCDIVF
// CUCDIVF: CUDA API:
// CUCDIVF-NEXT:   cuCdivf(c1 /*cuFloatComplex*/, c2 /*cuFloatComplex*/);
// CUCDIVF-NEXT: Is migrated to:
// CUCDIVF-NEXT:   dpct::cdiv<float>(c1, c2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCimag | FileCheck %s -check-prefix=CUCIMAG
// CUCIMAG: CUDA API:
// CUCIMAG-NEXT:   cuCimag(c /*cuDoubleComplex*/);
// CUCIMAG-NEXT: Is migrated to:
// CUCIMAG-NEXT:   c.y();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCimagf | FileCheck %s -check-prefix=CUCIMAGF
// CUCIMAGF: CUDA API:
// CUCIMAGF-NEXT:   cuCimagf(c /*cuFloatComplex*/);
// CUCIMAGF-NEXT: Is migrated to:
// CUCIMAGF-NEXT:   c.y();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCmul | FileCheck %s -check-prefix=CUCMUL
// CUCMUL: CUDA API:
// CUCMUL-NEXT:   cuCmul(c1 /*cuDoubleComplex*/, c2 /*cuDoubleComplex*/);
// CUCMUL-NEXT: Is migrated to:
// CUCMUL-NEXT:   dpct::cmul<double>(c1, c2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCmulf | FileCheck %s -check-prefix=CUCMULF
// CUCMULF: CUDA API:
// CUCMULF-NEXT:   cuCmulf(c1 /*cuFloatComplex*/, c2 /*cuFloatComplex*/);
// CUCMULF-NEXT: Is migrated to:
// CUCMULF-NEXT:   dpct::cmul<float>(c1, c2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuComplexDoubleToFloat | FileCheck %s -check-prefix=CUCOMPLEXDOUBLETOFLOAT
// CUCOMPLEXDOUBLETOFLOAT: CUDA API:
// CUCOMPLEXDOUBLETOFLOAT-NEXT:   cuComplexDoubleToFloat(c /*cuDoubleComplex*/);
// CUCOMPLEXDOUBLETOFLOAT-NEXT: Is migrated to:
// CUCOMPLEXDOUBLETOFLOAT-NEXT:   c.convert<float>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuComplexFloatToDouble | FileCheck %s -check-prefix=CUCOMPLEXFLOATTODOUBLE
// CUCOMPLEXFLOATTODOUBLE: CUDA API:
// CUCOMPLEXFLOATTODOUBLE-NEXT:   cuComplexFloatToDouble(c /*cuFloatComplex*/);
// CUCOMPLEXFLOATTODOUBLE-NEXT: Is migrated to:
// CUCOMPLEXFLOATTODOUBLE-NEXT:   c.convert<double>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuConj | FileCheck %s -check-prefix=CUCONJ
// CUCONJ: CUDA API:
// CUCONJ-NEXT:   cuConj(c /*cuDoubleComplex*/);
// CUCONJ-NEXT: Is migrated to:
// CUCONJ-NEXT:   dpct::conj<double>(c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuConjf | FileCheck %s -check-prefix=CUCONJF
// CUCONJF: CUDA API:
// CUCONJF-NEXT:   cuConjf(c /*cuFloatComplex*/);
// CUCONJF-NEXT: Is migrated to:
// CUCONJF-NEXT:   dpct::conj<float>(c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCreal | FileCheck %s -check-prefix=CUCREAL
// CUCREAL: CUDA API:
// CUCREAL-NEXT:   cuCreal(c /*cuDoubleComplex*/);
// CUCREAL-NEXT: Is migrated to:
// CUCREAL-NEXT:   c.x();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCrealf | FileCheck %s -check-prefix=CUCREALF
// CUCREALF: CUDA API:
// CUCREALF-NEXT:   cuCrealf(c /*cuFloatComplex*/);
// CUCREALF-NEXT: Is migrated to:
// CUCREALF-NEXT:   c.x();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCsub | FileCheck %s -check-prefix=CUCSUB
// CUCSUB: CUDA API:
// CUCSUB-NEXT:   cuCsub(c1 /*cuDoubleComplex*/, c2 /*cuDoubleComplex*/);
// CUCSUB-NEXT: Is migrated to:
// CUCSUB-NEXT:   c1 - c2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cuCsubf | FileCheck %s -check-prefix=CUCSUBF
// CUCSUBF: CUDA API:
// CUCSUBF-NEXT:   cuCsubf(c1 /*cuFloatComplex*/, c2 /*cuFloatComplex*/);
// CUCSUBF-NEXT: Is migrated to:
// CUCSUBF-NEXT:   c1 - c2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_cuComplex | FileCheck %s -check-prefix=MAKE_CUCOMPLEX
// MAKE_CUCOMPLEX: CUDA API:
// MAKE_CUCOMPLEX-NEXT:   make_cuComplex(f1 /*float*/, f2 /*float*/);
// MAKE_CUCOMPLEX-NEXT: Is migrated to:
// MAKE_CUCOMPLEX-NEXT:   sycl::float2(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_cuDoubleComplex | FileCheck %s -check-prefix=MAKE_CUDOUBLECOMPLEX
// MAKE_CUDOUBLECOMPLEX: CUDA API:
// MAKE_CUDOUBLECOMPLEX-NEXT:   make_cuDoubleComplex(d1 /*double*/, d2 /*double*/);
// MAKE_CUDOUBLECOMPLEX-NEXT: Is migrated to:
// MAKE_CUDOUBLECOMPLEX-NEXT:   sycl::double2(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=make_cuFloatComplex | FileCheck %s -check-prefix=MAKE_CUFLOATCOMPLEX
// MAKE_CUFLOATCOMPLEX: CUDA API:
// MAKE_CUFLOATCOMPLEX-NEXT:   make_cuFloatComplex(f1 /*float*/, f2 /*float*/);
// MAKE_CUFLOATCOMPLEX-NEXT: Is migrated to:
// MAKE_CUFLOATCOMPLEX-NEXT:   sycl::float2(f1, f2);
