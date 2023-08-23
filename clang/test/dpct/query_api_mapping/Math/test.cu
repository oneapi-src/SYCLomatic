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

/// Single Precision Mathematical Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=acosf | FileCheck %s -check-prefix=ACOSF
// ACOSF: CUDA API:
// ACOSF-NEXT:   acosf(f /*float*/);
// ACOSF-NEXT: Is migrated to:
// ACOSF-NEXT:   sycl::acos(f);
// ACOSF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=acoshf | FileCheck %s -check-prefix=ACOSHF
// ACOSHF: CUDA API:
// ACOSHF-NEXT:   acoshf(f /*float*/);
// ACOSHF-NEXT: Is migrated to:
// ACOSHF-NEXT:   sycl::acosh(f);
// ACOSHF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=asinf | FileCheck %s -check-prefix=ASINF
// ASINF: CUDA API:
// ASINF-NEXT:   asinf(f /*float*/);
// ASINF-NEXT: Is migrated to:
// ASINF-NEXT:   sycl::asin(f);
// ASINF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=asinhf | FileCheck %s -check-prefix=ASINHF
// ASINHF: CUDA API:
// ASINHF-NEXT:   asinhf(f /*float*/);
// ASINHF-NEXT: Is migrated to:
// ASINHF-NEXT:   sycl::asinh(f);
// ASINHF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atan2f | FileCheck %s -check-prefix=ATAN2F
// ATAN2F: CUDA API:
// ATAN2F-NEXT:   atan2f(f1 /*float*/, f2 /*float*/);
// ATAN2F-NEXT: Is migrated to:
// ATAN2F-NEXT:   sycl::atan2(f1, f2);
// ATAN2F-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atanf | FileCheck %s -check-prefix=ATANF
// ATANF: CUDA API:
// ATANF-NEXT:   atanf(f /*float*/);
// ATANF-NEXT: Is migrated to:
// ATANF-NEXT:   sycl::atan(f);
// ATANF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atanhf | FileCheck %s -check-prefix=ATANHF
// ATANHF: CUDA API:
// ATANHF-NEXT:   atanhf(f /*float*/);
// ATANHF-NEXT: Is migrated to:
// ATANHF-NEXT:   sycl::atanh(f);
// ATANHF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cbrtf | FileCheck %s -check-prefix=CBRTF
// CBRTF: CUDA API:
// CBRTF-NEXT:   cbrtf(f /*float*/);
// CBRTF-NEXT: Is migrated to:
// CBRTF-NEXT:   sycl::cbrt(f);
// CBRTF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ceilf | FileCheck %s -check-prefix=CEILF
// CEILF: CUDA API:
// CEILF-NEXT:   ceilf(f /*float*/);
// CEILF-NEXT: Is migrated to:
// CEILF-NEXT:   sycl::ceil(f);
// CEILF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=copysignf | FileCheck %s -check-prefix=COPYSIGNF
// COPYSIGNF: CUDA API:
// COPYSIGNF-NEXT:   copysignf(f1 /*float*/, f2 /*float*/);
// COPYSIGNF-NEXT: Is migrated to:
// COPYSIGNF-NEXT:   sycl::copysign(f1, f2);
// COPYSIGNF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cosf | FileCheck %s -check-prefix=COSF
// COSF: CUDA API:
// COSF-NEXT:   cosf(f /*float*/);
// COSF-NEXT: Is migrated to:
// COSF-NEXT:   sycl::cos(f);
// COSF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=coshf | FileCheck %s -check-prefix=COSHF
// COSHF: CUDA API:
// COSHF-NEXT:   coshf(f /*float*/);
// COSHF-NEXT: Is migrated to:
// COSHF-NEXT:   sycl::cosh(f);
// COSHF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cospif | FileCheck %s -check-prefix=COSPIF
// COSPIF: CUDA API:
// COSPIF-NEXT:   cospif(f /*float*/);
// COSPIF-NEXT: Is migrated to:
// COSPIF-NEXT:   sycl::cospi(f);
// COSPIF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cyl_bessel_i0f | FileCheck %s -check-prefix=CYL_BESSEL_I0F
// CYL_BESSEL_I0F: CUDA API:
// CYL_BESSEL_I0F-NEXT:   cyl_bessel_i0f(f /*float*/);
// CYL_BESSEL_I0F-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// CYL_BESSEL_I0F-NEXT:   sycl::ext::intel::math::cyl_bessel_i0(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cyl_bessel_i1f | FileCheck %s -check-prefix=CYL_BESSEL_I1F
// CYL_BESSEL_I1F: CUDA API:
// CYL_BESSEL_I1F-NEXT:   cyl_bessel_i1f(f /*float*/);
// CYL_BESSEL_I1F-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// CYL_BESSEL_I1F-NEXT:   sycl::ext::intel::math::cyl_bessel_i1(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=erfcf | FileCheck %s -check-prefix=ERFCF
// ERFCF: CUDA API:
// ERFCF-NEXT:   erfcf(f /*float*/);
// ERFCF-NEXT: Is migrated to:
// ERFCF-NEXT:   sycl::erfc(f);
// ERFCF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=erfcinvf | FileCheck %s -check-prefix=ERFCINVF
// ERFCINVF: CUDA API:
// ERFCINVF-NEXT:   erfcinvf(f /*float*/);
// ERFCINVF-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// ERFCINVF-NEXT:   sycl::ext::intel::math::erfcinv(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=erfcxf | FileCheck %s -check-prefix=ERFCXF
// ERFCXF: CUDA API:
// ERFCXF-NEXT:   erfcxf(f /*float*/);
// ERFCXF-NEXT: Is migrated to:
// ERFCXF-NEXT:   sycl::exp(f*f)*sycl::erfc(f);
// ERFCXF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=erff | FileCheck %s -check-prefix=ERFF
// ERFF: CUDA API:
// ERFF-NEXT:   erff(f /*float*/);
// ERFF-NEXT: Is migrated to:
// ERFF-NEXT:   sycl::erf(f);
// ERFF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=erfinvf | FileCheck %s -check-prefix=ERFINVF
// ERFINVF: CUDA API:
// ERFINVF-NEXT:   erfinvf(f /*float*/);
// ERFINVF-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// ERFINVF-NEXT:   sycl::ext::intel::math::erfinv(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=exp10f | FileCheck %s -check-prefix=EXP10F
// EXP10F: CUDA API:
// EXP10F-NEXT:   exp10f(f /*float*/);
// EXP10F-NEXT: Is migrated to:
// EXP10F-NEXT:   sycl::exp10(f);
// EXP10F-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=exp2f | FileCheck %s -check-prefix=EXP2F
// EXP2F: CUDA API:
// EXP2F-NEXT:   exp2f(f /*float*/);
// EXP2F-NEXT: Is migrated to:
// EXP2F-NEXT:   sycl::exp2(f);
// EXP2F-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=expf | FileCheck %s -check-prefix=EXPF
// EXPF: CUDA API:
// EXPF-NEXT:   expf(f /*float*/);
// EXPF-NEXT: Is migrated to:
// EXPF-NEXT:   sycl::native::exp(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=expm1f | FileCheck %s -check-prefix=EXPM1F
// EXPM1F: CUDA API:
// EXPM1F-NEXT:   expm1f(f /*float*/);
// EXPM1F-NEXT: Is migrated to:
// EXPM1F-NEXT:   sycl::expm1(f);
// EXPM1F-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fabsf | FileCheck %s -check-prefix=FABSF
// FABSF: CUDA API:
// FABSF-NEXT:   fabsf(f /*float*/);
// FABSF-NEXT: Is migrated to:
// FABSF-NEXT:   sycl::fabs(f);
// FABSF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fdimf | FileCheck %s -check-prefix=FDIMF
// FDIMF: CUDA API:
// FDIMF-NEXT:   fdimf(f1 /*float*/, f2 /*float*/);
// FDIMF-NEXT: Is migrated to:
// FDIMF-NEXT:   sycl::fdim(f1, f2);
// FDIMF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fdividef | FileCheck %s -check-prefix=FDIVIDEF
// FDIVIDEF: CUDA API:
// FDIVIDEF-NEXT:   fdividef(f1 /*float*/, f2 /*float*/);
// FDIVIDEF-NEXT: Is migrated to:
// FDIVIDEF-NEXT:   f1 / f2;
// FDIVIDEF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=floorf | FileCheck %s -check-prefix=FLOORF
// FLOORF: CUDA API:
// FLOORF-NEXT:   floorf(f /*float*/);
// FLOORF-NEXT: Is migrated to:
// FLOORF-NEXT:   sycl::floor(f);
// FLOORF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fmaf | FileCheck %s -check-prefix=FMAF
// FMAF: CUDA API:
// FMAF-NEXT:   fmaf(f1 /*float*/, f2 /*float*/, f3 /*float*/);
// FMAF-NEXT: Is migrated to:
// FMAF-NEXT:   sycl::fma(f1, f2, f3);
// FMAF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fmaxf | FileCheck %s -check-prefix=FMAXF
// FMAXF: CUDA API:
// FMAXF-NEXT:   fmaxf(f1 /*float*/, f2 /*float*/);
// FMAXF-NEXT: Is migrated to:
// FMAXF-NEXT:   sycl::fmax(f1, f2);
// FMAXF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fminf | FileCheck %s -check-prefix=FMINF
// FMINF: CUDA API:
// FMINF-NEXT:   fminf(f1 /*float*/, f2 /*float*/);
// FMINF-NEXT: Is migrated to:
// FMINF-NEXT:   sycl::fmin(f1, f2);
// FMINF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fmodf | FileCheck %s -check-prefix=FMODF
// FMODF: CUDA API:
// FMODF-NEXT:   fmodf(f1 /*float*/, f2 /*float*/);
// FMODF-NEXT: Is migrated to:
// FMODF-NEXT:   sycl::fmod(f1, f2);
// FMODF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=frexpf | FileCheck %s -check-prefix=FREXPF
// FREXPF: CUDA API:
// FREXPF-NEXT:   frexpf(f /*float*/, pi /*int **/);
// FREXPF-NEXT: Is migrated to:
// FREXPF-NEXT:   sycl::frexp(f, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes, int>(pi));
// FREXPF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hypotf | FileCheck %s -check-prefix=HYPOTF
// HYPOTF: CUDA API:
// HYPOTF-NEXT:   hypotf(f1 /*float*/, f2 /*float*/);
// HYPOTF-NEXT: Is migrated to:
// HYPOTF-NEXT:   sycl::hypot(f1, f2);
// HYPOTF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ilogbf | FileCheck %s -check-prefix=ILOGBF
// ILOGBF: CUDA API:
// ILOGBF-NEXT:   ilogbf(f /*float*/);
// ILOGBF-NEXT: Is migrated to:
// ILOGBF-NEXT:   sycl::ilogb(f);
// ILOGBF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=isfinite | FileCheck %s -check-prefix=ISFINITE
// ISFINITE: CUDA API:
// ISFINITE-NEXT:   isfinite(f /*float*/);
// ISFINITE-NEXT:   isfinite(d /*double*/);
// ISFINITE-NEXT: Is migrated to:
// ISFINITE-NEXT:   sycl::isfinite(f);
// ISFINITE-NEXT:   sycl::isfinite(d);
// ISFINITE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=isinf | FileCheck %s -check-prefix=ISINF
// ISINF: CUDA API:
// ISINF-NEXT:   isinf(f /*float*/);
// ISINF-NEXT:   isinf(d /*double*/);
// ISINF-NEXT: Is migrated to:
// ISINF-NEXT:   sycl::isinf(f);
// ISINF-NEXT:   sycl::isinf(d);
// ISINF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=isnan | FileCheck %s -check-prefix=ISNAN
// ISNAN: CUDA API:
// ISNAN-NEXT:   isnan(f /*float*/);
// ISNAN-NEXT:   isnan(d /*double*/);
// ISNAN-NEXT: Is migrated to:
// ISNAN-NEXT:   sycl::isnan(f);
// ISNAN-NEXT:   sycl::isnan(d);
// ISNAN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=j0f | FileCheck %s -check-prefix=J0F
// J0F: CUDA API:
// J0F-NEXT:   j0f(f /*float*/);
// J0F-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// J0F-NEXT:   sycl::ext::intel::math::j0(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=j1f | FileCheck %s -check-prefix=J1F
// J1F: CUDA API:
// J1F-NEXT:   j1f(f /*float*/);
// J1F-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// J1F-NEXT:   sycl::ext::intel::math::j1(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ldexpf | FileCheck %s -check-prefix=LDEXPF
// LDEXPF: CUDA API:
// LDEXPF-NEXT:   ldexpf(f /*float*/, i /*int*/);
// LDEXPF-NEXT: Is migrated to:
// LDEXPF-NEXT:   sycl::ldexp(f, i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=lgammaf | FileCheck %s -check-prefix=LGAMMAF
// LGAMMAF: CUDA API:
// LGAMMAF-NEXT:   lgammaf(f /*float*/);
// LGAMMAF-NEXT: Is migrated to:
// LGAMMAF-NEXT:   sycl::lgamma(f);
// LGAMMAF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=llrintf | FileCheck %s -check-prefix=LLRINTF
// LLRINTF: CUDA API:
// LLRINTF-NEXT:   llrintf(f /*float*/);
// LLRINTF-NEXT: Is migrated to:
// LLRINTF-NEXT:   sycl::rint(f);
// LLRINTF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=llroundf | FileCheck %s -check-prefix=LLROUNDF
// LLROUNDF: CUDA API:
// LLROUNDF-NEXT:   llroundf(f /*float*/);
// LLROUNDF-NEXT: Is migrated to:
// LLROUNDF-NEXT:   sycl::round(f);
// LLROUNDF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=log10f | FileCheck %s -check-prefix=LOG10F
// LOG10F: CUDA API:
// LOG10F-NEXT:   log10f(f /*float*/);
// LOG10F-NEXT: Is migrated to:
// LOG10F-NEXT:   sycl::log10(f);
// LOG10F-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=log1pf | FileCheck %s -check-prefix=LOG1PF
// LOG1PF: CUDA API:
// LOG1PF-NEXT:   log1pf(f /*float*/);
// LOG1PF-NEXT: Is migrated to:
// LOG1PF-NEXT:   sycl::log1p(f);
// LOG1PF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=log2f | FileCheck %s -check-prefix=LOG2F
// LOG2F: CUDA API:
// LOG2F-NEXT:   log2f(f /*float*/);
// LOG2F-NEXT: Is migrated to:
// LOG2F-NEXT:   sycl::log2(f);
// LOG2F-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=logbf | FileCheck %s -check-prefix=LOGBF
// LOGBF: CUDA API:
// LOGBF-NEXT:   logbf(f /*float*/);
// LOGBF-NEXT: Is migrated to:
// LOGBF-NEXT:   sycl::logb(f);
// LOGBF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=logf | FileCheck %s -check-prefix=LOGF
// LOGF: CUDA API:
// LOGF-NEXT:   logf(f /*float*/);
// LOGF-NEXT: Is migrated to:
// LOGF-NEXT:   sycl::log(f);
// LOGF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=lrintf | FileCheck %s -check-prefix=LRINTF
// LRINTF: CUDA API:
// LRINTF-NEXT:   lrintf(f /*float*/);
// LRINTF-NEXT: Is migrated to:
// LRINTF-NEXT:   sycl::rint(f);
// LRINTF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=lroundf | FileCheck %s -check-prefix=LROUNDF
// LROUNDF: CUDA API:
// LROUNDF-NEXT:   lroundf(f /*float*/);
// LROUNDF-NEXT: Is migrated to:
// LROUNDF-NEXT:   sycl::round(f);
// LROUNDF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=modff | FileCheck %s -check-prefix=MODFF
// MODFF: CUDA API:
// MODFF-NEXT:   modff(f /*float*/, pf /*float **/);
// MODFF-NEXT: Is migrated to:
// MODFF-NEXT:   sycl::modf(f, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes, float>(pf));
// MODFF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nanf | FileCheck %s -check-prefix=NANF
// NANF: CUDA API:
// NANF-NEXT:   nanf(pc /*const char **/);
// NANF-NEXT: Is migrated to:
// NANF-NEXT:   sycl::nan(0u);
// NANF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nearbyintf | FileCheck %s -check-prefix=NEARBYINTF
// NEARBYINTF: CUDA API:
// NEARBYINTF-NEXT:   nearbyintf(f /*float*/);
// NEARBYINTF-NEXT: Is migrated to:
// NEARBYINTF-NEXT:   sycl::floor(f + 0.5);
// NEARBYINTF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nextafterf | FileCheck %s -check-prefix=NEXTAFTERF
// NEXTAFTERF: CUDA API:
// NEXTAFTERF-NEXT:   nextafterf(f1 /*float*/, f2 /*float*/);
// NEXTAFTERF-NEXT: Is migrated to:
// NEXTAFTERF-NEXT:   sycl::nextafter(f1, f2);
// NEXTAFTERF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=normcdff | FileCheck %s -check-prefix=NORMCDFF
// NORMCDFF: CUDA API:
// NORMCDFF-NEXT:   normcdff(f /*float*/);
// NORMCDFF-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// NORMCDFF-NEXT:   sycl::ext::intel::math::cdfnorm(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=normcdfinvf | FileCheck %s -check-prefix=NORMCDFINVF
// NORMCDFINVF: CUDA API:
// NORMCDFINVF-NEXT:   normcdfinvf(f /*float*/);
// NORMCDFINVF-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// NORMCDFINVF-NEXT:   sycl::ext::intel::math::cdfnorminv(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=norm3df | FileCheck %s -check-prefix=NORM3DF
// NORM3DF: CUDA API:
// NORM3DF-NEXT:   norm3df(f1 /*float*/, f2 /*float*/, f3 /*float*/);
// NORM3DF-NEXT: Is migrated to:
// NORM3DF-NEXT:   sycl::length(sycl::float3(f1, f2, f3));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=norm4df | FileCheck %s -check-prefix=NORM4DF
// NORM4DF: CUDA API:
// NORM4DF-NEXT:   norm4df(f1 /*float*/, f2 /*float*/, f3 /*float*/, f4 /*float*/);
// NORM4DF-NEXT: Is migrated to:
// NORM4DF-NEXT:   sycl::length(sycl::float4(f1, f2, f3, f4));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=normf | FileCheck %s -check-prefix=NORMF
// NORMF: CUDA API:
// NORMF-NEXT:   normf(i /*int*/, f /*const float **/);
// NORMF-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// NORMF-NEXT:   sycl::ext::intel::math::norm(i, f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=powf | FileCheck %s -check-prefix=POWF
// POWF: CUDA API:
// POWF-NEXT:   powf(f1 /*float*/, f2 /*float*/);
// POWF-NEXT: Is migrated to:
// POWF-NEXT:   sycl::pow<float>(f1, f2);
// POWF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rcbrtf | FileCheck %s -check-prefix=RCBRTF
// RCBRTF: CUDA API:
// RCBRTF-NEXT:   rcbrtf(f /*float*/);
// RCBRTF-NEXT: Is migrated to:
// RCBRTF-NEXT:   sycl::native::recip(sycl::cbrt<float>(f));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=remainderf | FileCheck %s -check-prefix=REMAINDERF
// REMAINDERF: CUDA API:
// REMAINDERF-NEXT:   remainderf(f1 /*float*/, f2 /*float*/);
// REMAINDERF-NEXT: Is migrated to:
// REMAINDERF-NEXT:   sycl::remainder(f1, f2);
// REMAINDERF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=remquof | FileCheck %s -check-prefix=REMQUOF
// REMQUOF: CUDA API:
// REMQUOF-NEXT:   remquof(f1 /*float*/, f2 /*float*/, pi /*int **/);
// REMQUOF-NEXT: Is migrated to:
// REMQUOF-NEXT:   sycl::remquo(f1, f2, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes, int>(pi));
// REMQUOF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rhypotf | FileCheck %s -check-prefix=RHYPOTF
// RHYPOTF: CUDA API:
// RHYPOTF-NEXT:   rhypotf(f1 /*float*/, f2 /*float*/);
// RHYPOTF-NEXT: Is migrated to:
// RHYPOTF-NEXT:   1 / sycl::hypot(f1, f2);
// RHYPOTF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rintf | FileCheck %s -check-prefix=RINTF
// RINTF: CUDA API:
// RINTF-NEXT:   rintf(f /*float*/);
// RINTF-NEXT: Is migrated to:
// RINTF-NEXT:   sycl::rint(f);
// RINTF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rnorm3df | FileCheck %s -check-prefix=RNORM3DF
// RNORM3DF: CUDA API:
// RNORM3DF-NEXT:   rnorm3df(f1 /*float*/, f2 /*float*/, f3 /*float*/);
// RNORM3DF-NEXT: Is migrated to:
// RNORM3DF-NEXT:   sycl::native::recip(sycl::length(sycl::float3(f1, f2, f3)));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rnorm4df | FileCheck %s -check-prefix=RNORM4DF
// RNORM4DF: CUDA API:
// RNORM4DF-NEXT:   rnorm4df(f1 /*float*/, f2 /*float*/, f3 /*float*/, f4 /*float*/);
// RNORM4DF-NEXT: Is migrated to:
// RNORM4DF-NEXT:   sycl::native::recip(sycl::length(sycl::float4(f1, f2, f3, f4)));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rnormf | FileCheck %s -check-prefix=RNORMF
// RNORMF: CUDA API:
// RNORMF-NEXT:   rnormf(i /*int*/, f /*const float **/);
// RNORMF-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// RNORMF-NEXT:   sycl::ext::intel::math::rnorm(i, f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=roundf | FileCheck %s -check-prefix=ROUNDF
// ROUNDF: CUDA API:
// ROUNDF-NEXT:   roundf(f /*float*/);
// ROUNDF-NEXT: Is migrated to:
// ROUNDF-NEXT:   sycl::round(f);
// ROUNDF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rsqrtf | FileCheck %s -check-prefix=RSQRTF
// RSQRTF: CUDA API:
// RSQRTF-NEXT:   rsqrtf(f /*float*/);
// RSQRTF-NEXT: Is migrated to:
// RSQRTF-NEXT:   sycl::rsqrt(f);
// RSQRTF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=scalblnf | FileCheck %s -check-prefix=SCALBLNF
// SCALBLNF: CUDA API:
// SCALBLNF-NEXT:   scalblnf(f /*float*/, l /*long int*/);
// SCALBLNF-NEXT: Is migrated to:
// SCALBLNF-NEXT:   f*(2<<l);
// SCALBLNF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=scalbnf | FileCheck %s -check-prefix=SCALBNF
// SCALBNF: CUDA API:
// SCALBNF-NEXT:   scalbnf(f /*float*/, i /*int*/);
// SCALBNF-NEXT: Is migrated to:
// SCALBNF-NEXT:   f*(2<<i);
// SCALBNF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=signbit | FileCheck %s -check-prefix=SIGNBIT
// SIGNBIT: CUDA API:
// SIGNBIT-NEXT:   signbit(f /*float*/);
// SIGNBIT-NEXT:   signbit(d /*double*/);
// SIGNBIT-NEXT: Is migrated to:
// SIGNBIT-NEXT:   sycl::signbit(f);
// SIGNBIT-NEXT:   sycl::signbit(d);
// SIGNBIT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sincosf | FileCheck %s -check-prefix=SINCOSF
// SINCOSF: CUDA API:
// SINCOSF-NEXT:   sincosf(f /*float*/, pf1 /*float **/, pf2 /*float **/);
// SINCOSF-NEXT: Is migrated to:
// SINCOSF-NEXT:   *pf1 = sycl::sincos(f, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes, float>(pf2));
// SINCOSF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sincospif | FileCheck %s -check-prefix=SINCOSPIF
// SINCOSPIF: CUDA API:
// SINCOSPIF-NEXT:   sincospif(f /*float*/, pf1 /*float **/, pf2 /*float **/);
// SINCOSPIF-NEXT: Is migrated to:
// SINCOSPIF-NEXT:   *(pf1) = sycl::sincos(f * DPCT_PI_F, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes, float>(pf2));
// SINCOSPIF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sinf | FileCheck %s -check-prefix=SINF
// SINF: CUDA API:
// SINF-NEXT:   sinf(f /*float*/);
// SINF-NEXT: Is migrated to:
// SINF-NEXT:   sycl::sin(f);
// SINF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sinhf | FileCheck %s -check-prefix=SINHF
// SINHF: CUDA API:
// SINHF-NEXT:   sinhf(f /*float*/);
// SINHF-NEXT: Is migrated to:
// SINHF-NEXT:   sycl::sinh(f);
// SINHF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sinpif | FileCheck %s -check-prefix=SINPIF
// SINPIF: CUDA API:
// SINPIF-NEXT:   sinpif(f /*float*/);
// SINPIF-NEXT: Is migrated to:
// SINPIF-NEXT:   sycl::sinpi(f);
// SINPIF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sqrtf | FileCheck %s -check-prefix=SQRTF
// SQRTF: CUDA API:
// SQRTF-NEXT:   sqrtf(f /*float*/);
// SQRTF-NEXT: Is migrated to:
// SQRTF-NEXT:   sycl::sqrt(f);
// SQRTF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tanf | FileCheck %s -check-prefix=TANF
// TANF: CUDA API:
// TANF-NEXT:   tanf(f /*float*/);
// TANF-NEXT: Is migrated to:
// TANF-NEXT:   sycl::tan(f);
// TANF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tanhf | FileCheck %s -check-prefix=TANHF
// TANHF: CUDA API:
// TANHF-NEXT:   tanhf(f /*float*/);
// TANHF-NEXT: Is migrated to:
// TANHF-NEXT:   sycl::tanh(f);
// TANHF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tgammaf | FileCheck %s -check-prefix=TGAMMAF
// TGAMMAF: CUDA API:
// TGAMMAF-NEXT:   tgammaf(f /*float*/);
// TGAMMAF-NEXT: Is migrated to:
// TGAMMAF-NEXT:   sycl::tgamma(f);
// TGAMMAF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=truncf | FileCheck %s -check-prefix=TRUNCF
// TRUNCF: CUDA API:
// TRUNCF-NEXT:   truncf(f /*float*/);
// TRUNCF-NEXT: Is migrated to:
// TRUNCF-NEXT:   sycl::trunc(f);
// TRUNCF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=y0f | FileCheck %s -check-prefix=Y0F
// Y0F: CUDA API:
// Y0F-NEXT:   y0f(f /*float*/);
// Y0F-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// Y0F-NEXT:   sycl::ext::intel::math::y0(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=y1f | FileCheck %s -check-prefix=Y1F
// Y1F: CUDA API:
// Y1F-NEXT:   y1f(f /*float*/);
// Y1F-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// Y1F-NEXT:   sycl::ext::intel::math::y1(f);

/// Double Precision Mathematical Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=acos | FileCheck %s -check-prefix=ACOS
// ACOS: CUDA API:
// ACOS-NEXT:   acos(d /*double*/);
// ACOS-NEXT: Is migrated to:
// ACOS-NEXT:   sycl::acos(d);
// ACOS-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=acosh | FileCheck %s -check-prefix=ACOSH
// ACOSH: CUDA API:
// ACOSH-NEXT:   acosh(d /*double*/);
// ACOSH-NEXT: Is migrated to:
// ACOSH-NEXT:   sycl::acosh(d);
// ACOSH-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=asin | FileCheck %s -check-prefix=ASIN
// ASIN: CUDA API:
// ASIN-NEXT:   asin(d /*double*/);
// ASIN-NEXT: Is migrated to:
// ASIN-NEXT:   sycl::asin(d);
// ASIN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=asinh | FileCheck %s -check-prefix=ASINH
// ASINH: CUDA API:
// ASINH-NEXT:   asinh(d /*double*/);
// ASINH-NEXT: Is migrated to:
// ASINH-NEXT:   sycl::asinh(d);
// ASINH-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atan | FileCheck %s -check-prefix=ATAN
// ATAN: CUDA API:
// ATAN-NEXT:   atan(d /*double*/);
// ATAN-NEXT: Is migrated to:
// ATAN-NEXT:   sycl::atan(d);
// ATAN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atan2 | FileCheck %s -check-prefix=ATAN2
// ATAN2: CUDA API:
// ATAN2-NEXT:   atan2(d1 /*double*/, d2 /*double*/);
// ATAN2-NEXT: Is migrated to:
// ATAN2-NEXT:   sycl::atan2(d1, d2);
// ATAN2-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atanh | FileCheck %s -check-prefix=ATANH
// ATANH: CUDA API:
// ATANH-NEXT:   atanh(d /*double*/);
// ATANH-NEXT: Is migrated to:
// ATANH-NEXT:   sycl::atanh(d);
// ATANH-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cbrt | FileCheck %s -check-prefix=CBRT
// CBRT: CUDA API:
// CBRT-NEXT:   cbrt(d /*double*/);
// CBRT-NEXT: Is migrated to:
// CBRT-NEXT:   sycl::cbrt(d);
// CBRT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ceil | FileCheck %s -check-prefix=CEIL
// CEIL: CUDA API:
// CEIL-NEXT:   ceil(d /*double*/);
// CEIL-NEXT: Is migrated to:
// CEIL-NEXT:   sycl::ceil(d);
// CEIL-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=copysign | FileCheck %s -check-prefix=COPYSIGN
// COPYSIGN: CUDA API:
// COPYSIGN-NEXT:   copysign(d1 /*double*/, d2 /*double*/);
// COPYSIGN-NEXT: Is migrated to:
// COPYSIGN-NEXT:   sycl::copysign(d1, d2);
// COPYSIGN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cos | FileCheck %s -check-prefix=COS
// COS: CUDA API:
// COS-NEXT:   cos(d /*double*/);
// COS-NEXT: Is migrated to:
// COS-NEXT:   sycl::cos(d);
// COS-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cosh | FileCheck %s -check-prefix=COSH
// COSH: CUDA API:
// COSH-NEXT:   cosh(d /*double*/);
// COSH-NEXT: Is migrated to:
// COSH-NEXT:   sycl::cosh(d);
// COSH-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cospi | FileCheck %s -check-prefix=COSPI
// COSPI: CUDA API:
// COSPI-NEXT:   cospi(d /*double*/);
// COSPI-NEXT: Is migrated to:
// COSPI-NEXT:   sycl::cospi(d);
// COSPI-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cyl_bessel_i0 | FileCheck %s -check-prefix=CYL_BESSEL_I0
// CYL_BESSEL_I0: CUDA API:
// CYL_BESSEL_I0-NEXT:   cyl_bessel_i0(d /*double*/);
// CYL_BESSEL_I0-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// CYL_BESSEL_I0-NEXT:   sycl::ext::intel::math::cyl_bessel_i0(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cyl_bessel_i1 | FileCheck %s -check-prefix=CYL_BESSEL_I1
// CYL_BESSEL_I1: CUDA API:
// CYL_BESSEL_I1-NEXT:   cyl_bessel_i1(d /*double*/);
// CYL_BESSEL_I1-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// CYL_BESSEL_I1-NEXT:   sycl::ext::intel::math::cyl_bessel_i1(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=erf | FileCheck %s -check-prefix=ERF
// ERF: CUDA API:
// ERF-NEXT:   erf(d /*double*/);
// ERF-NEXT: Is migrated to:
// ERF-NEXT:   sycl::erf(d);
// ERF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=erfc | FileCheck %s -check-prefix=ERFC
// ERFC: CUDA API:
// ERFC-NEXT:   erfc(d /*double*/);
// ERFC-NEXT: Is migrated to:
// ERFC-NEXT:   sycl::erfc(d);
// ERFC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=erfcinv | FileCheck %s -check-prefix=ERFCINV
// ERFCINV: CUDA API:
// ERFCINV-NEXT:   erfcinv(d /*double*/);
// ERFCINV-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// ERFCINV-NEXT:   sycl::ext::intel::math::erfcinv(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=erfcx | FileCheck %s -check-prefix=ERFCX
// ERFCX: CUDA API:
// ERFCX-NEXT:   erfcx(d /*double*/);
// ERFCX-NEXT: Is migrated to:
// ERFCX-NEXT:   sycl::exp(d*d)*sycl::erfc(d);
// ERFCX-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=erfinv | FileCheck %s -check-prefix=ERFINV
// ERFINV: CUDA API:
// ERFINV-NEXT:   erfinv(d /*double*/);
// ERFINV-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// ERFINV-NEXT:   sycl::ext::intel::math::erfinv(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=exp | FileCheck %s -check-prefix=EXP
// EXP: CUDA API:
// EXP-NEXT:   exp(d /*double*/);
// EXP-NEXT: Is migrated to:
// EXP-NEXT:   sycl::exp(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=exp10 | FileCheck %s -check-prefix=EXP10
// EXP10: CUDA API:
// EXP10-NEXT:   exp10(d /*double*/);
// EXP10-NEXT: Is migrated to:
// EXP10-NEXT:   sycl::exp10(d);
// EXP10-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=exp2 | FileCheck %s -check-prefix=EXP2
// EXP2: CUDA API:
// EXP2-NEXT:   exp2(d /*double*/);
// EXP2-NEXT: Is migrated to:
// EXP2-NEXT:   sycl::exp2(d);
// EXP2-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=expm1 | FileCheck %s -check-prefix=EXPM1
// EXPM1: CUDA API:
// EXPM1-NEXT:   expm1(d /*double*/);
// EXPM1-NEXT: Is migrated to:
// EXPM1-NEXT:   sycl::expm1(d);
// EXPM1-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fabs | FileCheck %s -check-prefix=FABS
// FABS: CUDA API:
// FABS-NEXT:   fabs(d /*double*/);
// FABS-NEXT: Is migrated to:
// FABS-NEXT:   sycl::fabs(d);
// FABS-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fdim | FileCheck %s -check-prefix=FDIM
// FDIM: CUDA API:
// FDIM-NEXT:   fdim(d1 /*double*/, d2 /*double*/);
// FDIM-NEXT: Is migrated to:
// FDIM-NEXT:   sycl::fdim(d1, d2);
// FDIM-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=floor | FileCheck %s -check-prefix=FLOOR
// FLOOR: CUDA API:
// FLOOR-NEXT:   floor(d /*double*/);
// FLOOR-NEXT: Is migrated to:
// FLOOR-NEXT:   sycl::floor(d);
// FLOOR-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fma | FileCheck %s -check-prefix=FMA
// FMA: CUDA API:
// FMA-NEXT:   fma(d1 /*double*/, d2 /*double*/, d3 /*double*/);
// FMA-NEXT: Is migrated to:
// FMA-NEXT:   sycl::fma(d1, d2, d3);
// FMA-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fmax | FileCheck %s -check-prefix=FMAX
// FMAX: CUDA API:
// FMAX-NEXT:   fmax(d1 /*double*/, d2 /*double*/);
// FMAX-NEXT: Is migrated to:
// FMAX-NEXT:   sycl::fmax(d1, d2);
// FMAX-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fmin | FileCheck %s -check-prefix=FMIN
// FMIN: CUDA API:
// FMIN-NEXT:   fmin(d1 /*double*/, d2 /*double*/);
// FMIN-NEXT: Is migrated to:
// FMIN-NEXT:   sycl::fmin(d1, d2);
// FMIN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fmod | FileCheck %s -check-prefix=FMOD
// FMOD: CUDA API:
// FMOD-NEXT:   fmod(d1 /*double*/, d2 /*double*/);
// FMOD-NEXT: Is migrated to:
// FMOD-NEXT:   sycl::fmod(d1, d2);
// FMOD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=frexp | FileCheck %s -check-prefix=FREXP
// FREXP: CUDA API:
// FREXP-NEXT:   frexp(d /*double*/, pi /*int **/);
// FREXP-NEXT: Is migrated to:
// FREXP-NEXT:   sycl::frexp(d, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes, int>(pi));
// FREXP-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hypot | FileCheck %s -check-prefix=HYPOT
// HYPOT: CUDA API:
// HYPOT-NEXT:   hypot(d1 /*double*/, d2 /*double*/);
// HYPOT-NEXT: Is migrated to:
// HYPOT-NEXT:   sycl::hypot(d1, d2);
// HYPOT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ilogb | FileCheck %s -check-prefix=ILOGB
// ILOGB: CUDA API:
// ILOGB-NEXT:   ilogb(d /*double*/);
// ILOGB-NEXT: Is migrated to:
// ILOGB-NEXT:   sycl::ilogb(d);
// ILOGB-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=j0 | FileCheck %s -check-prefix=J0
// J0: CUDA API:
// J0-NEXT:   j0(d /*double*/);
// J0-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// J0-NEXT:   sycl::ext::intel::math::j0(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=j1 | FileCheck %s -check-prefix=J1
// J1: CUDA API:
// J1-NEXT:   j1(d /*double*/);
// J1-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// J1-NEXT:   sycl::ext::intel::math::j1(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ldexp | FileCheck %s -check-prefix=LDEXP
// LDEXP: CUDA API:
// LDEXP-NEXT:   ldexp(d /*double*/, i /*int*/);
// LDEXP-NEXT: Is migrated to:
// LDEXP-NEXT:   sycl::ldexp(d, i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=lgamma | FileCheck %s -check-prefix=LGAMMA
// LGAMMA: CUDA API:
// LGAMMA-NEXT:   lgamma(d /*double*/);
// LGAMMA-NEXT: Is migrated to:
// LGAMMA-NEXT:   sycl::lgamma(d);
// LGAMMA-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=llrint | FileCheck %s -check-prefix=LLRINT
// LLRINT: CUDA API:
// LLRINT-NEXT:   llrint(d /*double*/);
// LLRINT-NEXT: Is migrated to:
// LLRINT-NEXT:   sycl::rint(d);
// LLRINT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=llround | FileCheck %s -check-prefix=LLROUND
// LLROUND: CUDA API:
// LLROUND-NEXT:   llround(d /*double*/);
// LLROUND-NEXT: Is migrated to:
// LLROUND-NEXT:   sycl::round(d);
// LLROUND-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=log | FileCheck %s -check-prefix=LOG
// LOG: CUDA API:
// LOG-NEXT:   log(d /*double*/);
// LOG-NEXT: Is migrated to:
// LOG-NEXT:   sycl::log(d);
// LOG-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=log10 | FileCheck %s -check-prefix=LOG10
// LOG10: CUDA API:
// LOG10-NEXT:   log10(d /*double*/);
// LOG10-NEXT: Is migrated to:
// LOG10-NEXT:   sycl::log10(d);
// LOG10-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=log1p | FileCheck %s -check-prefix=LOG1P
// LOG1P: CUDA API:
// LOG1P-NEXT:   log1p(d /*double*/);
// LOG1P-NEXT: Is migrated to:
// LOG1P-NEXT:   sycl::log1p(d);
// LOG1P-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=log2 | FileCheck %s -check-prefix=LOG2
// LOG2: CUDA API:
// LOG2-NEXT:   log2(d /*double*/);
// LOG2-NEXT: Is migrated to:
// LOG2-NEXT:   sycl::log2(d);
// LOG2-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=logb | FileCheck %s -check-prefix=LOGB
// LOGB: CUDA API:
// LOGB-NEXT:   logb(d /*double*/);
// LOGB-NEXT: Is migrated to:
// LOGB-NEXT:   sycl::logb(d);
// LOGB-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=lrint | FileCheck %s -check-prefix=LRINT
// LRINT: CUDA API:
// LRINT-NEXT:   lrint(d /*double*/);
// LRINT-NEXT: Is migrated to:
// LRINT-NEXT:   sycl::rint(d);
// LRINT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=lround | FileCheck %s -check-prefix=LROUND
// LROUND: CUDA API:
// LROUND-NEXT:   lround(d /*double*/);
// LROUND-NEXT: Is migrated to:
// LROUND-NEXT:   sycl::round(d);
// LROUND-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=modf | FileCheck %s -check-prefix=MODF
// MODF: CUDA API:
// MODF-NEXT:   modf(d /*double*/, pd /*double **/);
// MODF-NEXT: Is migrated to:
// MODF-NEXT:   sycl::modf(d, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes, double>(pd));
// MODF-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nan | FileCheck %s -check-prefix=NAN
// NAN: CUDA API:
// NAN-NEXT:   nan(pc /*const char **/);
// NAN-NEXT: Is migrated to:
// NAN-NEXT:   sycl::nan(0u);
// NAN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nearbyint | FileCheck %s -check-prefix=NEARBYINT
// NEARBYINT: CUDA API:
// NEARBYINT-NEXT:   nearbyint(d /*double*/);
// NEARBYINT-NEXT: Is migrated to:
// NEARBYINT-NEXT:   sycl::floor(d + 0.5);
// NEARBYINT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nextafter | FileCheck %s -check-prefix=NEXTAFTER
// NEXTAFTER: CUDA API:
// NEXTAFTER-NEXT:   nextafter(d1 /*double*/, d2 /*double*/);
// NEXTAFTER-NEXT: Is migrated to:
// NEXTAFTER-NEXT:   sycl::nextafter(d1, d2);
// NEXTAFTER-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=norm | FileCheck %s -check-prefix=NORM
// NORM: CUDA API:
// NORM-NEXT:   norm(i /*int*/, d /*const double **/);
// NORM-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// NORM-NEXT:   sycl::ext::intel::math::norm(i, d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=norm3d | FileCheck %s -check-prefix=NORM3D
// NORM3D: CUDA API:
// NORM3D-NEXT:   norm3d(d1 /*double*/, d2 /*double*/, d3 /*double*/);
// NORM3D-NEXT: Is migrated to:
// NORM3D-NEXT:   sycl::length(sycl::double3(d1, d2, d3));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=norm4d | FileCheck %s -check-prefix=NORM4D
// NORM4D: CUDA API:
// NORM4D-NEXT:   norm4d(d1 /*double*/, d2 /*double*/, d3 /*double*/, d4 /*double*/);
// NORM4D-NEXT: Is migrated to:
// NORM4D-NEXT:   sycl::length(sycl::double4(d1, d2, d3, d4));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=normcdf | FileCheck %s -check-prefix=NORMCDF
// NORMCDF: CUDA API:
// NORMCDF-NEXT:   normcdf(d /*double*/);
// NORMCDF-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// NORMCDF-NEXT:   sycl::ext::intel::math::cdfnorm(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=normcdfinv | FileCheck %s -check-prefix=NORMCDFINV
// NORMCDFINV: CUDA API:
// NORMCDFINV-NEXT:   normcdfinv(d /*double*/);
// NORMCDFINV-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// NORMCDFINV-NEXT:   sycl::ext::intel::math::cdfnorminv(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=pow | FileCheck %s -check-prefix=POW
// POW: CUDA API:
// POW-NEXT:   pow(d1 /*double*/, d2 /*double*/);
// POW-NEXT: Is migrated to:
// POW-NEXT:   sycl::pow<double>(d1, d2);
// POW-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rcbrt | FileCheck %s -check-prefix=RCBRT
// RCBRT: CUDA API:
// RCBRT-NEXT:   rcbrt(d /*double*/);
// RCBRT-NEXT: Is migrated to:
// RCBRT-NEXT:   1 / sycl::cbrt<double>(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=remainder | FileCheck %s -check-prefix=REMAINDER
// REMAINDER: CUDA API:
// REMAINDER-NEXT:   remainder(d1 /*double*/, d2 /*double*/);
// REMAINDER-NEXT: Is migrated to:
// REMAINDER-NEXT:   sycl::remainder(d1, d2);
// REMAINDER-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=remquo | FileCheck %s -check-prefix=REMQUO
// REMQUO: CUDA API:
// REMQUO-NEXT:   remquo(d1 /*double*/, d2 /*double*/, pi /*int **/);
// REMQUO-NEXT: Is migrated to:
// REMQUO-NEXT:   sycl::remquo(d1, d2, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes, int>(pi));
// REMQUO-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rhypot | FileCheck %s -check-prefix=RHYPOT
// RHYPOT: CUDA API:
// RHYPOT-NEXT:   rhypot(d1 /*double*/, d2 /*double*/);
// RHYPOT-NEXT: Is migrated to:
// RHYPOT-NEXT:   1 / sycl::hypot(d1, d2);
// RHYPOT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rint | FileCheck %s -check-prefix=RINT
// RINT: CUDA API:
// RINT-NEXT:   rint(d /*double*/);
// RINT-NEXT: Is migrated to:
// RINT-NEXT:   sycl::rint(d);
// RINT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rnorm | FileCheck %s -check-prefix=RNORM
// RNORM: CUDA API:
// RNORM-NEXT:   rnorm(i /*int*/, d /*const double **/);
// RNORM-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// RNORM-NEXT:   sycl::ext::intel::math::rnorm(i, d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rnorm3d | FileCheck %s -check-prefix=RNORM3D
// RNORM3D: CUDA API:
// RNORM3D-NEXT:   rnorm3d(d1 /*double*/, d2 /*double*/, d3 /*double*/);
// RNORM3D-NEXT: Is migrated to:
// RNORM3D-NEXT:   1 / sycl::length(sycl::double3(d1, d2, d3));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rnorm4d | FileCheck %s -check-prefix=RNORM4D
// RNORM4D: CUDA API:
// RNORM4D-NEXT:   rnorm4d(d1 /*double*/, d2 /*double*/, d3 /*double*/, d4 /*double*/);
// RNORM4D-NEXT: Is migrated to:
// RNORM4D-NEXT:   1 / sycl::length(sycl::double4(d1, d2, d3, d4));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=round | FileCheck %s -check-prefix=ROUND
// ROUND: CUDA API:
// ROUND-NEXT:   round(d /*double*/);
// ROUND-NEXT: Is migrated to:
// ROUND-NEXT:   sycl::round(d);
// ROUND-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rsqrt | FileCheck %s -check-prefix=RSQRT
// RSQRT: CUDA API:
// RSQRT-NEXT:   rsqrt(d /*double*/);
// RSQRT-NEXT: Is migrated to:
// RSQRT-NEXT:   sycl::rsqrt(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=scalbln | FileCheck %s -check-prefix=SCALBLN
// SCALBLN: CUDA API:
// SCALBLN-NEXT:   scalbln(d /*double*/, l /*long int*/);
// SCALBLN-NEXT: Is migrated to:
// SCALBLN-NEXT:   d*(2<<l);
// SCALBLN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=scalbn | FileCheck %s -check-prefix=SCALBN
// SCALBN: CUDA API:
// SCALBN-NEXT:   scalbn(d /*double*/, i /*int*/);
// SCALBN-NEXT: Is migrated to:
// SCALBN-NEXT:   d*(2<<i);
// SCALBN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sin | FileCheck %s -check-prefix=SIN
// SIN: CUDA API:
// SIN-NEXT:   sin(d /*double*/);
// SIN-NEXT: Is migrated to:
// SIN-NEXT:   sycl::sin(d);
// SIN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sincos | FileCheck %s -check-prefix=SINCOS
// SINCOS: CUDA API:
// SINCOS-NEXT:   sincos(d /*double*/, pd1 /*double **/, pd2 /*double **/);
// SINCOS-NEXT: Is migrated to:
// SINCOS-NEXT:   *pd1 = sycl::sincos(d, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes, double>(pd2));
// SINCOS-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sincospi | FileCheck %s -check-prefix=SINCOSPI
// SINCOSPI: CUDA API:
// SINCOSPI-NEXT:   sincospi(d /*double*/, pd1 /*double **/, pd2 /*double **/);
// SINCOSPI-NEXT: Is migrated to:
// SINCOSPI-NEXT:   *(pd1) = sycl::sincos(d * DPCT_PI, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes, double>(pd2));
// SINCOSPI-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sinh | FileCheck %s -check-prefix=SINH
// SINH: CUDA API:
// SINH-NEXT:   sinh(d /*double*/);
// SINH-NEXT: Is migrated to:
// SINH-NEXT:   sycl::sinh(d);
// SINH-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sinpi | FileCheck %s -check-prefix=SINPI
// SINPI: CUDA API:
// SINPI-NEXT:   sinpi(d /*double*/);
// SINPI-NEXT: Is migrated to:
// SINPI-NEXT:   sycl::sinpi(d);
// SINPI-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sqrt | FileCheck %s -check-prefix=SQRT
// SQRT: CUDA API:
// SQRT-NEXT:   sqrt(d /*double*/);
// SQRT-NEXT: Is migrated to:
// SQRT-NEXT:   sycl::sqrt(d);
// SQRT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tan | FileCheck %s -check-prefix=TAN
// TAN: CUDA API:
// TAN-NEXT:   tan(d /*double*/);
// TAN-NEXT: Is migrated to:
// TAN-NEXT:   sycl::tan(d);
// TAN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tanh | FileCheck %s -check-prefix=TANH
// TANH: CUDA API:
// TANH-NEXT:   tanh(d /*double*/);
// TANH-NEXT: Is migrated to:
// TANH-NEXT:   sycl::tanh(d);
// TANH-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tgamma | FileCheck %s -check-prefix=TGAMMA
// TGAMMA: CUDA API:
// TGAMMA-NEXT:   tgamma(d /*double*/);
// TGAMMA-NEXT: Is migrated to:
// TGAMMA-NEXT:   sycl::tgamma(d);
// TGAMMA-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=trunc | FileCheck %s -check-prefix=TRUNC
// TRUNC: CUDA API:
// TRUNC-NEXT:   trunc(d /*double*/);
// TRUNC-NEXT: Is migrated to:
// TRUNC-NEXT:   sycl::trunc(d);
// TRUNC-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=y0 | FileCheck %s -check-prefix=Y0
// Y0: CUDA API:
// Y0-NEXT:   y0(d /*double*/);
// Y0-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// Y0-NEXT:   sycl::ext::intel::math::y0(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=y1 | FileCheck %s -check-prefix=Y1
// Y1: CUDA API:
// Y1-NEXT:   y1(d /*double*/);
// Y1-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// Y1-NEXT:   sycl::ext::intel::math::y1(d);

/// Single Precision Intrinsics

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__expf | FileCheck %s -check-prefix=_EXPF
// _EXPF: CUDA API:
// _EXPF-NEXT:   __expf(f /*float*/);
// _EXPF-NEXT: Is migrated to:
// _EXPF-NEXT:   sycl::native::exp(f);

/// Type Casting Intrinsics

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2float_rd | FileCheck %s -check-prefix=__DOUBLE2FLOAT_RD
// __DOUBLE2FLOAT_RD: CUDA API:
// __DOUBLE2FLOAT_RD-NEXT:   __double2float_rd(d /*double*/);
// __DOUBLE2FLOAT_RD-NEXT: Is migrated to:
// __DOUBLE2FLOAT_RD-NEXT:   sycl::vec<double, 1>{d}.convert<float, sycl::rounding_mode::rtn>()[0];
// __DOUBLE2FLOAT_RD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2float_rn | FileCheck %s -check-prefix=__DOUBLE2FLOAT_RN
// __DOUBLE2FLOAT_RN: CUDA API:
// __DOUBLE2FLOAT_RN-NEXT:   __double2float_rn(d /*double*/);
// __DOUBLE2FLOAT_RN-NEXT: Is migrated to:
// __DOUBLE2FLOAT_RN-NEXT:   sycl::vec<double, 1>{d}.convert<float, sycl::rounding_mode::rte>()[0];
// __DOUBLE2FLOAT_RN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2float_ru | FileCheck %s -check-prefix=__DOUBLE2FLOAT_RU
// __DOUBLE2FLOAT_RU: CUDA API:
// __DOUBLE2FLOAT_RU-NEXT:   __double2float_ru(d /*double*/);
// __DOUBLE2FLOAT_RU-NEXT: Is migrated to:
// __DOUBLE2FLOAT_RU-NEXT:   sycl::vec<double, 1>{d}.convert<float, sycl::rounding_mode::rtp>()[0];
// __DOUBLE2FLOAT_RU-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2float_rz | FileCheck %s -check-prefix=__DOUBLE2FLOAT_RZ
// __DOUBLE2FLOAT_RZ: CUDA API:
// __DOUBLE2FLOAT_RZ-NEXT:   __double2float_rz(d /*double*/);
// __DOUBLE2FLOAT_RZ-NEXT: Is migrated to:
// __DOUBLE2FLOAT_RZ-NEXT:   sycl::vec<double, 1>{d}.convert<float, sycl::rounding_mode::rtz>()[0];
// __DOUBLE2FLOAT_RZ-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2hiint | FileCheck %s -check-prefix=__DOUBLE2HIINT
// __DOUBLE2HIINT: CUDA API:
// __DOUBLE2HIINT-NEXT:   __double2hiint(d /*double*/);
// __DOUBLE2HIINT-NEXT: Is migrated to:
// __DOUBLE2HIINT-NEXT:   dpct::cast_double_to_int(d);
// __DOUBLE2HIINT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2int_rd | FileCheck %s -check-prefix=__DOUBLE2INT_RD
// __DOUBLE2INT_RD: CUDA API:
// __DOUBLE2INT_RD-NEXT:   __double2int_rd(d /*double*/);
// __DOUBLE2INT_RD-NEXT: Is migrated to:
// __DOUBLE2INT_RD-NEXT:   sycl::vec<double, 1>{d}.convert<int, sycl::rounding_mode::rtn>()[0];
// __DOUBLE2INT_RD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2int_rn | FileCheck %s -check-prefix=__DOUBLE2INT_RN
// __DOUBLE2INT_RN: CUDA API:
// __DOUBLE2INT_RN-NEXT:   __double2int_rn(d /*double*/);
// __DOUBLE2INT_RN-NEXT: Is migrated to:
// __DOUBLE2INT_RN-NEXT:   sycl::vec<double, 1>{d}.convert<int, sycl::rounding_mode::rte>()[0];
// __DOUBLE2INT_RN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2int_ru | FileCheck %s -check-prefix=__DOUBLE2INT_RU
// __DOUBLE2INT_RU: CUDA API:
// __DOUBLE2INT_RU-NEXT:   __double2int_ru(d /*double*/);
// __DOUBLE2INT_RU-NEXT: Is migrated to:
// __DOUBLE2INT_RU-NEXT:   sycl::vec<double, 1>{d}.convert<int, sycl::rounding_mode::rtp>()[0];
// __DOUBLE2INT_RU-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2int_rz | FileCheck %s -check-prefix=__DOUBLE2INT_RZ
// __DOUBLE2INT_RZ: CUDA API:
// __DOUBLE2INT_RZ-NEXT:   __double2int_rz(d /*double*/);
// __DOUBLE2INT_RZ-NEXT: Is migrated to:
// __DOUBLE2INT_RZ-NEXT:   sycl::vec<double, 1>{d}.convert<int, sycl::rounding_mode::rtz>()[0];
// __DOUBLE2INT_RZ-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2ll_rd | FileCheck %s -check-prefix=__DOUBLE2LL_RD
// __DOUBLE2LL_RD: CUDA API:
// __DOUBLE2LL_RD-NEXT:   __double2ll_rd(d /*double*/);
// __DOUBLE2LL_RD-NEXT: Is migrated to:
// __DOUBLE2LL_RD-NEXT:   sycl::vec<double, 1>{d}.convert<long long, sycl::rounding_mode::rtn>()[0];
// __DOUBLE2LL_RD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2ll_rn | FileCheck %s -check-prefix=__DOUBLE2LL_RN
// __DOUBLE2LL_RN: CUDA API:
// __DOUBLE2LL_RN-NEXT:   __double2ll_rn(d /*double*/);
// __DOUBLE2LL_RN-NEXT: Is migrated to:
// __DOUBLE2LL_RN-NEXT:   sycl::vec<double, 1>{d}.convert<long long, sycl::rounding_mode::rte>()[0];
// __DOUBLE2LL_RN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2ll_ru | FileCheck %s -check-prefix=__DOUBLE2LL_RU
// __DOUBLE2LL_RU: CUDA API:
// __DOUBLE2LL_RU-NEXT:   __double2ll_ru(d /*double*/);
// __DOUBLE2LL_RU-NEXT: Is migrated to:
// __DOUBLE2LL_RU-NEXT:   sycl::vec<double, 1>{d}.convert<long long, sycl::rounding_mode::rtp>()[0];
// __DOUBLE2LL_RU-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2ll_rz | FileCheck %s -check-prefix=__DOUBLE2LL_RZ
// __DOUBLE2LL_RZ: CUDA API:
// __DOUBLE2LL_RZ-NEXT:   __double2ll_rz(d /*double*/);
// __DOUBLE2LL_RZ-NEXT: Is migrated to:
// __DOUBLE2LL_RZ-NEXT:   sycl::vec<double, 1>{d}.convert<long long, sycl::rounding_mode::rtz>()[0];
// __DOUBLE2LL_RZ-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2loint | FileCheck %s -check-prefix=__DOUBLE2LOINT
// __DOUBLE2LOINT: CUDA API:
// __DOUBLE2LOINT-NEXT:   __double2loint(d /*double*/);
// __DOUBLE2LOINT-NEXT: Is migrated to:
// __DOUBLE2LOINT-NEXT:   dpct::cast_double_to_int(d, false);
// __DOUBLE2LOINT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2uint_rd | FileCheck %s -check-prefix=__DOUBLE2UINT_RD
// __DOUBLE2UINT_RD: CUDA API:
// __DOUBLE2UINT_RD-NEXT:   __double2uint_rd(d /*double*/);
// __DOUBLE2UINT_RD-NEXT: Is migrated to:
// __DOUBLE2UINT_RD-NEXT:   sycl::vec<double, 1>{d}.convert<unsigned int, sycl::rounding_mode::rtn>()[0];
// __DOUBLE2UINT_RD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2uint_rn | FileCheck %s -check-prefix=__DOUBLE2UINT_RN
// __DOUBLE2UINT_RN: CUDA API:
// __DOUBLE2UINT_RN-NEXT:   __double2uint_rn(d /*double*/);
// __DOUBLE2UINT_RN-NEXT: Is migrated to:
// __DOUBLE2UINT_RN-NEXT:   sycl::vec<double, 1>{d}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
// __DOUBLE2UINT_RN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2uint_ru | FileCheck %s -check-prefix=__DOUBLE2UINT_RU
// __DOUBLE2UINT_RU: CUDA API:
// __DOUBLE2UINT_RU-NEXT:   __double2uint_ru(d /*double*/);
// __DOUBLE2UINT_RU-NEXT: Is migrated to:
// __DOUBLE2UINT_RU-NEXT:   sycl::vec<double, 1>{d}.convert<unsigned int, sycl::rounding_mode::rtp>()[0];
// __DOUBLE2UINT_RU-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2uint_rz | FileCheck %s -check-prefix=__DOUBLE2UINT_RZ
// __DOUBLE2UINT_RZ: CUDA API:
// __DOUBLE2UINT_RZ-NEXT:   __double2uint_rz(d /*double*/);
// __DOUBLE2UINT_RZ-NEXT: Is migrated to:
// __DOUBLE2UINT_RZ-NEXT:   sycl::vec<double, 1>{d}.convert<unsigned int, sycl::rounding_mode::rtz>()[0];
// __DOUBLE2UINT_RZ-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2ull_rd | FileCheck %s -check-prefix=__DOUBLE2ULL_RD
// __DOUBLE2ULL_RD: CUDA API:
// __DOUBLE2ULL_RD-NEXT:   __double2ull_rd(d /*double*/);
// __DOUBLE2ULL_RD-NEXT: Is migrated to:
// __DOUBLE2ULL_RD-NEXT:   sycl::vec<double, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rtn>()[0];
// __DOUBLE2ULL_RD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2ull_rn | FileCheck %s -check-prefix=__DOUBLE2ULL_RN
// __DOUBLE2ULL_RN: CUDA API:
// __DOUBLE2ULL_RN-NEXT:   __double2ull_rn(d /*double*/);
// __DOUBLE2ULL_RN-NEXT: Is migrated to:
// __DOUBLE2ULL_RN-NEXT:   sycl::vec<double, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rte>()[0];
// __DOUBLE2ULL_RN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2ull_ru | FileCheck %s -check-prefix=__DOUBLE2ULL_RU
// __DOUBLE2ULL_RU: CUDA API:
// __DOUBLE2ULL_RU-NEXT:   __double2ull_ru(d /*double*/);
// __DOUBLE2ULL_RU-NEXT: Is migrated to:
// __DOUBLE2ULL_RU-NEXT:   sycl::vec<double, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rtp>()[0];
// __DOUBLE2ULL_RU-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double2ull_rz | FileCheck %s -check-prefix=__DOUBLE2ULL_RZ
// __DOUBLE2ULL_RZ: CUDA API:
// __DOUBLE2ULL_RZ-NEXT:   __double2ull_rz(d /*double*/);
// __DOUBLE2ULL_RZ-NEXT: Is migrated to:
// __DOUBLE2ULL_RZ-NEXT:   sycl::vec<double, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rtz>()[0];
// __DOUBLE2ULL_RZ-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__double_as_longlong | FileCheck %s -check-prefix=__DOUBLE_AS_LONGLONG
// __DOUBLE_AS_LONGLONG: CUDA API:
// __DOUBLE_AS_LONGLONG-NEXT:   __double_as_longlong(d /*double*/);
// __DOUBLE_AS_LONGLONG-NEXT: Is migrated to:
// __DOUBLE_AS_LONGLONG-NEXT:   sycl::bit_cast<long long>(d);
// __DOUBLE_AS_LONGLONG-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2int_rd | FileCheck %s -check-prefix=__FLOAT2INT_RD
// __FLOAT2INT_RD: CUDA API:
// __FLOAT2INT_RD-NEXT:   __float2int_rd(d /*float*/);
// __FLOAT2INT_RD-NEXT: Is migrated to:
// __FLOAT2INT_RD-NEXT:   sycl::vec<float, 1>{d}.convert<int, sycl::rounding_mode::rtn>()[0];
// __FLOAT2INT_RD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2int_rn | FileCheck %s -check-prefix=__FLOAT2INT_RN
// __FLOAT2INT_RN: CUDA API:
// __FLOAT2INT_RN-NEXT:   __float2int_rn(d /*float*/);
// __FLOAT2INT_RN-NEXT: Is migrated to:
// __FLOAT2INT_RN-NEXT:   sycl::vec<float, 1>{d}.convert<int, sycl::rounding_mode::rte>()[0];
// __FLOAT2INT_RN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2int_ru | FileCheck %s -check-prefix=__FLOAT2INT_RU
// __FLOAT2INT_RU: CUDA API:
// __FLOAT2INT_RU-NEXT:   __float2int_ru(d /*float*/);
// __FLOAT2INT_RU-NEXT: Is migrated to:
// __FLOAT2INT_RU-NEXT:   sycl::vec<float, 1>{d}.convert<int, sycl::rounding_mode::rtp>()[0];
// __FLOAT2INT_RU-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2int_rz | FileCheck %s -check-prefix=__FLOAT2INT_RZ
// __FLOAT2INT_RZ: CUDA API:
// __FLOAT2INT_RZ-NEXT:   __float2int_rz(d /*float*/);
// __FLOAT2INT_RZ-NEXT: Is migrated to:
// __FLOAT2INT_RZ-NEXT:   sycl::vec<float, 1>{d}.convert<int, sycl::rounding_mode::rtz>()[0];
// __FLOAT2INT_RZ-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2ll_rd | FileCheck %s -check-prefix=__FLOAT2LL_RD
// __FLOAT2LL_RD: CUDA API:
// __FLOAT2LL_RD-NEXT:   __float2ll_rd(d /*float*/);
// __FLOAT2LL_RD-NEXT: Is migrated to:
// __FLOAT2LL_RD-NEXT:   sycl::vec<float, 1>{d}.convert<long long, sycl::rounding_mode::rtn>()[0];
// __FLOAT2LL_RD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2ll_rn | FileCheck %s -check-prefix=__FLOAT2LL_RN
// __FLOAT2LL_RN: CUDA API:
// __FLOAT2LL_RN-NEXT:   __float2ll_rn(d /*float*/);
// __FLOAT2LL_RN-NEXT: Is migrated to:
// __FLOAT2LL_RN-NEXT:   sycl::vec<float, 1>{d}.convert<long long, sycl::rounding_mode::rte>()[0];
// __FLOAT2LL_RN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2ll_ru | FileCheck %s -check-prefix=__FLOAT2LL_RU
// __FLOAT2LL_RU: CUDA API:
// __FLOAT2LL_RU-NEXT:   __float2ll_ru(d /*float*/);
// __FLOAT2LL_RU-NEXT: Is migrated to:
// __FLOAT2LL_RU-NEXT:   sycl::vec<float, 1>{d}.convert<long long, sycl::rounding_mode::rtp>()[0];
// __FLOAT2LL_RU-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2ll_rz | FileCheck %s -check-prefix=__FLOAT2LL_RZ
// __FLOAT2LL_RZ: CUDA API:
// __FLOAT2LL_RZ-NEXT:   __float2ll_rz(d /*float*/);
// __FLOAT2LL_RZ-NEXT: Is migrated to:
// __FLOAT2LL_RZ-NEXT:   sycl::vec<float, 1>{d}.convert<long long, sycl::rounding_mode::rtz>()[0];
// __FLOAT2LL_RZ-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2uint_rd | FileCheck %s -check-prefix=__FLOAT2UINT_RD
// __FLOAT2UINT_RD: CUDA API:
// __FLOAT2UINT_RD-NEXT:   __float2uint_rd(d /*float*/);
// __FLOAT2UINT_RD-NEXT: Is migrated to:
// __FLOAT2UINT_RD-NEXT:   sycl::vec<float, 1>{d}.convert<unsigned int, sycl::rounding_mode::rtn>()[0];
// __FLOAT2UINT_RD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2uint_rn | FileCheck %s -check-prefix=__FLOAT2UINT_RN
// __FLOAT2UINT_RN: CUDA API:
// __FLOAT2UINT_RN-NEXT:   __float2uint_rn(d /*float*/);
// __FLOAT2UINT_RN-NEXT: Is migrated to:
// __FLOAT2UINT_RN-NEXT:   sycl::vec<float, 1>{d}.convert<unsigned int, sycl::rounding_mode::rte>()[0];
// __FLOAT2UINT_RN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2uint_ru | FileCheck %s -check-prefix=__FLOAT2UINT_RU
// __FLOAT2UINT_RU: CUDA API:
// __FLOAT2UINT_RU-NEXT:   __float2uint_ru(d /*float*/);
// __FLOAT2UINT_RU-NEXT: Is migrated to:
// __FLOAT2UINT_RU-NEXT:   sycl::vec<float, 1>{d}.convert<unsigned int, sycl::rounding_mode::rtp>()[0];
// __FLOAT2UINT_RU-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2uint_rz | FileCheck %s -check-prefix=__FLOAT2UINT_RZ
// __FLOAT2UINT_RZ: CUDA API:
// __FLOAT2UINT_RZ-NEXT:   __float2uint_rz(d /*float*/);
// __FLOAT2UINT_RZ-NEXT: Is migrated to:
// __FLOAT2UINT_RZ-NEXT:   sycl::vec<float, 1>{d}.convert<unsigned int, sycl::rounding_mode::rtz>()[0];
// __FLOAT2UINT_RZ-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2ull_rd | FileCheck %s -check-prefix=__FLOAT2ULL_RD
// __FLOAT2ULL_RD: CUDA API:
// __FLOAT2ULL_RD-NEXT:   __float2ull_rd(d /*float*/);
// __FLOAT2ULL_RD-NEXT: Is migrated to:
// __FLOAT2ULL_RD-NEXT:   sycl::vec<float, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rtn>()[0];
// __FLOAT2ULL_RD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2ull_rn | FileCheck %s -check-prefix=__FLOAT2ULL_RN
// __FLOAT2ULL_RN: CUDA API:
// __FLOAT2ULL_RN-NEXT:   __float2ull_rn(d /*float*/);
// __FLOAT2ULL_RN-NEXT: Is migrated to:
// __FLOAT2ULL_RN-NEXT:   sycl::vec<float, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rte>()[0];
// __FLOAT2ULL_RN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2ull_ru | FileCheck %s -check-prefix=__FLOAT2ULL_RU
// __FLOAT2ULL_RU: CUDA API:
// __FLOAT2ULL_RU-NEXT:   __float2ull_ru(d /*float*/);
// __FLOAT2ULL_RU-NEXT: Is migrated to:
// __FLOAT2ULL_RU-NEXT:   sycl::vec<float, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rtp>()[0];
// __FLOAT2ULL_RU-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float2ull_rz | FileCheck %s -check-prefix=__FLOAT2ULL_RZ
// __FLOAT2ULL_RZ: CUDA API:
// __FLOAT2ULL_RZ-NEXT:   __float2ull_rz(d /*float*/);
// __FLOAT2ULL_RZ-NEXT: Is migrated to:
// __FLOAT2ULL_RZ-NEXT:   sycl::vec<float, 1>{d}.convert<unsigned long long, sycl::rounding_mode::rtz>()[0];
// __FLOAT2ULL_RZ-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float_as_int | FileCheck %s -check-prefix=__FLOAT_AS_INT
// __FLOAT_AS_INT: CUDA API:
// __FLOAT_AS_INT-NEXT:   __float_as_int(d /*float*/);
// __FLOAT_AS_INT-NEXT: Is migrated to:
// __FLOAT_AS_INT-NEXT:   sycl::bit_cast<int>(d);
// __FLOAT_AS_INT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__float_as_uint | FileCheck %s -check-prefix=__FLOAT_AS_UINT
// __FLOAT_AS_UINT: CUDA API:
// __FLOAT_AS_UINT-NEXT:   __float_as_uint(d /*float*/);
// __FLOAT_AS_UINT-NEXT: Is migrated to:
// __FLOAT_AS_UINT-NEXT:   sycl::bit_cast<unsigned int>(d);
// __FLOAT_AS_UINT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__hiloint2double | FileCheck %s -check-prefix=__HILOINT2DOUBLE
// __HILOINT2DOUBLE: CUDA API:
// __HILOINT2DOUBLE-NEXT:   __hiloint2double(i1 /*int*/, i2 /*int*/);
// __HILOINT2DOUBLE-NEXT: Is migrated to:
// __HILOINT2DOUBLE-NEXT:   dpct::cast_ints_to_double(i1, i2);
// __HILOINT2DOUBLE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int2double_rn | FileCheck %s -check-prefix=__INT2DOUBLE_RN
// __INT2DOUBLE_RN: CUDA API:
// __INT2DOUBLE_RN-NEXT:   __int2double_rn(i /*int*/);
// __INT2DOUBLE_RN-NEXT: Is migrated to:
// __INT2DOUBLE_RN-NEXT:   sycl::vec<int, 1>{i}.convert<double, sycl::rounding_mode::rte>()[0];
// __INT2DOUBLE_RN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int2float_rd | FileCheck %s -check-prefix=__INT2FLOAT_RD
// __INT2FLOAT_RD: CUDA API:
// __INT2FLOAT_RD-NEXT:   __int2float_rd(i /*int*/);
// __INT2FLOAT_RD-NEXT: Is migrated to:
// __INT2FLOAT_RD-NEXT:   sycl::vec<int, 1>{i}.convert<float, sycl::rounding_mode::rtn>()[0];
// __INT2FLOAT_RD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int2float_rn | FileCheck %s -check-prefix=__INT2FLOAT_RN
// __INT2FLOAT_RN: CUDA API:
// __INT2FLOAT_RN-NEXT:   __int2float_rn(i /*int*/);
// __INT2FLOAT_RN-NEXT: Is migrated to:
// __INT2FLOAT_RN-NEXT:   sycl::vec<int, 1>{i}.convert<float, sycl::rounding_mode::rte>()[0];
// __INT2FLOAT_RN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int2float_ru | FileCheck %s -check-prefix=__INT2FLOAT_RU
// __INT2FLOAT_RU: CUDA API:
// __INT2FLOAT_RU-NEXT:   __int2float_ru(i /*int*/);
// __INT2FLOAT_RU-NEXT: Is migrated to:
// __INT2FLOAT_RU-NEXT:   sycl::vec<int, 1>{i}.convert<float, sycl::rounding_mode::rtp>()[0];
// __INT2FLOAT_RU-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int2float_rz | FileCheck %s -check-prefix=__INT2FLOAT_RZ
// __INT2FLOAT_RZ: CUDA API:
// __INT2FLOAT_RZ-NEXT:   __int2float_rz(i /*int*/);
// __INT2FLOAT_RZ-NEXT: Is migrated to:
// __INT2FLOAT_RZ-NEXT:   sycl::vec<int, 1>{i}.convert<float, sycl::rounding_mode::rtz>()[0];
// __INT2FLOAT_RZ-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__int_as_float | FileCheck %s -check-prefix=__INT_AS_FLOAT
// __INT_AS_FLOAT: CUDA API:
// __INT_AS_FLOAT-NEXT:   __int_as_float(i /*int*/);
// __INT_AS_FLOAT-NEXT: Is migrated to:
// __INT_AS_FLOAT-NEXT:   sycl::bit_cast<float>(i);
// __INT_AS_FLOAT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2double_rd | FileCheck %s -check-prefix=__LL2DOUBLE_RD
// __LL2DOUBLE_RD: CUDA API:
// __LL2DOUBLE_RD-NEXT:   __ll2double_rd(ll /*long long int*/);
// __LL2DOUBLE_RD-NEXT: Is migrated to:
// __LL2DOUBLE_RD-NEXT:   sycl::vec<long long, 1>{ll}.convert<double, sycl::rounding_mode::rtn>()[0];
// __LL2DOUBLE_RD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2double_rn | FileCheck %s -check-prefix=__LL2DOUBLE_RN
// __LL2DOUBLE_RN: CUDA API:
// __LL2DOUBLE_RN-NEXT:   __ll2double_rn(ll /*long long int*/);
// __LL2DOUBLE_RN-NEXT: Is migrated to:
// __LL2DOUBLE_RN-NEXT:   sycl::vec<long long, 1>{ll}.convert<double, sycl::rounding_mode::rte>()[0];
// __LL2DOUBLE_RN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2double_ru | FileCheck %s -check-prefix=__LL2DOUBLE_RU
// __LL2DOUBLE_RU: CUDA API:
// __LL2DOUBLE_RU-NEXT:   __ll2double_ru(ll /*long long int*/);
// __LL2DOUBLE_RU-NEXT: Is migrated to:
// __LL2DOUBLE_RU-NEXT:   sycl::vec<long long, 1>{ll}.convert<double, sycl::rounding_mode::rtp>()[0];
// __LL2DOUBLE_RU-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2double_rz | FileCheck %s -check-prefix=__LL2DOUBLE_RZ
// __LL2DOUBLE_RZ: CUDA API:
// __LL2DOUBLE_RZ-NEXT:   __ll2double_rz(ll /*long long int*/);
// __LL2DOUBLE_RZ-NEXT: Is migrated to:
// __LL2DOUBLE_RZ-NEXT:   sycl::vec<long long, 1>{ll}.convert<double, sycl::rounding_mode::rtz>()[0];
// __LL2DOUBLE_RZ-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2float_rd | FileCheck %s -check-prefix=__LL2FLOAT_RD
// __LL2FLOAT_RD: CUDA API:
// __LL2FLOAT_RD-NEXT:   __ll2float_rd(ll /*long long int*/);
// __LL2FLOAT_RD-NEXT: Is migrated to:
// __LL2FLOAT_RD-NEXT:   sycl::vec<long long, 1>{ll}.convert<float, sycl::rounding_mode::rtn>()[0];
// __LL2FLOAT_RD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2float_rn | FileCheck %s -check-prefix=__LL2FLOAT_RN
// __LL2FLOAT_RN: CUDA API:
// __LL2FLOAT_RN-NEXT:   __ll2float_rn(ll /*long long int*/);
// __LL2FLOAT_RN-NEXT: Is migrated to:
// __LL2FLOAT_RN-NEXT:   sycl::vec<long long, 1>{ll}.convert<float, sycl::rounding_mode::rte>()[0];
// __LL2FLOAT_RN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2float_ru | FileCheck %s -check-prefix=__LL2FLOAT_RU
// __LL2FLOAT_RU: CUDA API:
// __LL2FLOAT_RU-NEXT:   __ll2float_ru(ll /*long long int*/);
// __LL2FLOAT_RU-NEXT: Is migrated to:
// __LL2FLOAT_RU-NEXT:   sycl::vec<long long, 1>{ll}.convert<float, sycl::rounding_mode::rtp>()[0];
// __LL2FLOAT_RU-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ll2float_rz | FileCheck %s -check-prefix=__LL2FLOAT_RZ
// __LL2FLOAT_RZ: CUDA API:
// __LL2FLOAT_RZ-NEXT:   __ll2float_rz(ll /*long long int*/);
// __LL2FLOAT_RZ-NEXT: Is migrated to:
// __LL2FLOAT_RZ-NEXT:   sycl::vec<long long, 1>{ll}.convert<float, sycl::rounding_mode::rtz>()[0];
// __LL2FLOAT_RZ-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__longlong_as_double | FileCheck %s -check-prefix=__LONGLONG_AS_DOUBLE
// __LONGLONG_AS_DOUBLE: CUDA API:
// __LONGLONG_AS_DOUBLE-NEXT:   __longlong_as_double(ll /*long long int*/);
// __LONGLONG_AS_DOUBLE-NEXT: Is migrated to:
// __LONGLONG_AS_DOUBLE-NEXT:   sycl::bit_cast<double>(ll);
// __LONGLONG_AS_DOUBLE-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint2double_rn | FileCheck %s -check-prefix=__UINT2DOUBLE_RN
// __UINT2DOUBLE_RN: CUDA API:
// __UINT2DOUBLE_RN-NEXT:   __uint2double_rn(u /*unsigned int*/);
// __UINT2DOUBLE_RN-NEXT: Is migrated to:
// __UINT2DOUBLE_RN-NEXT:   sycl::vec<unsigned int, 1>{u}.convert<double, sycl::rounding_mode::rte>()[0];
// __UINT2DOUBLE_RN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint2float_rd | FileCheck %s -check-prefix=__UINT2FLOAT_RD
// __UINT2FLOAT_RD: CUDA API:
// __UINT2FLOAT_RD-NEXT:   __uint2float_rd(u /*unsigned int*/);
// __UINT2FLOAT_RD-NEXT: Is migrated to:
// __UINT2FLOAT_RD-NEXT:   sycl::vec<unsigned int, 1>{u}.convert<float, sycl::rounding_mode::rtn>()[0];
// __UINT2FLOAT_RD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint2float_rn | FileCheck %s -check-prefix=__UINT2FLOAT_RN
// __UINT2FLOAT_RN: CUDA API:
// __UINT2FLOAT_RN-NEXT:   __uint2float_rn(u /*unsigned int*/);
// __UINT2FLOAT_RN-NEXT: Is migrated to:
// __UINT2FLOAT_RN-NEXT:   sycl::vec<unsigned int, 1>{u}.convert<float, sycl::rounding_mode::rte>()[0];
// __UINT2FLOAT_RN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint2float_ru | FileCheck %s -check-prefix=__UINT2FLOAT_RU
// __UINT2FLOAT_RU: CUDA API:
// __UINT2FLOAT_RU-NEXT:   __uint2float_ru(u /*unsigned int*/);
// __UINT2FLOAT_RU-NEXT: Is migrated to:
// __UINT2FLOAT_RU-NEXT:   sycl::vec<unsigned int, 1>{u}.convert<float, sycl::rounding_mode::rtp>()[0];
// __UINT2FLOAT_RU-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint2float_rz | FileCheck %s -check-prefix=__UINT2FLOAT_RZ
// __UINT2FLOAT_RZ: CUDA API:
// __UINT2FLOAT_RZ-NEXT:   __uint2float_rz(u /*unsigned int*/);
// __UINT2FLOAT_RZ-NEXT: Is migrated to:
// __UINT2FLOAT_RZ-NEXT:   sycl::vec<unsigned int, 1>{u}.convert<float, sycl::rounding_mode::rtz>()[0];
// __UINT2FLOAT_RZ-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uint_as_float | FileCheck %s -check-prefix=__UINT_AS_FLOAT
// __UINT_AS_FLOAT: CUDA API:
// __UINT_AS_FLOAT-NEXT:   __uint_as_float(u /*unsigned int*/);
// __UINT_AS_FLOAT-NEXT: Is migrated to:
// __UINT_AS_FLOAT-NEXT:   sycl::bit_cast<float>(u);
// __UINT_AS_FLOAT-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2double_rd | FileCheck %s -check-prefix=__ULL2DOUBLE_RD
// __ULL2DOUBLE_RD: CUDA API:
// __ULL2DOUBLE_RD-NEXT:   __ull2double_rd(ull /*unsigned long long int*/);
// __ULL2DOUBLE_RD-NEXT: Is migrated to:
// __ULL2DOUBLE_RD-NEXT:   sycl::vec<unsigned long long, 1>{ull}.convert<double, sycl::rounding_mode::rtn>()[0];
// __ULL2DOUBLE_RD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2double_rn | FileCheck %s -check-prefix=__ULL2DOUBLE_RN
// __ULL2DOUBLE_RN: CUDA API:
// __ULL2DOUBLE_RN-NEXT:   __ull2double_rn(ull /*unsigned long long int*/);
// __ULL2DOUBLE_RN-NEXT: Is migrated to:
// __ULL2DOUBLE_RN-NEXT:   sycl::vec<unsigned long long, 1>{ull}.convert<double, sycl::rounding_mode::rte>()[0];
// __ULL2DOUBLE_RN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2double_ru | FileCheck %s -check-prefix=__ULL2DOUBLE_RU
// __ULL2DOUBLE_RU: CUDA API:
// __ULL2DOUBLE_RU-NEXT:   __ull2double_ru(ull /*unsigned long long int*/);
// __ULL2DOUBLE_RU-NEXT: Is migrated to:
// __ULL2DOUBLE_RU-NEXT:   sycl::vec<unsigned long long, 1>{ull}.convert<double, sycl::rounding_mode::rtp>()[0];
// __ULL2DOUBLE_RU-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2double_rz | FileCheck %s -check-prefix=__ULL2DOUBLE_RZ
// __ULL2DOUBLE_RZ: CUDA API:
// __ULL2DOUBLE_RZ-NEXT:   __ull2double_rz(ull /*unsigned long long int*/);
// __ULL2DOUBLE_RZ-NEXT: Is migrated to:
// __ULL2DOUBLE_RZ-NEXT:   sycl::vec<unsigned long long, 1>{ull}.convert<double, sycl::rounding_mode::rtz>()[0];
// __ULL2DOUBLE_RZ-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2float_rd | FileCheck %s -check-prefix=__ULL2FLOAT_RD
// __ULL2FLOAT_RD: CUDA API:
// __ULL2FLOAT_RD-NEXT:   __ull2float_rd(ull /*unsigned long long int*/);
// __ULL2FLOAT_RD-NEXT: Is migrated to:
// __ULL2FLOAT_RD-NEXT:   sycl::vec<unsigned long long, 1>{ull}.convert<float, sycl::rounding_mode::rtn>()[0];
// __ULL2FLOAT_RD-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2float_rn | FileCheck %s -check-prefix=__ULL2FLOAT_RN
// __ULL2FLOAT_RN: CUDA API:
// __ULL2FLOAT_RN-NEXT:   __ull2float_rn(ull /*unsigned long long int*/);
// __ULL2FLOAT_RN-NEXT: Is migrated to:
// __ULL2FLOAT_RN-NEXT:   sycl::vec<unsigned long long, 1>{ull}.convert<float, sycl::rounding_mode::rte>()[0];
// __ULL2FLOAT_RN-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2float_ru | FileCheck %s -check-prefix=__ULL2FLOAT_RU
// __ULL2FLOAT_RU: CUDA API:
// __ULL2FLOAT_RU-NEXT:   __ull2float_ru(ull /*unsigned long long int*/);
// __ULL2FLOAT_RU-NEXT: Is migrated to:
// __ULL2FLOAT_RU-NEXT:   sycl::vec<unsigned long long, 1>{ull}.convert<float, sycl::rounding_mode::rtp>()[0];
// __ULL2FLOAT_RU-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ull2float_rz | FileCheck %s -check-prefix=__ULL2FLOAT_RZ
// __ULL2FLOAT_RZ: CUDA API:
// __ULL2FLOAT_RZ-NEXT:   __ull2float_rz(ull /*unsigned long long int*/);
// __ULL2FLOAT_RZ-NEXT: Is migrated to:
// __ULL2FLOAT_RZ-NEXT:   sycl::vec<unsigned long long, 1>{ull}.convert<float, sycl::rounding_mode::rtz>()[0];
// __ULL2FLOAT_RZ-EMPTY:

/// SIMD Intrinsics

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vabs2 | FileCheck %s -check-prefix=VABS2
// VABS2: CUDA API:
// VABS2-NEXT:   __vabs2(u /*unsigned int*/);
// VABS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VABS2-NEXT:   sycl::ext::intel::math::vabs2(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vabs4 | FileCheck %s -check-prefix=VABS4
// VABS4: CUDA API:
// VABS4-NEXT:   __vabs4(u /*unsigned int*/);
// VABS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VABS4-NEXT:   sycl::ext::intel::math::vabs4(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vabsdiffs2 | FileCheck %s -check-prefix=VABSDIFFS2
// VABSDIFFS2: CUDA API:
// VABSDIFFS2-NEXT:   __vabsdiffs2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VABSDIFFS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VABSDIFFS2-NEXT:   sycl::ext::intel::math::vabsdiffs2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vabsdiffs4 | FileCheck %s -check-prefix=VABSDIFFS4
// VABSDIFFS4: CUDA API:
// VABSDIFFS4-NEXT:   __vabsdiffs4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VABSDIFFS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VABSDIFFS4-NEXT:   sycl::ext::intel::math::vabsdiffs4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vabsdiffu2 | FileCheck %s -check-prefix=VABSDIFFU2
// VABSDIFFU2: CUDA API:
// VABSDIFFU2-NEXT:   __vabsdiffu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VABSDIFFU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VABSDIFFU2-NEXT:   sycl::ext::intel::math::vabsdiffu2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vabsdiffu4 | FileCheck %s -check-prefix=VABSDIFFU4
// VABSDIFFU4: CUDA API:
// VABSDIFFU4-NEXT:   __vabsdiffu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VABSDIFFU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VABSDIFFU4-NEXT:   sycl::ext::intel::math::vabsdiffu4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vabsss2 | FileCheck %s -check-prefix=VABSSS2
// VABSSS2: CUDA API:
// VABSSS2-NEXT:   __vabsss2(u /*unsigned int*/);
// VABSSS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VABSSS2-NEXT:   sycl::ext::intel::math::vabsss2(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vabsss4 | FileCheck %s -check-prefix=VABSSS4
// VABSSS4: CUDA API:
// VABSSS4-NEXT:   __vabsss4(u /*unsigned int*/);
// VABSSS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VABSSS4-NEXT:   sycl::ext::intel::math::vabsss4(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vadd2 | FileCheck %s -check-prefix=VADD2
// VADD2: CUDA API:
// VADD2-NEXT:   __vadd2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VADD2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VADD2-NEXT:   sycl::ext::intel::math::vadd2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vadd4 | FileCheck %s -check-prefix=VADD4
// VADD4: CUDA API:
// VADD4-NEXT:   __vadd4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VADD4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VADD4-NEXT:   sycl::ext::intel::math::vadd4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vaddss2 | FileCheck %s -check-prefix=VADDSS2
// VADDSS2: CUDA API:
// VADDSS2-NEXT:   __vaddss2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VADDSS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VADDSS2-NEXT:   sycl::ext::intel::math::vaddss2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vaddss4 | FileCheck %s -check-prefix=VADDSS4
// VADDSS4: CUDA API:
// VADDSS4-NEXT:   __vaddss4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VADDSS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VADDSS4-NEXT:   sycl::ext::intel::math::vaddss4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vaddus2 | FileCheck %s -check-prefix=VADDUS2
// VADDUS2: CUDA API:
// VADDUS2-NEXT:   __vaddus2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VADDUS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VADDUS2-NEXT:   sycl::ext::intel::math::vaddus2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vaddus4 | FileCheck %s -check-prefix=VADDUS4
// VADDUS4: CUDA API:
// VADDUS4-NEXT:   __vaddus4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VADDUS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VADDUS4-NEXT:   sycl::ext::intel::math::vaddus4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vavgs2 | FileCheck %s -check-prefix=VAVGS2
// VAVGS2: CUDA API:
// VAVGS2-NEXT:   __vavgs2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VAVGS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VAVGS2-NEXT:   sycl::ext::intel::math::vavgs2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vavgs4 | FileCheck %s -check-prefix=VAVGS4
// VAVGS4: CUDA API:
// VAVGS4-NEXT:   __vavgs4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VAVGS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VAVGS4-NEXT:   sycl::ext::intel::math::vavgs4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vavgu2 | FileCheck %s -check-prefix=VAVGU2
// VAVGU2: CUDA API:
// VAVGU2-NEXT:   __vavgu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VAVGU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VAVGU2-NEXT:   sycl::ext::intel::math::vavgu2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vavgu4 | FileCheck %s -check-prefix=VAVGU4
// VAVGU4: CUDA API:
// VAVGU4-NEXT:   __vavgu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VAVGU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VAVGU4-NEXT:   sycl::ext::intel::math::vavgu4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpeq2 | FileCheck %s -check-prefix=VCMPEQ2
// VCMPEQ2: CUDA API:
// VCMPEQ2-NEXT:   __vcmpeq2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPEQ2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPEQ2-NEXT:   sycl::ext::intel::math::vcmpeq2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpeq4 | FileCheck %s -check-prefix=VCMPEQ4
// VCMPEQ4: CUDA API:
// VCMPEQ4-NEXT:   __vcmpeq4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPEQ4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPEQ4-NEXT:   sycl::ext::intel::math::vcmpeq4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpges2 | FileCheck %s -check-prefix=VCMPGES2
// VCMPGES2: CUDA API:
// VCMPGES2-NEXT:   __vcmpges2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPGES2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPGES2-NEXT:   sycl::ext::intel::math::vcmpges2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpges4 | FileCheck %s -check-prefix=VCMPGES4
// VCMPGES4: CUDA API:
// VCMPGES4-NEXT:   __vcmpges4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPGES4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPGES4-NEXT:   sycl::ext::intel::math::vcmpges4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpgeu2 | FileCheck %s -check-prefix=VCMPGEU2
// VCMPGEU2: CUDA API:
// VCMPGEU2-NEXT:   __vcmpgeu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPGEU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPGEU2-NEXT:   sycl::ext::intel::math::vcmpgeu2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpgeu4 | FileCheck %s -check-prefix=VCMPGEU4
// VCMPGEU4: CUDA API:
// VCMPGEU4-NEXT:   __vcmpgeu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPGEU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPGEU4-NEXT:   sycl::ext::intel::math::vcmpgeu4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpgts2 | FileCheck %s -check-prefix=VCMPGTS2
// VCMPGTS2: CUDA API:
// VCMPGTS2-NEXT:   __vcmpgts2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPGTS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPGTS2-NEXT:   sycl::ext::intel::math::vcmpgts2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpgts4 | FileCheck %s -check-prefix=VCMPGTS4
// VCMPGTS4: CUDA API:
// VCMPGTS4-NEXT:   __vcmpgts4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPGTS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPGTS4-NEXT:   sycl::ext::intel::math::vcmpgts4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpgtu2 | FileCheck %s -check-prefix=VCMPGTU2
// VCMPGTU2: CUDA API:
// VCMPGTU2-NEXT:   __vcmpgtu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPGTU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPGTU2-NEXT:   sycl::ext::intel::math::vcmpgtu2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpgtu4 | FileCheck %s -check-prefix=VCMPGTU4
// VCMPGTU4: CUDA API:
// VCMPGTU4-NEXT:   __vcmpgtu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPGTU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPGTU4-NEXT:   sycl::ext::intel::math::vcmpgtu4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmples2 | FileCheck %s -check-prefix=VCMPLES2
// VCMPLES2: CUDA API:
// VCMPLES2-NEXT:   __vcmples2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPLES2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPLES2-NEXT:   sycl::ext::intel::math::vcmples2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmples4 | FileCheck %s -check-prefix=VCMPLES4
// VCMPLES4: CUDA API:
// VCMPLES4-NEXT:   __vcmples4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPLES4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPLES4-NEXT:   sycl::ext::intel::math::vcmples4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpleu2 | FileCheck %s -check-prefix=VCMPLEU2
// VCMPLEU2: CUDA API:
// VCMPLEU2-NEXT:   __vcmpleu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPLEU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPLEU2-NEXT:   sycl::ext::intel::math::vcmpleu2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpleu4 | FileCheck %s -check-prefix=VCMPLEU4
// VCMPLEU4: CUDA API:
// VCMPLEU4-NEXT:   __vcmpleu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPLEU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPLEU4-NEXT:   sycl::ext::intel::math::vcmpleu4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmplts2 | FileCheck %s -check-prefix=VCMPLTS2
// VCMPLTS2: CUDA API:
// VCMPLTS2-NEXT:   __vcmplts2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPLTS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPLTS2-NEXT:   sycl::ext::intel::math::vcmplts2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmplts4 | FileCheck %s -check-prefix=VCMPLTS4
// VCMPLTS4: CUDA API:
// VCMPLTS4-NEXT:   __vcmplts4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPLTS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPLTS4-NEXT:   sycl::ext::intel::math::vcmplts4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpltu2 | FileCheck %s -check-prefix=VCMPLTU2
// VCMPLTU2: CUDA API:
// VCMPLTU2-NEXT:   __vcmpltu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPLTU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPLTU2-NEXT:   sycl::ext::intel::math::vcmpltu2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpltu4 | FileCheck %s -check-prefix=VCMPLTU4
// VCMPLTU4: CUDA API:
// VCMPLTU4-NEXT:   __vcmpltu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPLTU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPLTU4-NEXT:   sycl::ext::intel::math::vcmpltu4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpne2 | FileCheck %s -check-prefix=VCMPNE2
// VCMPNE2: CUDA API:
// VCMPNE2-NEXT:   __vcmpne2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPNE2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPNE2-NEXT:   sycl::ext::intel::math::vcmpne2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpne4 | FileCheck %s -check-prefix=VCMPNE4
// VCMPNE4: CUDA API:
// VCMPNE4-NEXT:   __vcmpne4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPNE4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPNE4-NEXT:   sycl::ext::intel::math::vcmpne4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vhaddu2 | FileCheck %s -check-prefix=VHADDU2
// VHADDU2: CUDA API:
// VHADDU2-NEXT:   __vhaddu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VHADDU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VHADDU2-NEXT:   sycl::ext::intel::math::vhaddu2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vhaddu4 | FileCheck %s -check-prefix=VHADDU4
// VHADDU4: CUDA API:
// VHADDU4-NEXT:   __vhaddu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VHADDU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VHADDU4-NEXT:   sycl::ext::intel::math::vhaddu4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vmaxs2 | FileCheck %s -check-prefix=VMAXS2
// VMAXS2: CUDA API:
// VMAXS2-NEXT:   __vmaxs2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VMAXS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VMAXS2-NEXT:   sycl::ext::intel::math::vmaxs2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vmaxs4 | FileCheck %s -check-prefix=VMAXS4
// VMAXS4: CUDA API:
// VMAXS4-NEXT:   __vmaxs4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VMAXS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VMAXS4-NEXT:   sycl::ext::intel::math::vmaxs4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vmaxu2 | FileCheck %s -check-prefix=VMAXU2
// VMAXU2: CUDA API:
// VMAXU2-NEXT:   __vmaxu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VMAXU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VMAXU2-NEXT:   sycl::ext::intel::math::vmaxu2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vmaxu4 | FileCheck %s -check-prefix=VMAXU4
// VMAXU4: CUDA API:
// VMAXU4-NEXT:   __vmaxu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VMAXU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VMAXU4-NEXT:   sycl::ext::intel::math::vmaxu4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vmins2 | FileCheck %s -check-prefix=VMINS2
// VMINS2: CUDA API:
// VMINS2-NEXT:   __vmins2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VMINS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VMINS2-NEXT:   sycl::ext::intel::math::vmins2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vmins4 | FileCheck %s -check-prefix=VMINS4
// VMINS4: CUDA API:
// VMINS4-NEXT:   __vmins4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VMINS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VMINS4-NEXT:   sycl::ext::intel::math::vmins4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vminu2 | FileCheck %s -check-prefix=VMINU2
// VMINU2: CUDA API:
// VMINU2-NEXT:   __vminu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VMINU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VMINU2-NEXT:   sycl::ext::intel::math::vminu2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vminu4 | FileCheck %s -check-prefix=VMINU4
// VMINU4: CUDA API:
// VMINU4-NEXT:   __vminu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VMINU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VMINU4-NEXT:   sycl::ext::intel::math::vminu4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vneg2 | FileCheck %s -check-prefix=VNEG2
// VNEG2: CUDA API:
// VNEG2-NEXT:   __vneg2(u /*unsigned int*/);
// VNEG2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VNEG2-NEXT:   sycl::ext::intel::math::vneg2(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vneg4 | FileCheck %s -check-prefix=VNEG4
// VNEG4: CUDA API:
// VNEG4-NEXT:   __vneg4(u /*unsigned int*/);
// VNEG4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VNEG4-NEXT:   sycl::ext::intel::math::vneg4(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vnegss2 | FileCheck %s -check-prefix=VNEGSS2
// VNEGSS2: CUDA API:
// VNEGSS2-NEXT:   __vnegss2(u /*unsigned int*/);
// VNEGSS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VNEGSS2-NEXT:   sycl::ext::intel::math::vnegss2(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vnegss4 | FileCheck %s -check-prefix=VNEGSS4
// VNEGSS4: CUDA API:
// VNEGSS4-NEXT:   __vnegss4(u /*unsigned int*/);
// VNEGSS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VNEGSS4-NEXT:   sycl::ext::intel::math::vnegss4(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsads2 | FileCheck %s -check-prefix=VSADS2
// VSADS2: CUDA API:
// VSADS2-NEXT:   __vsads2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSADS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSADS2-NEXT:   sycl::ext::intel::math::vsads2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsads4 | FileCheck %s -check-prefix=VSADS4
// VSADS4: CUDA API:
// VSADS4-NEXT:   __vsads4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSADS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSADS4-NEXT:   sycl::ext::intel::math::vsads4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsadu2 | FileCheck %s -check-prefix=VSADU2
// VSADU2: CUDA API:
// VSADU2-NEXT:   __vsadu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSADU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSADU2-NEXT:   sycl::ext::intel::math::vsadu2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsadu4 | FileCheck %s -check-prefix=VSADU4
// VSADU4: CUDA API:
// VSADU4-NEXT:   __vsadu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSADU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSADU4-NEXT:   sycl::ext::intel::math::vsadu4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vseteq2 | FileCheck %s -check-prefix=VSETEQ2
// VSETEQ2: CUDA API:
// VSETEQ2-NEXT:   __vseteq2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETEQ2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETEQ2-NEXT:   sycl::ext::intel::math::vseteq2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vseteq4 | FileCheck %s -check-prefix=VSETEQ4
// VSETEQ4: CUDA API:
// VSETEQ4-NEXT:   __vseteq4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETEQ4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETEQ4-NEXT:   sycl::ext::intel::math::vseteq4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetges2 | FileCheck %s -check-prefix=VSETGES2
// VSETGES2: CUDA API:
// VSETGES2-NEXT:   __vsetges2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETGES2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETGES2-NEXT:   sycl::ext::intel::math::vsetges2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetges4 | FileCheck %s -check-prefix=VSETGES4
// VSETGES4: CUDA API:
// VSETGES4-NEXT:   __vsetges4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETGES4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETGES4-NEXT:   sycl::ext::intel::math::vsetges4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetgeu2 | FileCheck %s -check-prefix=VSETGEU2
// VSETGEU2: CUDA API:
// VSETGEU2-NEXT:   __vsetgeu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETGEU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETGEU2-NEXT:   sycl::ext::intel::math::vsetgeu2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetgeu4 | FileCheck %s -check-prefix=VSETGEU4
// VSETGEU4: CUDA API:
// VSETGEU4-NEXT:   __vsetgeu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETGEU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETGEU4-NEXT:   sycl::ext::intel::math::vsetgeu4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetgts2 | FileCheck %s -check-prefix=VSETGTS2
// VSETGTS2: CUDA API:
// VSETGTS2-NEXT:   __vsetgts2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETGTS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETGTS2-NEXT:   sycl::ext::intel::math::vsetgts2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetgts4 | FileCheck %s -check-prefix=VSETGTS4
// VSETGTS4: CUDA API:
// VSETGTS4-NEXT:   __vsetgts4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETGTS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETGTS4-NEXT:   sycl::ext::intel::math::vsetgts4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetgtu2 | FileCheck %s -check-prefix=VSETGTU2
// VSETGTU2: CUDA API:
// VSETGTU2-NEXT:   __vsetgtu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETGTU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETGTU2-NEXT:   sycl::ext::intel::math::vsetgtu2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetgtu4 | FileCheck %s -check-prefix=VSETGTU4
// VSETGTU4: CUDA API:
// VSETGTU4-NEXT:   __vsetgtu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETGTU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETGTU4-NEXT:   sycl::ext::intel::math::vsetgtu4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetles2 | FileCheck %s -check-prefix=VSETLES2
// VSETLES2: CUDA API:
// VSETLES2-NEXT:   __vsetles2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETLES2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETLES2-NEXT:   sycl::ext::intel::math::vsetles2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetles4 | FileCheck %s -check-prefix=VSETLES4
// VSETLES4: CUDA API:
// VSETLES4-NEXT:   __vsetles4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETLES4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETLES4-NEXT:   sycl::ext::intel::math::vsetles4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetleu2 | FileCheck %s -check-prefix=VSETLEU2
// VSETLEU2: CUDA API:
// VSETLEU2-NEXT:   __vsetleu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETLEU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETLEU2-NEXT:   sycl::ext::intel::math::vsetleu2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetleu4 | FileCheck %s -check-prefix=VSETLEU4
// VSETLEU4: CUDA API:
// VSETLEU4-NEXT:   __vsetleu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETLEU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETLEU4-NEXT:   sycl::ext::intel::math::vsetleu4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetlts2 | FileCheck %s -check-prefix=VSETLTS2
// VSETLTS2: CUDA API:
// VSETLTS2-NEXT:   __vsetlts2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETLTS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETLTS2-NEXT:   sycl::ext::intel::math::vsetlts2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetlts4 | FileCheck %s -check-prefix=VSETLTS4
// VSETLTS4: CUDA API:
// VSETLTS4-NEXT:   __vsetlts4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETLTS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETLTS4-NEXT:   sycl::ext::intel::math::vsetlts4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetltu2 | FileCheck %s -check-prefix=VSETLTU2
// VSETLTU2: CUDA API:
// VSETLTU2-NEXT:   __vsetltu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETLTU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETLTU2-NEXT:   sycl::ext::intel::math::vsetltu2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetltu4 | FileCheck %s -check-prefix=VSETLTU4
// VSETLTU4: CUDA API:
// VSETLTU4-NEXT:   __vsetltu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETLTU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETLTU4-NEXT:   sycl::ext::intel::math::vsetltu4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetne2 | FileCheck %s -check-prefix=VSETNE2
// VSETNE2: CUDA API:
// VSETNE2-NEXT:   __vsetne2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETNE2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETNE2-NEXT:   sycl::ext::intel::math::vsetne2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetne4 | FileCheck %s -check-prefix=VSETNE4
// VSETNE4: CUDA API:
// VSETNE4-NEXT:   __vsetne4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETNE4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETNE4-NEXT:   sycl::ext::intel::math::vsetne4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsub2 | FileCheck %s -check-prefix=VSUB2
// VSUB2: CUDA API:
// VSUB2-NEXT:   __vsub2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSUB2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSUB2-NEXT:   sycl::ext::intel::math::vsub2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsub4 | FileCheck %s -check-prefix=VSUB4
// VSUB4: CUDA API:
// VSUB4-NEXT:   __vsub4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSUB4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSUB4-NEXT:   sycl::ext::intel::math::vsub4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsubss2 | FileCheck %s -check-prefix=VSUBSS2
// VSUBSS2: CUDA API:
// VSUBSS2-NEXT:   __vsubss2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSUBSS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSUBSS2-NEXT:   sycl::ext::intel::math::vsubss2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsubss4 | FileCheck %s -check-prefix=VSUBSS4
// VSUBSS4: CUDA API:
// VSUBSS4-NEXT:   __vsubss4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSUBSS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSUBSS4-NEXT:   sycl::ext::intel::math::vsubss4(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsubus2 | FileCheck %s -check-prefix=VSUBUS2
// VSUBUS2: CUDA API:
// VSUBUS2-NEXT:   __vsubus2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSUBUS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSUBUS2-NEXT:   sycl::ext::intel::math::vsubus2(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsubus4 | FileCheck %s -check-prefix=VSUBUS4
// VSUBUS4: CUDA API:
// VSUBUS4-NEXT:   __vsubus4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSUBUS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSUBUS4-NEXT:   sycl::ext::intel::math::vsubus4(u1, u2);
