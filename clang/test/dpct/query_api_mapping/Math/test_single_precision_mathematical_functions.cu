// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

/// Single Precision Mathematical Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=acosf | FileCheck %s -check-prefix=ACOSF
// ACOSF: CUDA API:
// ACOSF-NEXT:   acosf(f /*float*/);
// ACOSF-NEXT: Is migrated to:
// ACOSF-NEXT:   sycl::acos(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=acoshf | FileCheck %s -check-prefix=ACOSHF
// ACOSHF: CUDA API:
// ACOSHF-NEXT:   acoshf(f /*float*/);
// ACOSHF-NEXT: Is migrated to:
// ACOSHF-NEXT:   sycl::acosh(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=asinf | FileCheck %s -check-prefix=ASINF
// ASINF: CUDA API:
// ASINF-NEXT:   asinf(f /*float*/);
// ASINF-NEXT: Is migrated to:
// ASINF-NEXT:   sycl::asin(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=asinhf | FileCheck %s -check-prefix=ASINHF
// ASINHF: CUDA API:
// ASINHF-NEXT:   asinhf(f /*float*/);
// ASINHF-NEXT: Is migrated to:
// ASINHF-NEXT:   sycl::asinh(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atan2f | FileCheck %s -check-prefix=ATAN2F
// ATAN2F: CUDA API:
// ATAN2F-NEXT:   atan2f(f1 /*float*/, f2 /*float*/);
// ATAN2F-NEXT: Is migrated to:
// ATAN2F-NEXT:   sycl::atan2(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atanf | FileCheck %s -check-prefix=ATANF
// ATANF: CUDA API:
// ATANF-NEXT:   atanf(f /*float*/);
// ATANF-NEXT: Is migrated to:
// ATANF-NEXT:   sycl::atan(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atanhf | FileCheck %s -check-prefix=ATANHF
// ATANHF: CUDA API:
// ATANHF-NEXT:   atanhf(f /*float*/);
// ATANHF-NEXT: Is migrated to:
// ATANHF-NEXT:   sycl::atanh(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cbrtf | FileCheck %s -check-prefix=CBRTF
// CBRTF: CUDA API:
// CBRTF-NEXT:   cbrtf(f /*float*/);
// CBRTF-NEXT: Is migrated to:
// CBRTF-NEXT:   sycl::cbrt(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ceilf | FileCheck %s -check-prefix=CEILF
// CEILF: CUDA API:
// CEILF-NEXT:   ceilf(f /*float*/);
// CEILF-NEXT: Is migrated to:
// CEILF-NEXT:   sycl::ceil(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=copysignf | FileCheck %s -check-prefix=COPYSIGNF
// COPYSIGNF: CUDA API:
// COPYSIGNF-NEXT:   copysignf(f1 /*float*/, f2 /*float*/);
// COPYSIGNF-NEXT: Is migrated to:
// COPYSIGNF-NEXT:   sycl::copysign(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cosf | FileCheck %s -check-prefix=COSF
// COSF: CUDA API:
// COSF-NEXT:   cosf(f /*float*/);
// COSF-NEXT: Is migrated to:
// COSF-NEXT:   sycl::cos(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=coshf | FileCheck %s -check-prefix=COSHF
// COSHF: CUDA API:
// COSHF-NEXT:   coshf(f /*float*/);
// COSHF-NEXT: Is migrated to:
// COSHF-NEXT:   sycl::cosh(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cospif | FileCheck %s -check-prefix=COSPIF
// COSPIF: CUDA API:
// COSPIF-NEXT:   cospif(f /*float*/);
// COSPIF-NEXT: Is migrated to:
// COSPIF-NEXT:   sycl::cospi(f);

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=erff | FileCheck %s -check-prefix=ERFF
// ERFF: CUDA API:
// ERFF-NEXT:   erff(f /*float*/);
// ERFF-NEXT: Is migrated to:
// ERFF-NEXT:   sycl::erf(f);

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=exp2f | FileCheck %s -check-prefix=EXP2F
// EXP2F: CUDA API:
// EXP2F-NEXT:   exp2f(f /*float*/);
// EXP2F-NEXT: Is migrated to:
// EXP2F-NEXT:   sycl::exp2(f);

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fabsf | FileCheck %s -check-prefix=FABSF
// FABSF: CUDA API:
// FABSF-NEXT:   fabsf(f /*float*/);
// FABSF-NEXT: Is migrated to:
// FABSF-NEXT:   sycl::fabs(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fdimf | FileCheck %s -check-prefix=FDIMF
// FDIMF: CUDA API:
// FDIMF-NEXT:   fdimf(f1 /*float*/, f2 /*float*/);
// FDIMF-NEXT: Is migrated to:
// FDIMF-NEXT:   sycl::fdim(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fdividef | FileCheck %s -check-prefix=FDIVIDEF
// FDIVIDEF: CUDA API:
// FDIVIDEF-NEXT:   fdividef(f1 /*float*/, f2 /*float*/);
// FDIVIDEF-NEXT: Is migrated to:
// FDIVIDEF-NEXT:   f1 / f2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=floorf | FileCheck %s -check-prefix=FLOORF
// FLOORF: CUDA API:
// FLOORF-NEXT:   floorf(f /*float*/);
// FLOORF-NEXT: Is migrated to:
// FLOORF-NEXT:   sycl::floor(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fmaf | FileCheck %s -check-prefix=FMAF
// FMAF: CUDA API:
// FMAF-NEXT:   fmaf(f1 /*float*/, f2 /*float*/, f3 /*float*/);
// FMAF-NEXT: Is migrated to:
// FMAF-NEXT:   sycl::fma(f1, f2, f3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fmaxf | FileCheck %s -check-prefix=FMAXF
// FMAXF: CUDA API:
// FMAXF-NEXT:   fmaxf(f1 /*float*/, f2 /*float*/);
// FMAXF-NEXT: Is migrated to:
// FMAXF-NEXT:   sycl::fmax(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fminf | FileCheck %s -check-prefix=FMINF
// FMINF: CUDA API:
// FMINF-NEXT:   fminf(f1 /*float*/, f2 /*float*/);
// FMINF-NEXT: Is migrated to:
// FMINF-NEXT:   sycl::fmin(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fmodf | FileCheck %s -check-prefix=FMODF
// FMODF: CUDA API:
// FMODF-NEXT:   fmodf(f1 /*float*/, f2 /*float*/);
// FMODF-NEXT: Is migrated to:
// FMODF-NEXT:   sycl::fmod(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=frexpf | FileCheck %s -check-prefix=FREXPF
// FREXPF: CUDA API:
// FREXPF-NEXT:   frexpf(f /*float*/, pi /*int **/);
// FREXPF-NEXT: Is migrated to:
// FREXPF-NEXT:   sycl::frexp(f, sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::yes>(pi));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hypotf | FileCheck %s -check-prefix=HYPOTF
// HYPOTF: CUDA API:
// HYPOTF-NEXT:   hypotf(f1 /*float*/, f2 /*float*/);
// HYPOTF-NEXT: Is migrated to:
// HYPOTF-NEXT:   sycl::hypot(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ilogbf | FileCheck %s -check-prefix=ILOGBF
// ILOGBF: CUDA API:
// ILOGBF-NEXT:   ilogbf(f /*float*/);
// ILOGBF-NEXT: Is migrated to:
// ILOGBF-NEXT:   sycl::ilogb(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=isfinite | FileCheck %s -check-prefix=ISFINITE
// ISFINITE: CUDA API:
// ISFINITE-NEXT:   isfinite(f /*float*/);
// ISFINITE-NEXT:   isfinite(d /*double*/);
// ISFINITE-NEXT: Is migrated to:
// ISFINITE-NEXT:   sycl::isfinite(f);
// ISFINITE-NEXT:   sycl::isfinite(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=isinf | FileCheck %s -check-prefix=ISINF
// ISINF: CUDA API:
// ISINF-NEXT:   isinf(f /*float*/);
// ISINF-NEXT:   isinf(d /*double*/);
// ISINF-NEXT: Is migrated to:
// ISINF-NEXT:   sycl::isinf(f);
// ISINF-NEXT:   sycl::isinf(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=isnan | FileCheck %s -check-prefix=ISNAN
// ISNAN: CUDA API:
// ISNAN-NEXT:   isnan(f /*float*/);
// ISNAN-NEXT:   isnan(d /*double*/);
// ISNAN-NEXT: Is migrated to:
// ISNAN-NEXT:   sycl::isnan(f);
// ISNAN-NEXT:   sycl::isnan(d);

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=llrintf | FileCheck %s -check-prefix=LLRINTF
// LLRINTF: CUDA API:
// LLRINTF-NEXT:   llrintf(f /*float*/);
// LLRINTF-NEXT: Is migrated to:
// LLRINTF-NEXT:   sycl::rint(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=llroundf | FileCheck %s -check-prefix=LLROUNDF
// LLROUNDF: CUDA API:
// LLROUNDF-NEXT:   llroundf(f /*float*/);
// LLROUNDF-NEXT: Is migrated to:
// LLROUNDF-NEXT:   sycl::round(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=log10f | FileCheck %s -check-prefix=LOG10F
// LOG10F: CUDA API:
// LOG10F-NEXT:   log10f(f /*float*/);
// LOG10F-NEXT: Is migrated to:
// LOG10F-NEXT:   sycl::log10(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=log1pf | FileCheck %s -check-prefix=LOG1PF
// LOG1PF: CUDA API:
// LOG1PF-NEXT:   log1pf(f /*float*/);
// LOG1PF-NEXT: Is migrated to:
// LOG1PF-NEXT:   sycl::log1p(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=log2f | FileCheck %s -check-prefix=LOG2F
// LOG2F: CUDA API:
// LOG2F-NEXT:   log2f(f /*float*/);
// LOG2F-NEXT: Is migrated to:
// LOG2F-NEXT:   sycl::log2(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=logbf | FileCheck %s -check-prefix=LOGBF
// LOGBF: CUDA API:
// LOGBF-NEXT:   logbf(f /*float*/);
// LOGBF-NEXT: Is migrated to:
// LOGBF-NEXT:   sycl::logb(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=logf | FileCheck %s -check-prefix=LOGF
// LOGF: CUDA API:
// LOGF-NEXT:   logf(f /*float*/);
// LOGF-NEXT: Is migrated to:
// LOGF-NEXT:   sycl::log(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=lrintf | FileCheck %s -check-prefix=LRINTF
// LRINTF: CUDA API:
// LRINTF-NEXT:   lrintf(f /*float*/);
// LRINTF-NEXT: Is migrated to:
// LRINTF-NEXT:   sycl::rint(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=lroundf | FileCheck %s -check-prefix=LROUNDF
// LROUNDF: CUDA API:
// LROUNDF-NEXT:   lroundf(f /*float*/);
// LROUNDF-NEXT: Is migrated to:
// LROUNDF-NEXT:   sycl::round(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=modff | FileCheck %s -check-prefix=MODFF
// MODFF: CUDA API:
// MODFF-NEXT:   modff(f /*float*/, pf /*float **/);
// MODFF-NEXT: Is migrated to:
// MODFF-NEXT:   sycl::modf(f, sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::yes>(pf));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nanf | FileCheck %s -check-prefix=NANF
// NANF: CUDA API:
// NANF-NEXT:   nanf(pc /*const char **/);
// NANF-NEXT: Is migrated to:
// NANF-NEXT:   sycl::nan(0u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nearbyintf | FileCheck %s -check-prefix=NEARBYINTF
// NEARBYINTF: CUDA API:
// NEARBYINTF-NEXT:   nearbyintf(f /*float*/);
// NEARBYINTF-NEXT: Is migrated to:
// NEARBYINTF-NEXT:   sycl::floor(f + 0.5);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nextafterf | FileCheck %s -check-prefix=NEXTAFTERF
// NEXTAFTERF: CUDA API:
// NEXTAFTERF-NEXT:   nextafterf(f1 /*float*/, f2 /*float*/);
// NEXTAFTERF-NEXT: Is migrated to:
// NEXTAFTERF-NEXT:   sycl::nextafter(f1, f2);

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
// POWF-NEXT:   dpct::pow(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rcbrtf | FileCheck %s -check-prefix=RCBRTF
// RCBRTF: CUDA API:
// RCBRTF-NEXT:   rcbrtf(f /*float*/);
// RCBRTF-NEXT: Is migrated to:
// RCBRTF-NEXT:   sycl::native::recip(dpct::cbrt<float>(f));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=remainderf | FileCheck %s -check-prefix=REMAINDERF
// REMAINDERF: CUDA API:
// REMAINDERF-NEXT:   remainderf(f1 /*float*/, f2 /*float*/);
// REMAINDERF-NEXT: Is migrated to:
// REMAINDERF-NEXT:   sycl::remainder(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=remquof | FileCheck %s -check-prefix=REMQUOF
// REMQUOF: CUDA API:
// REMQUOF-NEXT:   remquof(f1 /*float*/, f2 /*float*/, pi /*int **/);
// REMQUOF-NEXT: Is migrated to:
// REMQUOF-NEXT:   sycl::remquo(f1, f2, sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::yes>(pi));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rhypotf | FileCheck %s -check-prefix=RHYPOTF
// RHYPOTF: CUDA API:
// RHYPOTF-NEXT:   rhypotf(f1 /*float*/, f2 /*float*/);
// RHYPOTF-NEXT: Is migrated to:
// RHYPOTF-NEXT:   1 / sycl::hypot(f1, f2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rintf | FileCheck %s -check-prefix=RINTF
// RINTF: CUDA API:
// RINTF-NEXT:   rintf(f /*float*/);
// RINTF-NEXT: Is migrated to:
// RINTF-NEXT:   sycl::rint(f);

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rsqrtf | FileCheck %s -check-prefix=RSQRTF
// RSQRTF: CUDA API:
// RSQRTF-NEXT:   rsqrtf(f /*float*/);
// RSQRTF-NEXT: Is migrated to:
// RSQRTF-NEXT:   sycl::rsqrt(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=scalblnf | FileCheck %s -check-prefix=SCALBLNF
// SCALBLNF: CUDA API:
// SCALBLNF-NEXT:   scalblnf(f /*float*/, l /*long int*/);
// SCALBLNF-NEXT: Is migrated to:
// SCALBLNF-NEXT:   f*(2<<l);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=scalbnf | FileCheck %s -check-prefix=SCALBNF
// SCALBNF: CUDA API:
// SCALBNF-NEXT:   scalbnf(f /*float*/, i /*int*/);
// SCALBNF-NEXT: Is migrated to:
// SCALBNF-NEXT:   f*(2<<i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=signbit | FileCheck %s -check-prefix=SIGNBIT
// SIGNBIT: CUDA API:
// SIGNBIT-NEXT:   signbit(f /*float*/);
// SIGNBIT-NEXT:   signbit(d /*double*/);
// SIGNBIT-NEXT: Is migrated to:
// SIGNBIT-NEXT:   sycl::signbit(f);
// SIGNBIT-NEXT:   sycl::signbit(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sincosf | FileCheck %s -check-prefix=SINCOSF
// SINCOSF: CUDA API:
// SINCOSF-NEXT:   sincosf(f /*float*/, pf1 /*float **/, pf2 /*float **/);
// SINCOSF-NEXT: Is migrated to:
// SINCOSF-NEXT:   *pf1 = sycl::sincos(f, sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::yes>(pf2));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sincospif | FileCheck %s -check-prefix=SINCOSPIF
// SINCOSPIF: CUDA API:
// SINCOSPIF-NEXT:   sincospif(f /*float*/, pf1 /*float **/, pf2 /*float **/);
// SINCOSPIF-NEXT: Is migrated to:
// SINCOSPIF-NEXT:   *pf1 = sycl::sincos(f * DPCT_PI_F, sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::yes>(pf2));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sinf | FileCheck %s -check-prefix=SINF
// SINF: CUDA API:
// SINF-NEXT:   sinf(f /*float*/);
// SINF-NEXT: Is migrated to:
// SINF-NEXT:   sycl::sin(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sinhf | FileCheck %s -check-prefix=SINHF
// SINHF: CUDA API:
// SINHF-NEXT:   sinhf(f /*float*/);
// SINHF-NEXT: Is migrated to:
// SINHF-NEXT:   sycl::sinh(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sinpif | FileCheck %s -check-prefix=SINPIF
// SINPIF: CUDA API:
// SINPIF-NEXT:   sinpif(f /*float*/);
// SINPIF-NEXT: Is migrated to:
// SINPIF-NEXT:   sycl::sinpi(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sqrtf | FileCheck %s -check-prefix=SQRTF
// SQRTF: CUDA API:
// SQRTF-NEXT:   sqrtf(f /*float*/);
// SQRTF-NEXT: Is migrated to:
// SQRTF-NEXT:   sycl::sqrt(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tanf | FileCheck %s -check-prefix=TANF
// TANF: CUDA API:
// TANF-NEXT:   tanf(f /*float*/);
// TANF-NEXT: Is migrated to:
// TANF-NEXT:   sycl::tan(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tanhf | FileCheck %s -check-prefix=TANHF
// TANHF: CUDA API:
// TANHF-NEXT:   tanhf(f /*float*/);
// TANHF-NEXT: Is migrated to:
// TANHF-NEXT:   sycl::tanh(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tgammaf | FileCheck %s -check-prefix=TGAMMAF
// TGAMMAF: CUDA API:
// TGAMMAF-NEXT:   tgammaf(f /*float*/);
// TGAMMAF-NEXT: Is migrated to:
// TGAMMAF-NEXT:   sycl::tgamma(f);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=truncf | FileCheck %s -check-prefix=TRUNCF
// TRUNCF: CUDA API:
// TRUNCF-NEXT:   truncf(f /*float*/);
// TRUNCF-NEXT: Is migrated to:
// TRUNCF-NEXT:   sycl::trunc(f);

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
