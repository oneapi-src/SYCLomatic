// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0

/// Double Precision Mathematical Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=acos | FileCheck %s -check-prefix=ACOS
// ACOS: CUDA API:
// ACOS-NEXT:   acos(d /*double*/);
// ACOS-NEXT: Is migrated to:
// ACOS-NEXT:   sycl::acos(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=acosh | FileCheck %s -check-prefix=ACOSH
// ACOSH: CUDA API:
// ACOSH-NEXT:   acosh(d /*double*/);
// ACOSH-NEXT: Is migrated to:
// ACOSH-NEXT:   sycl::acosh(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=asin | FileCheck %s -check-prefix=ASIN
// ASIN: CUDA API:
// ASIN-NEXT:   asin(d /*double*/);
// ASIN-NEXT: Is migrated to:
// ASIN-NEXT:   sycl::asin(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=asinh | FileCheck %s -check-prefix=ASINH
// ASINH: CUDA API:
// ASINH-NEXT:   asinh(d /*double*/);
// ASINH-NEXT: Is migrated to:
// ASINH-NEXT:   sycl::asinh(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atan | FileCheck %s -check-prefix=ATAN
// ATAN: CUDA API:
// ATAN-NEXT:   atan(d /*double*/);
// ATAN-NEXT: Is migrated to:
// ATAN-NEXT:   sycl::atan(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atan2 | FileCheck %s -check-prefix=ATAN2
// ATAN2: CUDA API:
// ATAN2-NEXT:   atan2(d1 /*double*/, d2 /*double*/);
// ATAN2-NEXT: Is migrated to:
// ATAN2-NEXT:   sycl::atan2(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=atanh | FileCheck %s -check-prefix=ATANH
// ATANH: CUDA API:
// ATANH-NEXT:   atanh(d /*double*/);
// ATANH-NEXT: Is migrated to:
// ATANH-NEXT:   sycl::atanh(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cbrt | FileCheck %s -check-prefix=CBRT
// CBRT: CUDA API:
// CBRT-NEXT:   cbrt(d /*double*/);
// CBRT-NEXT: Is migrated to:
// CBRT-NEXT:   sycl::cbrt(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ceil | FileCheck %s -check-prefix=CEIL
// CEIL: CUDA API:
// CEIL-NEXT:   ceil(d /*double*/);
// CEIL-NEXT: Is migrated to:
// CEIL-NEXT:   sycl::ceil(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=copysign | FileCheck %s -check-prefix=COPYSIGN
// COPYSIGN: CUDA API:
// COPYSIGN-NEXT:   copysign(d1 /*double*/, d2 /*double*/);
// COPYSIGN-NEXT: Is migrated to:
// COPYSIGN-NEXT:   sycl::copysign(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cos | FileCheck %s -check-prefix=COS
// COS: CUDA API:
// COS-NEXT:   cos(d /*double*/);
// COS-NEXT: Is migrated to:
// COS-NEXT:   sycl::cos(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cosh | FileCheck %s -check-prefix=COSH
// COSH: CUDA API:
// COSH-NEXT:   cosh(d /*double*/);
// COSH-NEXT: Is migrated to:
// COSH-NEXT:   sycl::cosh(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cospi | FileCheck %s -check-prefix=COSPI
// COSPI: CUDA API:
// COSPI-NEXT:   cospi(d /*double*/);
// COSPI-NEXT: Is migrated to:
// COSPI-NEXT:   sycl::cospi(d);

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=erfc | FileCheck %s -check-prefix=ERFC
// ERFC: CUDA API:
// ERFC-NEXT:   erfc(d /*double*/);
// ERFC-NEXT: Is migrated to:
// ERFC-NEXT:   sycl::erfc(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=erfcinv | FileCheck %s -check-prefix=ERFCINV
// ERFCINV: CUDA API:
// ERFCINV-NEXT:   erfcinv(d /*double*/);
// ERFCINV-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// ERFCINV-NEXT:   sycl::ext::intel::math::erfcinv(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=erfcx | FileCheck %s -check-prefix=ERFCX
// ERFCX: CUDA API:
// ERFCX-NEXT:   erfcx(d /*double*/);
// ERFCX-NEXT: Is migrated to:
// ERFCX-NEXT:   sycl::exp(d * d) * sycl::erfc(d);

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=exp2 | FileCheck %s -check-prefix=EXP2
// EXP2: CUDA API:
// EXP2-NEXT:   exp2(d /*double*/);
// EXP2-NEXT: Is migrated to:
// EXP2-NEXT:   sycl::exp2(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=expm1 | FileCheck %s -check-prefix=EXPM1
// EXPM1: CUDA API:
// EXPM1-NEXT:   expm1(d /*double*/);
// EXPM1-NEXT: Is migrated to:
// EXPM1-NEXT:   sycl::expm1(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fabs | FileCheck %s -check-prefix=FABS
// FABS: CUDA API:
// FABS-NEXT:   fabs(d /*double*/);
// FABS-NEXT: Is migrated to:
// FABS-NEXT:   sycl::fabs(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fdim | FileCheck %s -check-prefix=FDIM
// FDIM: CUDA API:
// FDIM-NEXT:   fdim(d1 /*double*/, d2 /*double*/);
// FDIM-NEXT: Is migrated to:
// FDIM-NEXT:   sycl::fdim(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=floor | FileCheck %s -check-prefix=FLOOR
// FLOOR: CUDA API:
// FLOOR-NEXT:   floor(d /*double*/);
// FLOOR-NEXT: Is migrated to:
// FLOOR-NEXT:   sycl::floor(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fma | FileCheck %s -check-prefix=FMA
// FMA: CUDA API:
// FMA-NEXT:   fma(d1 /*double*/, d2 /*double*/, d3 /*double*/);
// FMA-NEXT: Is migrated to:
// FMA-NEXT:   sycl::fma(d1, d2, d3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fmax | FileCheck %s -check-prefix=FMAX
// FMAX: CUDA API:
// FMAX-NEXT:   fmax(d1 /*double*/, d2 /*double*/);
// FMAX-NEXT: Is migrated to:
// FMAX-NEXT:   sycl::fmax(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fmin | FileCheck %s -check-prefix=FMIN
// FMIN: CUDA API:
// FMIN-NEXT:   fmin(d1 /*double*/, d2 /*double*/);
// FMIN-NEXT: Is migrated to:
// FMIN-NEXT:   sycl::fmin(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=fmod | FileCheck %s -check-prefix=FMOD
// FMOD: CUDA API:
// FMOD-NEXT:   fmod(d1 /*double*/, d2 /*double*/);
// FMOD-NEXT: Is migrated to:
// FMOD-NEXT:   sycl::fmod(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=frexp | FileCheck %s -check-prefix=FREXP
// FREXP: CUDA API:
// FREXP-NEXT:   frexp(d /*double*/, pi /*int **/);
// FREXP-NEXT: Is migrated to:
// FREXP-NEXT:   sycl::frexp(d, sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::yes>(pi));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=hypot | FileCheck %s -check-prefix=HYPOT
// HYPOT: CUDA API:
// HYPOT-NEXT:   hypot(d1 /*double*/, d2 /*double*/);
// HYPOT-NEXT: Is migrated to:
// HYPOT-NEXT:   sycl::hypot(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ilogb | FileCheck %s -check-prefix=ILOGB
// ILOGB: CUDA API:
// ILOGB-NEXT:   ilogb(d /*double*/);
// ILOGB-NEXT: Is migrated to:
// ILOGB-NEXT:   sycl::ilogb(d);

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=llrint | FileCheck %s -check-prefix=LLRINT
// LLRINT: CUDA API:
// LLRINT-NEXT:   llrint(d /*double*/);
// LLRINT-NEXT: Is migrated to:
// LLRINT-NEXT:   sycl::rint(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=llround | FileCheck %s -check-prefix=LLROUND
// LLROUND: CUDA API:
// LLROUND-NEXT:   llround(d /*double*/);
// LLROUND-NEXT: Is migrated to:
// LLROUND-NEXT:   sycl::round(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=log | FileCheck %s -check-prefix=LOG
// LOG: CUDA API:
// LOG-NEXT:   log(d /*double*/);
// LOG-NEXT: Is migrated to:
// LOG-NEXT:   sycl::log(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=log10 | FileCheck %s -check-prefix=LOG10
// LOG10: CUDA API:
// LOG10-NEXT:   log10(d /*double*/);
// LOG10-NEXT: Is migrated to:
// LOG10-NEXT:   sycl::log10(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=log1p | FileCheck %s -check-prefix=LOG1P
// LOG1P: CUDA API:
// LOG1P-NEXT:   log1p(d /*double*/);
// LOG1P-NEXT: Is migrated to:
// LOG1P-NEXT:   sycl::log1p(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=log2 | FileCheck %s -check-prefix=LOG2
// LOG2: CUDA API:
// LOG2-NEXT:   log2(d /*double*/);
// LOG2-NEXT: Is migrated to:
// LOG2-NEXT:   sycl::log2(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=logb | FileCheck %s -check-prefix=LOGB
// LOGB: CUDA API:
// LOGB-NEXT:   logb(d /*double*/);
// LOGB-NEXT: Is migrated to:
// LOGB-NEXT:   sycl::logb(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=lrint | FileCheck %s -check-prefix=LRINT
// LRINT: CUDA API:
// LRINT-NEXT:   lrint(d /*double*/);
// LRINT-NEXT: Is migrated to:
// LRINT-NEXT:   sycl::rint(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=lround | FileCheck %s -check-prefix=LROUND
// LROUND: CUDA API:
// LROUND-NEXT:   lround(d /*double*/);
// LROUND-NEXT: Is migrated to:
// LROUND-NEXT:   sycl::round(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=modf | FileCheck %s -check-prefix=MODF
// MODF: CUDA API:
// MODF-NEXT:   modf(d /*double*/, pd /*double **/);
// MODF-NEXT: Is migrated to:
// MODF-NEXT:   sycl::modf(d, sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::yes>(pd));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nan | FileCheck %s -check-prefix=NAN
// NAN: CUDA API:
// NAN-NEXT:   nan(pc /*const char **/);
// NAN-NEXT: Is migrated to:
// NAN-NEXT:   sycl::nan(0u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nearbyint | FileCheck %s -check-prefix=NEARBYINT
// NEARBYINT: CUDA API:
// NEARBYINT-NEXT:   nearbyint(d /*double*/);
// NEARBYINT-NEXT: Is migrated to:
// NEARBYINT-NEXT:   sycl::floor(d + 0.5);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nextafter | FileCheck %s -check-prefix=NEXTAFTER
// NEXTAFTER: CUDA API:
// NEXTAFTER-NEXT:   nextafter(d1 /*double*/, d2 /*double*/);
// NEXTAFTER-NEXT: Is migrated to:
// NEXTAFTER-NEXT:   sycl::nextafter(d1, d2);

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
// POW-NEXT:   dpct::pow(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rcbrt | FileCheck %s -check-prefix=RCBRT
// RCBRT: CUDA API:
// RCBRT-NEXT:   rcbrt(d /*double*/);
// RCBRT-NEXT: Is migrated to:
// RCBRT-NEXT:   1 / dpct::cbrt<double>(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=remainder | FileCheck %s -check-prefix=REMAINDER
// REMAINDER: CUDA API:
// REMAINDER-NEXT:   remainder(d1 /*double*/, d2 /*double*/);
// REMAINDER-NEXT: Is migrated to:
// REMAINDER-NEXT:   sycl::remainder(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=remquo | FileCheck %s -check-prefix=REMQUO
// REMQUO: CUDA API:
// REMQUO-NEXT:   remquo(d1 /*double*/, d2 /*double*/, pi /*int **/);
// REMQUO-NEXT: Is migrated to:
// REMQUO-NEXT:   sycl::remquo(d1, d2, sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::yes>(pi));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rhypot | FileCheck %s -check-prefix=RHYPOT
// RHYPOT: CUDA API:
// RHYPOT-NEXT:   rhypot(d1 /*double*/, d2 /*double*/);
// RHYPOT-NEXT: Is migrated to:
// RHYPOT-NEXT:   1 / sycl::hypot(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=rint | FileCheck %s -check-prefix=RINT
// RINT: CUDA API:
// RINT-NEXT:   rint(d /*double*/);
// RINT-NEXT: Is migrated to:
// RINT-NEXT:   sycl::rint(d);

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=scalbn | FileCheck %s -check-prefix=SCALBN
// SCALBN: CUDA API:
// SCALBN-NEXT:   scalbn(d /*double*/, i /*int*/);
// SCALBN-NEXT: Is migrated to:
// SCALBN-NEXT:   d*(2<<i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sin | FileCheck %s -check-prefix=SIN
// SIN: CUDA API:
// SIN-NEXT:   sin(d /*double*/);
// SIN-NEXT: Is migrated to:
// SIN-NEXT:   sycl::sin(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sincos | FileCheck %s -check-prefix=SINCOS
// SINCOS: CUDA API:
// SINCOS-NEXT:   sincos(d /*double*/, pd1 /*double **/, pd2 /*double **/);
// SINCOS-NEXT: Is migrated to:
// SINCOS-NEXT:   *pd1 = sycl::sincos(d, sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::yes>(pd2));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sincospi | FileCheck %s -check-prefix=SINCOSPI
// SINCOSPI: CUDA API:
// SINCOSPI-NEXT:   sincospi(d /*double*/, pd1 /*double **/, pd2 /*double **/);
// SINCOSPI-NEXT: Is migrated to:
// SINCOSPI-NEXT:   *pd1 = sycl::sincos(d * DPCT_PI, sycl::address_space_cast<sycl::access::address_space::generic_space, sycl::access::decorated::yes>(pd2));

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sinh | FileCheck %s -check-prefix=SINH
// SINH: CUDA API:
// SINH-NEXT:   sinh(d /*double*/);
// SINH-NEXT: Is migrated to:
// SINH-NEXT:   sycl::sinh(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sinpi | FileCheck %s -check-prefix=SINPI
// SINPI: CUDA API:
// SINPI-NEXT:   sinpi(d /*double*/);
// SINPI-NEXT: Is migrated to:
// SINPI-NEXT:   sycl::sinpi(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=sqrt | FileCheck %s -check-prefix=SQRT
// SQRT: CUDA API:
// SQRT-NEXT:   sqrt(d /*double*/);
// SQRT-NEXT: Is migrated to:
// SQRT-NEXT:   sycl::sqrt(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tan | FileCheck %s -check-prefix=TAN
// TAN: CUDA API:
// TAN-NEXT:   tan(d /*double*/);
// TAN-NEXT: Is migrated to:
// TAN-NEXT:   sycl::tan(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tanh | FileCheck %s -check-prefix=TANH
// TANH: CUDA API:
// TANH-NEXT:   tanh(d /*double*/);
// TANH-NEXT: Is migrated to:
// TANH-NEXT:   sycl::tanh(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=tgamma | FileCheck %s -check-prefix=TGAMMA
// TGAMMA: CUDA API:
// TGAMMA-NEXT:   tgamma(d /*double*/);
// TGAMMA-NEXT: Is migrated to:
// TGAMMA-NEXT:   sycl::tgamma(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=trunc | FileCheck %s -check-prefix=TRUNC
// TRUNC: CUDA API:
// TRUNC-NEXT:   trunc(d /*double*/);
// TRUNC-NEXT: Is migrated to:
// TRUNC-NEXT:   sycl::trunc(d);

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
