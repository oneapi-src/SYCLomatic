//===--------------- HeaderTypes.inc --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HEADER
#define HEADER(X, Y)
#endif

#ifndef STD_HEADER
#define STD_HEADER(X, Y) HEADER(X, Y)
#endif

#ifndef ONEDPL_HEADER
#define ONEDPL_HEADER(X, Y) HEADER(DPL_ ## X, Y)
#endif

#ifndef MKL_HEADER
#define MKL_HEADER(X, Y) HEADER(MKL_ ## X, Y)
#endif

#ifndef DPCT_HEADER
#define DPCT_HEADER(X, Y) HEADER(DPCT_ ## X, Y)
#endif

STD_HEADER(SYCL, "<sycl/sycl.hpp>")
STD_HEADER(Math, "<cmath>")
STD_HEADER(Algorithm, "<algorithm>")
STD_HEADER(Complex, "<complex>")
STD_HEADER(Thread, "<thread>")
STD_HEADER(Future, "<future>")
STD_HEADER(Time, "<time.h>")
STD_HEADER(Stdlib, "<cstdlib>")
STD_HEADER(Chrono, "<chrono>")
STD_HEADER(Limits, "<limits>")
STD_HEADER(Numeric, "<numeric>")
#ifdef _WIN32
STD_HEADER(DL, "<libloaderapi.h>")
#else
STD_HEADER(DL, "<dlfcn.h>")
#endif

ONEDPL_HEADER(Algorithm, "<oneapi/dpl/algorithm>")
ONEDPL_HEADER(Execution, "<oneapi/dpl/execution>")
ONEDPL_HEADER(Iterator, "<oneapi/dpl/iterator>")

MKL_HEADER(Mkl, "<oneapi/mkl.hpp>")
MKL_HEADER(RNG, "<oneapi/mkl/rng/device.hpp>")
MKL_HEADER(BFloat16, "<oneapi/mkl/bfloat16.hpp>")

DPCT_HEADER(Dpct, "<dpct/dpct.hpp>")
DPCT_HEADER(DPL_Utils, "<dpct/dpl_utils.hpp>")
DPCT_HEADER(FFT_Utils, "<dpct/fft_utils.hpp>")
DPCT_HEADER(RNG_Utils, "<dpct/rng_utils.hpp>")
DPCT_HEADER(CCL_Utils, "<dpct/ccl_utils.hpp>")
DPCT_HEADER(BLAS_Utils, "<dpct/blas_utils.hpp>")
DPCT_HEADER(DNNL_Utils, "<dpct/dnnl_utils.hpp>")
DPCT_HEADER(LAPACK_Utils, "<dpct/lapack_utils.hpp>")
DPCT_HEADER(COMMON_Utils, "<dpct/lib_common_utils.hpp>")
DPCT_HEADER(Atomic, "<dpct/atomic.hpp>")

#undef MKL_HEADER
#undef ONEDPL_HEADER
#undef STD_HEADER
#undef HEADER
