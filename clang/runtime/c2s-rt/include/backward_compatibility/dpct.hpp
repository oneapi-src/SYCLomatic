//==---- dpct.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __DPCT_HPP__
#define __DPCT_HPP__

#warning This file is deprecated. Please use "include/c2s/c2s.hpp" instead.

#ifdef DPCT_USM_LEVEL_NONE
#define C2S_USM_LEVEL_NONE
#endif
#ifdef __USE_DPCT
#define __USE_C2S
#endif

#include "../c2s/c2s.hpp"

namespace dpct {
using namespace c2s;
}

#ifndef __dpct_align__
#define __dpct_align__              __c2s_align__
#endif
#ifndef __dpct_inline__
#define __dpct_inline__             __c2s_inline__
#endif
#ifndef __dpct_noinline__
#define __dpct_noinline__           __c2s_noinline__
#endif
#ifndef DPCT_COMPATIBILITY_TEMP
#define DPCT_COMPATIBILITY_TEMP     C2S_COMPATIBILITY_TEMP
#endif
#ifndef DPCT_PI_F
#define DPCT_PI_F                   C2S_PI_F
#endif
#ifndef DPCT_PI
#define DPCT_PI                     C2S_PI
#endif

#ifndef dpct_malloc
#define dpct_malloc                 c2s_malloc
#endif
#ifndef dpct_memset
#define dpct_memset                 c2s_memset
#endif
#ifndef async_dpct_memset
#define async_dpct_memset           async_c2s_memset
#endif
#ifndef dpct_memcpy
#define dpct_memcpy                 c2s_memcpy
#endif
#ifndef async_dpct_memcpy
#define async_dpct_memcpy           async_c2s_memcpy
#endif
#ifndef dpct_free
#define dpct_free                   c2s_free
#endif
#ifndef async_dpct_free
#define async_dpct_free             async_c2s_free
#endif
#ifndef dpct_kernel_name
#define dpct_kernel_name            c2s_kernel_name
#endif
#ifndef dpct_kernel_scalar
#define dpct_kernel_scalar          c2s_kernel_scalar
#endif

#endif // __DPCT_HPP__
