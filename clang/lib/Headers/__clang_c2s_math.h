/*===---- __clang_c2s_math.h -----------------------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __CLANG_C2S_MATH_H__
#define __CLANG_C2S_MATH_H__

float max(float a, float b);
int min(int a, int b);

#if (defined(_WIN64) || defined(_WIN32))

#else
#if !defined(_GLIBCXX_RELEASE) ||                                              \
    defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE < 7
int signbit(float x);
int signbit(double x);
int signbit(long double x);
int isfinite(float x);
int isfinite(double x);
int isfinite(long double x);
float modf(float a, float *pb);
#endif
#endif

float powif(float a, int b);
double powi(double a, int b);
float norm3d(float a, float b, float c);
float norm4d(float a, float b, float c, float d);
void sincos(float a, float *pb, float *pc);
float cyl_bessel_i0(float a);
float cyl_bessel_i1(float a);

__host__ double atomicAdd(double *pa, double b);
__host__ unsigned long long atomicAdd(unsigned long long *pa, unsigned long long b);
__host__ long atomicAdd(long *pa, long b);
__host__ unsigned long atomicAdd(unsigned long *pa, unsigned long b);
__host__ int atomicAdd(int *pa, int b);
__host__ unsigned int atomicAdd(unsigned int *pa, unsigned int b);

/// ---Fix for Window CUDA10.1--------------------------------
#if CUDA_VERSION >= 10000 && (defined(_WIN64) || defined(_WIN32))
extern __host__ __device__ unsigned
cudaConfigureCall(dim3 a, dim3 b, size_t c = 0, void *pd = 0);
#endif

/// ---Fix for Windows CUDA >=9---------------------------------
#if (CUDA_VERSION >= 9000) && (defined(_WIN64) || defined(_WIN32))
__device__ float roundf(float a);
extern __device__ int __finitel(long double a);
extern __device__ int __isinfl(long double a);
extern __device__ int __isnanl(long double a);
#endif

/// Fix issue:va_printf is not defined in _CubLog.
/// solution: for migration, just define _CubLog to empty.
#define _CubLog(format, ...)

#endif
