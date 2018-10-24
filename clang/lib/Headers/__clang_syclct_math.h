#ifndef __CLANG_SYCLCT_MATH_H__
#define __CLANG_SYCLCT_MATH_H__

#define __SYCLCT__

#ifdef __SYCLCT__

double fmax(double a, double b) {
  if (__isnan(a) && __isnan(b))
    return a + b;
  if (__isnan(a))
    return b;
  if (__isnan(b))
    return a;
  if ((a == 0.0) && (b == 0.0) && __signbit(b))
    return a;
  return a > b ? a : b;
}

float fmaxf(float a, float b) { return (float)fmax((double)a, (double)b); }

float max(float a, float b) { return fmaxf(a, b); }

#endif
#endif
