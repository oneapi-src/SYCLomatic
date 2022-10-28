// UNSUPPORTED: -windows-
// RUN: ldd $(dirname `which intercept-build`)/../lib/libear/libear.so | grep "libsvml.so"
// RUN: ldd $(dirname `which intercept-build`)/../lib/libear/libear.so | grep "libirng.so"
// RUN: ldd $(dirname `which intercept-build`)/../lib/libear/libear.so | grep "libimf.so"
// RUN: ldd $(dirname `which intercept-build`)/../lib/libear/libear.so | grep "libintlc.so"


// Test description:
// This test is to check whether libear.so depends on libsvml.so when it is built out by Intel icx compiler
__global__ void foo(){}
