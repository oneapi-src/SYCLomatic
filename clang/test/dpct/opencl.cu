// ------ prepare test directory
// RUN: cd %T
// RUN: rm -rf opencl-build
// RUN: mkdir  opencl-build
// RUN: cd     opencl-build
// RUN: cp %s opencl.cu
//
// ------ run dpct
// RUN: dpct opencl.cu --cuda-include-path="%cuda-path/include"
//
// ------ ensure file inclusion of CL/opencl.h is kept
// RUN: FileCheck --input-file dpct_output/opencl.dp.cpp --match-full-lines %s
//
// ------ cleanup test directory
// RUN: cd ..
// RUN: rm -rf ./opencl-build

// CHECK: #include <sycl/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>
// CHECK: #include <iostream>
// CHECK: #include <math.h>
// CHECK: #include <CL/cl.h>
// CHECK: #include <CL/cl_egl.h>
// CHECK: #include <CL/cl_ext.h>
// CHECK: #include <CL/cl_gl_ext.h>
// CHECK: #include <CL/cl_platform.h>
// CHECK: #include <CL/opencl.h>

#include <iostream>
#include <math.h>
#include <CL/cl.h>
#include <CL/cl_egl.h>
#include <CL/cl_ext.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_platform.h>
#include <CL/opencl.h>
//#include <CL/cl.hpp>          do not test, needs GL/gl.h
//#include <CL/cl_gl.h>         do not test, needs GL/gl.h

__global__
void recip( double *x, double *y)
{
    int a = *x;

    *y = __drcp_rn(*x);
}
