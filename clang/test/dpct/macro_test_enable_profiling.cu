// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --enable-profiling   -out-root %T/macro_test_enable_profiling %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/macro_test_enable_profiling/macro_test_enable_profiling.dp.cpp --match-full-lines %s

// CHECK: #include <oneapi/dpl/execution>
// CHECK-NEXT: #include <oneapi/dpl/algorithm>
// CHECK-NEXT:#define DPCT_PROFILING_ENABLED
// CHECK-NEXT:#include <sycl/sycl.hpp>
// CHECK-NEXT:#include <dpct/dpct.hpp>
// CHECK-NEXT:#include <math.h>
// CHECK-NEXT:#include <iostream>
// CHECK-NEXT:#include <cmath>
// CHECK-NEXT:#include <iomanip>
// CHECK-NEXT:#include <limits>
// CHECK-NEXT:#include <algorithm>
#include <math.h>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include <algorithm>

#include <stdio.h>


#include <thrust/inner_product.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/reduce.h>


//     CHECK:#define CMC_PROFILING_BEGIN()                                                  \
//CHECK-NEXT:  dpct::event_ptr start;                                                         \
//CHECK-NEXT:  dpct::event_ptr stop;                                                          \
//CHECK-NEXT:  if (CMC_profile)                                                             \
//CHECK-NEXT:  {                                                                            \
//CHECK-NEXT:    start = new sycl::event();                                                 \
//CHECK-NEXT:    stop = new sycl::event();                                                  \
//CHECK-NEXT:    *start = q_ct1.ext_oneapi_submit_barrier();                                \
//CHECK-NEXT:  }
#define CMC_PROFILING_BEGIN()                                                                                      \
  cudaEvent_t start;                                                                                               \
  cudaEvent_t stop;                                                                                                \
  if (CMC_profile)                                                                                                 \
  {                                                                                                                \
    cudaEventCreate(&start);                                                                                       \
    cudaEventCreate(&stop);                                                                                        \
    cudaGetLastError();                                                                                            \
    cudaEventRecord(start);                                                                                        \
  }

//     CHECK:#define CMC_PROFILING_END(lineno)                                                                                                                                         \
//CHECK-NEXT:  if (CMC_profile)                                                                                                                                                        \
//CHECK-NEXT:  {                                                                                                                                                                       \
//CHECK-NEXT:    *stop = q_ct1.ext_oneapi_submit_barrier();                                                                                                                            \
//CHECK-NEXT:    stop->wait_and_throw();                                                                                                                                               \
//CHECK-NEXT:    float time = 0.0f;                                                                                                                                                    \
//CHECK-NEXT:    time = (stop->get_profiling_info<sycl::info::event_profiling::command_end>() - start->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f; \
//CHECK-NEXT:    dpct::destroy_event(start);                                                                                                                                           \
//CHECK-NEXT:    dpct::destroy_event(stop);                                                                                                                                            \
//CHECK-NEXT:  }                                                                                                                                                                       \
//CHECK-NEXT:  int error = 0;
#define CMC_PROFILING_END(lineno)                                                                          \
  if (CMC_profile)                                                                                         \
  {                                                                                                        \
    cudaEventRecord(stop);                                                                                 \
    cudaEventSynchronize(stop);                                                                            \
    float time = 0.0f;                                                                                     \
    cudaEventElapsedTime(&time, start, stop);                                                              \
    cudaEventDestroy(start);                                                                               \
    cudaEventDestroy(stop);                                                                                \
  }                                                                                                        \
  cudaError_t error = cudaGetLastError();                                                                  \
  if (error)                                                                                               \
  {                                                                                                        \
    printf("%s\nCUDA ERROR!!! Detected at end of CMC_PROFILING_END in BsplineJastrowCudaPBC line %d!!!\n", \
           cudaGetErrorString(error),                                                                      \
           lineno);                                                                                        \
    exit(1);                                                                                               \
  }

void foo20() {
  bool CMC_profile = true;
  CMC_PROFILING_BEGIN();
  CMC_PROFILING_END(__LINE__);
}