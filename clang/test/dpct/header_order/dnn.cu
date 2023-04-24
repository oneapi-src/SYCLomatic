// RUN: echo
// CHECK: #include <dpct/dnnl_utils.hpp>
// CHECK: #include <sycl/sycl.hpp>
// CHECK: #include <dpct/dpct.hpp>
#include "dnn.h"

int test(cudnnHandle_t handle){

 cudnnCreate(&handle);
 return 0;

}
