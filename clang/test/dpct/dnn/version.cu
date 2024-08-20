// RUN: dpct -in-root %S -out-root %T/version %S/version.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/version/version.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/version/version.dp.cpp -o %T/version/version.dp.o %}
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>


int main() {
    // CHECK:   /*
    // CHECK:   DPCT1043:{{[0-9]+}}: The version-related API is different in SYCL. An initial code was generated, but you need to adjust it.
    // CHECK:   */
    // CHECK: size_t version = dpct::dnnl::get_version();
    size_t version = cudnnGetVersion();
    // CHECK:   /*
    // CHECK:   DPCT1043:{{[0-9]+}}: The version-related API is different in SYCL. An initial code was generated, but you need to adjust it.
    // CHECK:   */
    // CHECK: version = dpct::get_major_version(dpct::get_current_device());
    version = cudnnGetCudartVersion();

    return 0;
}