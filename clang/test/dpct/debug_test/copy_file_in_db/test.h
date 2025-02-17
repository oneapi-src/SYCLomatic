#include <cuda_runtime.h>

// CHECK: /*
// CHECK: DPCT1057:{{[0-9]+}}: Variable float_to_force was used in host code and device code. float_to_force type was updated to be used in SYCL device code and new float_to_force_host_ct1 was generated to be used in host code. You need to update the host code manually to use the new float_to_force_host_ct1.
// CHECK: */
// CHECK: static const float float_to_force_host_ct1 = (float)(1ll << 40);
// CHECK: static dpct::constant_memory<const float, 0> float_to_force((float)(1ll << 40));
static __constant__ const float float_to_force = (float)(1ll << 40);
