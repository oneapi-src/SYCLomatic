// RUN: echo "empty command"

// CHECK:static void static_func_mid() {}
__global__ static void static_func_mid() {}

namespace
{
// CHECK:  void static_func_in_anonymous_namespace_mid(){}
  __global__ void static_func_in_anonymous_namespace_mid(){}
}
