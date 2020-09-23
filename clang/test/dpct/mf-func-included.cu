// RUN: echo "empty command"

// CHECK:static void static_func() {}
__global__ static void static_func() {}

namespace
{
// CHECK:  void static_func_in_anonymous_namespace(){}
  __global__ void static_func_in_anonymous_namespace(){}
}
