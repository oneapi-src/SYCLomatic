// RUN: dpct --format-range=none --no-dpcpp-extensions=peer_access -out-root %T/cuda_peer_access %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_peer_access/cuda_peer_access.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/cuda_peer_access/cuda_peer_access.dp.cpp -o %T/cuda_peer_access/cuda_peer_access.dp.o %}

#include <cuda.h>
#include <cuda_runtime.h>

int main() {
  int r;
// CHECK:  /*
// CHECK:  DPCT1031:{{[0-9]+}}: Memory access across peer devices is an implementation-specific feature which may not be supported by some SYCL backends and compilers. The output parameter(s) are set to 0. You can migrate the code with peer access extension if you do not specify -no-dpcpp-extensions=peer_access.
// CHECK:  */
// CHECK:  r = 0;
  cudaDeviceCanAccessPeer(&r, 0, 0);
// CHECK:  /*
// CHECK:  DPCT1026:{{[0-9]+}}: The call to cudaDeviceEnablePeerAccess was removed because SYCL currently does not support memory access across peer devices. You can migrate the code with peer access extension by not specifying -no-dpcpp-extensions=peer_access.
// CHECK:  */
  cudaDeviceEnablePeerAccess(0, 0);
// CHECK:  /*
// CHECK:  DPCT1026:{{[0-9]+}}: The call to cudaDeviceDisablePeerAccess was removed because SYCL currently does not support memory access across peer devices. You can migrate the code with peer access extension by not specifying -no-dpcpp-extensions=peer_access.
// CHECK:  */
  cudaDeviceDisablePeerAccess(0);

// CHECK:  /*
// CHECK:  DPCT1031:{{[0-9]+}}: Memory access across peer devices is an implementation-specific feature which may not be supported by some SYCL backends and compilers. The output parameter(s) are set to 0. You can migrate the code with peer access extension if you do not specify -no-dpcpp-extensions=peer_access.
// CHECK:  */
// CHECK:  r = 0;
  cuDeviceCanAccessPeer(&r, 0, 0);

  return 0;
}
