// RUN: dpct --format-range=none --use-dpcpp-extensions=peer_access -out-root %T/cuda_peer_access %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/cuda_peer_access/cuda_peer_access.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/cuda_peer_access/cuda_peer_access.dp.cpp -o %T/cuda_peer_access/cuda_peer_access.dp.o %}

#include <cuda_runtime.h>

int main() {
  int r;

// CHECK:  r = dpct::dev_mgr::instance().get_device(0).ext_oneapi_can_access_peer(dpct::dev_mgr::instance().get_device(0));
// CHECK:  dpct::get_current_device().ext_oneapi_enable_peer_access(dpct::dev_mgr::instance().get_device(0));
// CHECK:  dpct::get_current_device().ext_oneapi_disable_peer_access(dpct::dev_mgr::instance().get_device(0));
  cudaDeviceCanAccessPeer(&r, 0, 0);
  cudaDeviceEnablePeerAccess(0, 0);
  cudaDeviceDisablePeerAccess(0);

// CHECK:  auto p = DPCT_CHECK_ERROR(r = dpct::dev_mgr::instance().get_device(0).ext_oneapi_can_access_peer(dpct::dev_mgr::instance().get_device(0)));
// CHECK:  p = DPCT_CHECK_ERROR(dpct::get_current_device().ext_oneapi_enable_peer_access(dpct::dev_mgr::instance().get_device(0)));
// CHECK:  p = DPCT_CHECK_ERROR(dpct::get_current_device().ext_oneapi_disable_peer_access(dpct::dev_mgr::instance().get_device(0)));
  auto p = cudaDeviceCanAccessPeer(&r, 0, 0);
  p = cudaDeviceEnablePeerAccess(0, 0);
  p = cudaDeviceDisablePeerAccess(0);

  return 0;
}
