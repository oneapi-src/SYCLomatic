#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/find.h>

void exclusive_scan_test() {
  // clang-format off
  // Start
  thrust::host_vector<int> h;
  thrust::device_vector<int> d;
  /*1*/ thrust::find(thrust::seq, h.begin(), h.end(), 1);
  /*2*/ thrust::find(h.begin(), h.end(), 1);
  /*3*/ thrust::find(d.begin(), d.end(), 1);
  // End
  // clang-format on
}
