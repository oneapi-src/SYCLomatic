#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
#include <thrust/partition.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

void count_test() {
  // clang-format off
  // Start
  std::vector<int> v;
  /*1*/ thrust::count(thrust::seq, v.begin(), v.end(), 23);
  /*2*/ thrust::count(v.begin(), v.end(), 23);
  // End
  // clang-format on
}
