#include <thrust/advance.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

void thrust_advance() {
  const int N = 5;
  // Start
  thrust::device_vector<int> vec(N);
  thrust::device_vector<int>::iterator iter = vec.begin();
  thrust::advance(iter, 2);
  // End
}
