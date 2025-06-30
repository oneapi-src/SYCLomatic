#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

void thrust_make_transform_iterator() {
  // clang-format off
  // Start
  thrust::device_vector<int> int_in(3);
  thrust::device_vector<float> float_in(3);
  typedef thrust::device_vector<int>::iterator int_iterator;
  typedef thrust::device_vector<float>::iterator float_iterator;
  typedef thrust::tuple<int_iterator, float_iterator> iterator_tuple;
  thrust::zip_iterator<iterator_tuple> ret = thrust::make_zip_iterator(thrust::make_tuple(int_in.begin(), float_in.begin()));
  // End
  // clang-format on
}
