// RUN: clang++ -fsycl memcpy2d3d.cpp -o memcpy2d3d -lOpenCL -lsycl
#include <unistd.h>
//#define DPCT_USM_LEVEL_NONE
#include "dpct/dpct.hpp"

cl::sycl::range<3> &operator+=(cl::sycl::range<3> &range,
                               const cl::sycl::id<3> &id) {
  range[0] += id[0];
  range[1] += id[1];
  range[2] += id[2];
  return range;
}

bool process(int *data, cl::sycl::range<3> range, cl::sycl::id<3> offset,
             cl::sycl::range<3> size,
             const std::function<bool(int &data, int pos)> &processor) {
  size += offset;
  for (unsigned i = offset.get(2); i < size.get(2); ++i) {
    for (unsigned j = offset.get(1); j < size.get(1); ++j) {
      for (unsigned k = offset.get(0); k < size.get(0); ++k) {
        auto pos = i * range.get(1) * range.get(0) + j * range.get(0) + k;
        if (!processor(*(data + pos), pos))
          return false;
      }
    }
  }
  return true;
}

bool check(int *data, cl::sycl::range<3> range, cl::sycl::id<3> offset,
           cl::sycl::range<3> size) {
  return process(data, range, offset, size,
                 [&](int &data, int pos) -> bool { return data == 30; });
}

void init_partial(int *data, cl::sycl::range<3> range, cl::sycl::id<3> offset,
                  cl::sycl::range<3> size) {
  process(data, range, offset, size, [&](int &data, int pos) -> bool {
    data = pos;
    return true;
  });
}

void init(int *data, cl::sycl::range<3> range) {
  init_partial(data, range, cl::sycl::id<3>(0, 0, 0), range);
}

void init_zero(int *data, cl::sycl::range<3> range) {
  process(data, range, cl::sycl::id<3>(0, 0, 0), range,
          [&](int &data, int pos) -> bool {
            data = 0;
            return true;
          });
}

int main() {
  cl::sycl::range<3> size(3, 2, 2), range(6, 6, 6);
  cl::sycl::id<3> offset(2, 2, 1);
  
  int *src_data = (int *)std::malloc(sizeof(int) * range.size());
  int *dst_data = (int *)std::malloc(sizeof(int) * range.size());

  copy_range = range;
  copy_range[0] *= sizeof(int);
  init(src_data, range);
  init_zero(dst_data, range);
  int *d_data;
  dpct::dpct_malloc(&d_data, &copy_range[0], copy_range.get(0), copy_range.get(0) * copy_range.get(1));

  auto copy_size = size;
  copy_size[0] *= sizeof(int);
  auto copy_offset = offset;
  copy_offset[0] *= sizeof(int);
  dpct::pitched_data parms_from_data_ct1, parms_to_data_ct1;
  sycl::id<3> parms_from_pos_ct1(0, 0, 0), parms_to_pos_ct1(0, 0, 0);
  sycl::range<3> parms_size_ct1(0, 0, 0);
  
  dpct::malloc(&parms_from_data_ct1, copy_size);
  dpct::memset(parms_from_data_ct1, 30, copy_range);
  parms_from_pos_ct1 = copy_offset;
  parms_to_data_ct1 = dpct::pitched_data(d_data, copy_range.get(0),
                                  copy_range.get(0), copy_range.get(1));
  parms_to_pos_ct1 = copy_offset;
  parms_size_ct1 = copy_size;
  dpct::memcpy_direction parms_direction_ct1 = dpct::automatic;
  dpct::dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1);
  parms_from_data_ct1 = dpct::pitched_data(d_data, range.get(0) * sizeof(int),
                                    range.get(0), range.get(1));
  parms_to_data_ct1 = dpct::pitched_data(dst_data, range.get(0) * sizeof(int),
                                  range.get(0), range.get(1));
  dpct::dpct_memcpy(parms_to_data_ct1, parms_to_pos_ct1, parms_from_data_ct1, parms_from_pos_ct1, parms_size_ct1, parms_direction_ct1);
  if (check(dst_data, range, offset, size))
    printf("test success!\n");
  else
    printf("test fail!\n");

  dpct::dpct_free(d_data);
  std::free(src_data);
  std::free(dst_data);
}

