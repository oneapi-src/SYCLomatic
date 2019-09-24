// RUN: clang++ -fsycl memcpy2d3d.cpp -o memcpy2d3d -lOpenCL -lsycl
#include <unistd.h>
//#define DPCT_USM_LEVEL_NONE
#include "../include/dpct.hpp"

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
                 [&](int &data, int pos) -> bool { return data == pos; });
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

  init(src_data, range);
  init_zero(dst_data, range);
  int *d_data;
  dpct::dpct_malloc(&d_data, sizeof(int) * range.size());

  auto copy_size = size;
  copy_size[0] *= sizeof(int);
  auto copy_offset = offset;
  copy_offset[0] *= sizeof(int);
  dpct::dpct_memcpy_param p;
  p.from.pitch() = dpct::dpct_create_pitch(src_data, range.get(0) * sizeof(int),
                                    range.get(0), range.get(1));
  p.from.offset = copy_offset;
  p.to.pitch() = dpct::dpct_create_pitch(d_data, range.get(0) * sizeof(int),
                                  range.get(0), range.get(1));
  p.to.offset = copy_offset;
  p.copy_size = copy_size;
  dpct::dpct_memcpy(&p);
  p.from.pitch() = dpct::dpct_create_pitch(d_data, range.get(0) * sizeof(int),
                                    range.get(0), range.get(1));
  p.to.pitch() = dpct::dpct_create_pitch(dst_data, range.get(0) * sizeof(int),
                                  range.get(0), range.get(1));
  dpct::dpct_memcpy(&p);
  if (check(dst_data, range, offset, size))
    printf("test success!\n");
  else
    printf("test fail!\n");

  dpct::dpct_free(d_data);
  std::free(src_data);
  std::free(dst_data);
}
