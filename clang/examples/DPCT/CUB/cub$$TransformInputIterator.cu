// clang-format off
#include <cstddef>
#include <cub/cub.cuh>

struct UserDefMul {
  __device__ double operator()(double d) const {
    return d * 3.0;
  }
};

void test(double *d_in, UserDefMul op) {
  // Start
  cub::TransformInputIterator<double, UserDefMul, double *> iter(d_in /*double **/, op /*Op*/);
  // End
}
