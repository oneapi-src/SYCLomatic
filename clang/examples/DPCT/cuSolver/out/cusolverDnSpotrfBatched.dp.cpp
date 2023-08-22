#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/lapack_utils.hpp>

void test(sycl::queue *handle, oneapi::mkl::uplo upper_lower, int n, float **a,
          int lda, int *info, int group_count) {
  // Start
  dpct::lapack::potrf_batch(*handle, upper_lower, n, a, lda, info, group_count);
  // End
}
