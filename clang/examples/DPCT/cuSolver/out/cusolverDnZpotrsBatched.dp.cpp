#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/lapack_utils.hpp>
#include <complex>

void test(sycl::queue *handle, oneapi::mkl::uplo upper_lower, int n, int nrhs,
          sycl::double2 **a, int lda, sycl::double2 **b, int ldb, int *info,
          int group_count) {
  // Start
  dpct::lapack::potrs_batch(*handle, upper_lower, n, nrhs, a, lda, b, ldb, info,
                            group_count);
  // End
}
