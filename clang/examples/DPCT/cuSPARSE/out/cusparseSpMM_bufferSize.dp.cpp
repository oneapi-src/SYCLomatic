#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/sparse_utils.hpp>
#include <dpct/blas_utils.hpp>
#include <dpct/lib_common_utils.hpp>

void test(sycl::queue *handle, oneapi::mkl::transpose transa,
          oneapi::mkl::transpose transb, const void *alpha,
          cusparseConstSpMatDescr_t a, cusparseConstDnMatDescr_t b,
          const void *beta, std::shared_ptr<dpct::sparse::dense_matrix_desc> c,
          dpct::library_data_t computetype, int algo, size_t *workspace_size) {
  // Start
  *workspace_size = 0;
  // End
}
