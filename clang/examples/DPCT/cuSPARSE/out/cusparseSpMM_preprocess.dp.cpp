#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/sparse_utils.hpp>
#include <dpct/blas_utils.hpp>
#include <dpct/lib_common_utils.hpp>

void test(sycl::queue *handle, oneapi::mkl::transpose transa,
          oneapi::mkl::transpose transb, const void *alpha,
          cusparseConstSpMatDescr_t a, cusparseConstDnMatDescr_t b,
          const void *beta, std::shared_ptr<dpct::sparse::dense_matrix_desc> c,
          dpct::library_data_t computetype, int algo, void *workspace) {
  // Start
  /*
  DPCT1026:0: The call to cusparseSpMM_preprocess was removed because this call
  is redundant in SYCL.
  */
  // End
}
