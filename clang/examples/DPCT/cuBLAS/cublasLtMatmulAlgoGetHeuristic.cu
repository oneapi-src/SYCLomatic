#include "cublasLt.h"

void test(cublasLtHandle_t lthandle, cublasLtMatmulDesc_t mm_desc,
          cublasLtMatrixLayout_t a_desc, cublasLtMatrixLayout_t b_desc,
          cublasLtMatrixLayout_t c_desc, cublasLtMatrixLayout_t d_desc,
          cublasLtMatmulPreference_t preference, int requested_algo_count,
          cublasLtMatmulHeuristicResult_t heuristic_results_array[],
          int *return_algo_count) {
  // Start
  cublasLtMatmulAlgoGetHeuristic(
      lthandle /*cublasLtHandle_t*/, mm_desc /*cublasLtMatmulDesc_t*/,
      a_desc /*cublasLtMatrixLayout_t*/, b_desc /*cublasLtMatrixLayout_t*/,
      c_desc /*cublasLtMatrixLayout_t*/, d_desc /*cublasLtMatrixLayout_t*/,
      preference /*cublasLtMatmulPreference_t*/, requested_algo_count /*int*/,
      heuristic_results_array /*cublasLtMatmulHeuristicResult_t[]*/,
      return_algo_count /*int **/);
  // End
}
