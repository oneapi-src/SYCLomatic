// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtCreate | FileCheck %s -check-prefix=cublasLtCreate
// cublasLtCreate: CUDA API:
// cublasLtCreate-NEXT:   cublasLtCreate(lthandle /*cublasLtHandle_t **/);
// cublasLtCreate-NEXT: Is migrated to:
// cublasLtCreate-NEXT:   *lthandle = new dpct::blas_gemm::experimental::descriptor();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtDestroy | FileCheck %s -check-prefix=cublasLtDestroy
// cublasLtDestroy: CUDA API:
// cublasLtDestroy-NEXT:   cublasLtDestroy(lthandle /*cublasLtHandle_t*/);
// cublasLtDestroy-NEXT: Is migrated to:
// cublasLtDestroy-NEXT:   delete (lthandle);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtGetVersion | FileCheck %s -check-prefix=cublasLtGetVersion
// cublasLtGetVersion: CUDA API:
// cublasLtGetVersion-NEXT:   size_t ver = cublasLtGetVersion();
// cublasLtGetVersion-NEXT: Is migrated to:
// cublasLtGetVersion-NEXT:   size_t ver = dpct::dnnl::get_version();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatmul | FileCheck %s -check-prefix=cublasLtMatmul
// cublasLtMatmul: CUDA API:
// cublasLtMatmul-NEXT:   cublasLtMatmul(
// cublasLtMatmul-NEXT:       lthandle /*cublasLtHandle_t*/, mm_desc /*cublasLtMatmulDesc_t*/,
// cublasLtMatmul-NEXT:       alpha /*const void **/, a /*const void **/,
// cublasLtMatmul-NEXT:       a_desc /*cublasLtMatrixLayout_t*/, b /*const void **/,
// cublasLtMatmul-NEXT:       b_desc /*cublasLtMatrixLayout_t*/, beta /*const void **/,
// cublasLtMatmul-NEXT:       c /*const void **/, c_desc /*cublasLtMatrixLayout_t*/, d /*void **/,
// cublasLtMatmul-NEXT:       d_desc /*cublasLtMatrixLayout_t*/, algo /*const cublasLtMatmulAlgo_t **/,
// cublasLtMatmul-NEXT:       workspace /*void **/, workspace_size /*size_t*/, stream /*cudaStream_t*/);
// cublasLtMatmul-NEXT: Is migrated to:
// cublasLtMatmul-NEXT:   dpct::blas_gemm::experimental::matmul(lthandle, mm_desc, alpha, a, a_desc, b, b_desc, beta, c, c_desc, d, d_desc, stream);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatmulAlgoGetHeuristic | FileCheck %s -check-prefix=cublasLtMatmulAlgoGetHeuristic
// cublasLtMatmulAlgoGetHeuristic: CUDA API:
// cublasLtMatmulAlgoGetHeuristic-NEXT:   cublasLtMatmulAlgoGetHeuristic(
// cublasLtMatmulAlgoGetHeuristic-NEXT:       lthandle /*cublasLtHandle_t*/, mm_desc /*cublasLtMatmulDesc_t*/,
// cublasLtMatmulAlgoGetHeuristic-NEXT:       a_desc /*cublasLtMatrixLayout_t*/, b_desc /*cublasLtMatrixLayout_t*/,
// cublasLtMatmulAlgoGetHeuristic-NEXT:       c_desc /*cublasLtMatrixLayout_t*/, d_desc /*cublasLtMatrixLayout_t*/,
// cublasLtMatmulAlgoGetHeuristic-NEXT:       preference /*cublasLtMatmulPreference_t*/, requested_algo_count /*int*/,
// cublasLtMatmulAlgoGetHeuristic-NEXT:       heuristic_results_array /*cublasLtMatmulHeuristicResult_t[]*/,
// cublasLtMatmulAlgoGetHeuristic-NEXT:       return_algo_count /*int **/);
// cublasLtMatmulAlgoGetHeuristic-NEXT: Is migrated to:
// cublasLtMatmulAlgoGetHeuristic-NEXT:   *return_algo_count = 1;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatmulDescCreate | FileCheck %s -check-prefix=cublasLtMatmulDescCreate
// cublasLtMatmulDescCreate: CUDA API:
// cublasLtMatmulDescCreate-NEXT:   cublasLtMatmulDescCreate(mm_esc /*cublasLtMatmulDesc_t **/,
// cublasLtMatmulDescCreate-NEXT:                            compute_type /*cublasComputeType_t*/,
// cublasLtMatmulDescCreate-NEXT:                            scale_type /*cudaDataType_t*/);
// cublasLtMatmulDescCreate-NEXT: Is migrated to:
// cublasLtMatmulDescCreate-NEXT:   *mm_esc = new dpct::blas_gemm::experimental::matmul_desc_t(compute_type, scale_type);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatmulDescDestroy | FileCheck %s -check-prefix=cublasLtMatmulDescDestroy
// cublasLtMatmulDescDestroy: CUDA API:
// cublasLtMatmulDescDestroy-NEXT:   cublasLtMatmulDescDestroy(mm_esc /*cublasLtMatmulDesc_t*/);
// cublasLtMatmulDescDestroy-NEXT: Is migrated to:
// cublasLtMatmulDescDestroy-NEXT:   delete (mm_esc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatmulDescGetAttribute | FileCheck %s -check-prefix=cublasLtMatmulDescGetAttribute
// cublasLtMatmulDescGetAttribute: CUDA API:
// cublasLtMatmulDescGetAttribute-NEXT:   cublasLtMatmulDescGetAttribute(
// cublasLtMatmulDescGetAttribute-NEXT:       mm_esc /*cublasLtMatmulDesc_t*/, attr /*cublasLtMatmulDescAttributes_t*/,
// cublasLtMatmulDescGetAttribute-NEXT:       buf /*void **/, size_in_bytes /*size_t*/, size_written /*size_t **/);
// cublasLtMatmulDescGetAttribute-NEXT: Is migrated to:
// cublasLtMatmulDescGetAttribute-NEXT:   mm_esc->get_attribute(attr, buf);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatmulDescSetAttribute | FileCheck %s -check-prefix=cublasLtMatmulDescSetAttribute
// cublasLtMatmulDescSetAttribute: CUDA API:
// cublasLtMatmulDescSetAttribute-NEXT:   cublasLtMatmulDescSetAttribute(
// cublasLtMatmulDescSetAttribute-NEXT:       mm_esc /*cublasLtMatmulDesc_t*/, attr /*cublasLtMatmulDescAttributes_t*/,
// cublasLtMatmulDescSetAttribute-NEXT:       buf /*const void **/, size_in_bytes /*size_t*/);
// cublasLtMatmulDescSetAttribute-NEXT: Is migrated to:
// cublasLtMatmulDescSetAttribute-NEXT:   mm_esc->set_attribute(attr, buf);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatmulPreferenceCreate | FileCheck %s -check-prefix=cublasLtMatmulPreferenceCreate
// cublasLtMatmulPreferenceCreate: CUDA API:
// cublasLtMatmulPreferenceCreate-NEXT:   cublasLtMatmulPreferenceCreate(pref /*cublasLtMatmulPreference_t **/);
// cublasLtMatmulPreferenceCreate-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatmulPreferenceDestroy | FileCheck %s -check-prefix=cublasLtMatmulPreferenceDestroy
// cublasLtMatmulPreferenceDestroy: CUDA API:
// cublasLtMatmulPreferenceDestroy-NEXT:   cublasLtMatmulPreferenceDestroy(pref /*cublasLtMatmulPreference_t*/);
// cublasLtMatmulPreferenceDestroy-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatmulPreferenceGetAttribute | FileCheck %s -check-prefix=cublasLtMatmulPreferenceGetAttribute
// cublasLtMatmulPreferenceGetAttribute: CUDA API:
// cublasLtMatmulPreferenceGetAttribute-NEXT:   cublasLtMatmulPreferenceGetAttribute(
// cublasLtMatmulPreferenceGetAttribute-NEXT:       pref /*cublasLtMatmulPreference_t*/,
// cublasLtMatmulPreferenceGetAttribute-NEXT:       attr /*cublasLtMatmulPreferenceAttributes_t*/, buf /*void **/,
// cublasLtMatmulPreferenceGetAttribute-NEXT:       size_in_bytes /*size_t*/, size_written /*size_t **/);
// cublasLtMatmulPreferenceGetAttribute-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatmulPreferenceSetAttribute | FileCheck %s -check-prefix=cublasLtMatmulPreferenceSetAttribute
// cublasLtMatmulPreferenceSetAttribute: CUDA API:
// cublasLtMatmulPreferenceSetAttribute-NEXT:   cublasLtMatmulPreferenceSetAttribute(
// cublasLtMatmulPreferenceSetAttribute-NEXT:       pref /*cublasLtMatmulPreference_t*/,
// cublasLtMatmulPreferenceSetAttribute-NEXT:       attr /*cublasLtMatmulPreferenceAttributes_t*/, buf /*const void **/,
// cublasLtMatmulPreferenceSetAttribute-NEXT:       size_in_bytes /*size_t*/);
// cublasLtMatmulPreferenceSetAttribute-NEXT: The API is Removed.

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatrixLayoutCreate | FileCheck %s -check-prefix=cublasLtMatrixLayoutCreate
// cublasLtMatrixLayoutCreate: CUDA API:
// cublasLtMatrixLayoutCreate-NEXT:   cublasLtMatrixLayoutCreate(layout /*cublasLtMatrixLayout_t **/,
// cublasLtMatrixLayoutCreate-NEXT:                              type /*cudaDataType*/, rows /*uint64_t*/,
// cublasLtMatrixLayoutCreate-NEXT:                              cols /*uint64_t*/, ld /*int64_t*/);
// cublasLtMatrixLayoutCreate-NEXT: Is migrated to:
// cublasLtMatrixLayoutCreate-NEXT:   *layout = new dpct::blas_gemm::experimental::matrix_layout_t(type, rows, cols, ld);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatrixLayoutDestroy | FileCheck %s -check-prefix=cublasLtMatrixLayoutDestroy
// cublasLtMatrixLayoutDestroy: CUDA API:
// cublasLtMatrixLayoutDestroy-NEXT:   cublasLtMatrixLayoutDestroy(layout /*cublasLtMatrixLayout_t*/);
// cublasLtMatrixLayoutDestroy-NEXT: Is migrated to:
// cublasLtMatrixLayoutDestroy-NEXT:   delete (layout);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatrixLayoutGetAttribute | FileCheck %s -check-prefix=cublasLtMatrixLayoutGetAttribute
// cublasLtMatrixLayoutGetAttribute: CUDA API:
// cublasLtMatrixLayoutGetAttribute-NEXT:   cublasLtMatrixLayoutGetAttribute(layout /*cublasLtMatrixLayout_t*/,
// cublasLtMatrixLayoutGetAttribute-NEXT:                                    attr /*cublasLtMatrixLayoutAttribute_t*/,
// cublasLtMatrixLayoutGetAttribute-NEXT:                                    buf /*void **/, size_in_bytes /*size_t*/,
// cublasLtMatrixLayoutGetAttribute-NEXT:                                    size_written /*size_t **/);
// cublasLtMatrixLayoutGetAttribute-NEXT: Is migrated to:
// cublasLtMatrixLayoutGetAttribute-NEXT:   layout->get_attribute(attr, buf);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatrixLayoutSetAttribute | FileCheck %s -check-prefix=cublasLtMatrixLayoutSetAttribute
// cublasLtMatrixLayoutSetAttribute: CUDA API:
// cublasLtMatrixLayoutSetAttribute-NEXT:   cublasLtMatrixLayoutSetAttribute(layout /*cublasLtMatrixLayout_t*/,
// cublasLtMatrixLayoutSetAttribute-NEXT:                                    attr /*cublasLtMatrixLayoutAttribute_t*/,
// cublasLtMatrixLayoutSetAttribute-NEXT:                                    buf /*const void **/,
// cublasLtMatrixLayoutSetAttribute-NEXT:                                    size_in_bytes /*size_t*/);
// cublasLtMatrixLayoutSetAttribute-NEXT: Is migrated to:
// cublasLtMatrixLayoutSetAttribute-NEXT:   layout->set_attribute(attr, buf);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatrixTransform | FileCheck %s -check-prefix=cublasLtMatrixTransform
// cublasLtMatrixTransform: CUDA API:
// cublasLtMatrixTransform-NEXT:   cublasLtMatrixTransform(
// cublasLtMatrixTransform-NEXT:       lthandle /*cublasLtHandle_t*/,
// cublasLtMatrixTransform-NEXT:       trans_desc /*cublasLtMatrixTransformDesc_t*/, alpha /*const void **/,
// cublasLtMatrixTransform-NEXT:       a /*const void **/, a_desc /*cublasLtMatrixLayout_t*/,
// cublasLtMatrixTransform-NEXT:       beta /*const void **/, b /*const void **/,
// cublasLtMatrixTransform-NEXT:       b_desc /*cublasLtMatrixLayout_t*/, c /*void **/,
// cublasLtMatrixTransform-NEXT:       c_desc /*cublasLtMatrixLayout_t*/, stream /*cudaStream_t*/);
// cublasLtMatrixTransform-NEXT: Is migrated to:
// cublasLtMatrixTransform-NEXT:   dpct::blas_gemm::experimental::matrix_transform(trans_desc, alpha, a, a_desc, beta, b, b_desc, c, c_desc, stream);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatrixTransformDescCreate | FileCheck %s -check-prefix=cublasLtMatrixTransformDescCreate
// cublasLtMatrixTransformDescCreate: CUDA API:
// cublasLtMatrixTransformDescCreate-NEXT:   cublasLtMatrixTransformDescCreate(
// cublasLtMatrixTransformDescCreate-NEXT:       trans_desc /*cublasLtMatrixTransformDesc_t **/,
// cublasLtMatrixTransformDescCreate-NEXT:       scale_type /*cudaDataType*/);
// cublasLtMatrixTransformDescCreate-NEXT: Is migrated to:
// cublasLtMatrixTransformDescCreate-NEXT:   *trans_desc = new dpct::blas_gemm::experimental::transform_desc_t(scale_type);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatrixTransformDescDestroy | FileCheck %s -check-prefix=cublasLtMatrixTransformDescDestroy
// cublasLtMatrixTransformDescDestroy: CUDA API:
// cublasLtMatrixTransformDescDestroy-NEXT:   cublasLtMatrixTransformDescDestroy(
// cublasLtMatrixTransformDescDestroy-NEXT:       trans_desc /*cublasLtMatrixTransformDesc_t*/);
// cublasLtMatrixTransformDescDestroy-NEXT: Is migrated to:
// cublasLtMatrixTransformDescDestroy-NEXT:   delete (trans_desc);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatrixTransformDescGetAttribute | FileCheck %s -check-prefix=cublasLtMatrixTransformDescGetAttribute
// cublasLtMatrixTransformDescGetAttribute: CUDA API:
// cublasLtMatrixTransformDescGetAttribute-NEXT:   cublasLtMatrixTransformDescGetAttribute(
// cublasLtMatrixTransformDescGetAttribute-NEXT:       transformDesc /*cublasLtMatrixTransformDesc_t*/,
// cublasLtMatrixTransformDescGetAttribute-NEXT:       attr /*cublasLtMatrixTransformDescAttributes_t*/, buf /*void **/,
// cublasLtMatrixTransformDescGetAttribute-NEXT:       sizeInBytes /*size_t*/, sizeWritten /*size_t **/);
// cublasLtMatrixTransformDescGetAttribute-NEXT: Is migrated to:
// cublasLtMatrixTransformDescGetAttribute-NEXT:   transformDesc->get_attribute(attr, buf);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cublasLtMatrixTransformDescSetAttribute | FileCheck %s -check-prefix=cublasLtMatrixTransformDescSetAttribute
// cublasLtMatrixTransformDescSetAttribute: CUDA API:
// cublasLtMatrixTransformDescSetAttribute-NEXT:   cublasLtMatrixTransformDescSetAttribute(
// cublasLtMatrixTransformDescSetAttribute-NEXT:       transformDesc /*cublasLtMatrixTransformDesc_t*/,
// cublasLtMatrixTransformDescSetAttribute-NEXT:       attr /*cublasLtMatrixTransformDescAttributes_t*/, buf /*const void **/,
// cublasLtMatrixTransformDescSetAttribute-NEXT:       sizeInBytes /*size_t*/);
// cublasLtMatrixTransformDescSetAttribute-NEXT: Is migrated to:
// cublasLtMatrixTransformDescSetAttribute-NEXT:   transformDesc->set_attribute(attr, buf);
