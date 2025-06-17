// UNSUPPORTED: cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6, cuda-12.8, cuda-12.9
// UNSUPPORTED: v12.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6, v12.8, v12.9

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseCreateCsrgemm2Info | FileCheck %s -check-prefix=cusparseCreateCsrgemm2Info
// cusparseCreateCsrgemm2Info: CUDA API:
// cusparseCreateCsrgemm2Info-NEXT:   csrgemm2Info_t info;
// cusparseCreateCsrgemm2Info-NEXT:   cusparseCreateCsrgemm2Info(&info /*csrgemm2Info_t **/);
// cusparseCreateCsrgemm2Info-NEXT: Is migrated to:
// cusparseCreateCsrgemm2Info-NEXT:   std::shared_ptr<dpct::sparse::csrgemm2_info> info;
// cusparseCreateCsrgemm2Info-NEXT:   info = std::make_shared<dpct::sparse::csrgemm2_info>();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cusparseDestroyCsrgemm2Info | FileCheck %s -check-prefix=cusparseDestroyCsrgemm2Info
// cusparseDestroyCsrgemm2Info: CUDA API:
// cusparseDestroyCsrgemm2Info-NEXT:   cusparseDestroyCsrgemm2Info(info /*csrgemm2Info_t*/);
// cusparseDestroyCsrgemm2Info-NEXT: Is migrated to:
// cusparseDestroyCsrgemm2Info-NEXT:   info.reset();
