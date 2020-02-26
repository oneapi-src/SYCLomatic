// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublas_curandInMacro.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>


#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}


int main() {
    cublasHandle_t handle;
    int N = 275;
    float *d_A_S = 0;
    float *d_B_S = 0;
    float *d_C_S = 0;
    float alpha_S = 1.0f;
    float beta_S = 0.0f;
    int trans0 = 0;
    int trans1 = 1;
    int fill0 = 0;
    int side0 = 0;
    int diag0 = 0;
    int *result = 0;
    const float *x_S = 0;


    // CHECK: /*
    // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: cublasErrCheck([&](){
    // CHECK-NEXT: auto transpose_ct1 = trans0;
    // CHECK-NEXT: auto transpose_ct2 = trans1;
    // CHECK-NEXT: auto d_A_S_buff_ct1 = dpct::get_buffer<float>(d_A_S);
    // CHECK-NEXT: auto d_B_S_buff_ct1 = dpct::get_buffer<float>(d_B_S);
    // CHECK-NEXT: auto d_C_S_buff_ct1 = dpct::get_buffer<float>(d_C_S);
    // CHECK-NEXT: mkl::blas::gemm(handle, (((int)transpose_ct1)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct1)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), N, N, N, *(&alpha_S), d_A_S_buff_ct1, N, d_B_S_buff_ct1, N, *(&beta_S), d_C_S_buff_ct1, N);
    // CHECK-NEXT: return 0;
    // CHECK-NEXT: }());
    cublasErrCheck(cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N));


    // CHECK: /*
    // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: cublasErrCheck([&](){
    // CHECK-NEXT: auto x_S_buff_ct1 = dpct::get_buffer<float>(x_S);
    // CHECK-NEXT: auto result_buff_ct1 = dpct::get_buffer<int>(result);
    // CHECK-NEXT: sycl::buffer<int64_t> result_temp_buffer(sycl::range<1>(1));
    // CHECK-NEXT: mkl::blas::iamax(handle, N, x_S_buff_ct1, N, result_temp_buffer);
    // CHECK-NEXT: result_buff_ct1.get_access<sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<sycl::access::mode::read>()[0];
    // CHECK-NEXT: return 0;
    // CHECK-NEXT: }());
    cublasErrCheck(cublasIsamax(handle, N, x_S, N, result));


    //CHECK: /*
    //CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: cublasErrCheck([&](){
    //CHECK-NEXT: auto transpose_ct2 = trans1;
    //CHECK-NEXT: auto d_A_S_buff_ct1 = dpct::get_buffer<float>(d_A_S);
    //CHECK-NEXT: auto d_B_S_buff_ct1 = dpct::get_buffer<float>(d_B_S);
    //CHECK-NEXT: auto d_C_S_buff_ct1 = dpct::get_buffer<float>(d_C_S);
    //CHECK-NEXT: mkl::blas::gemmt(handle, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct2)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct2)), ((((int)transpose_ct2)==0)?(mkl::transpose::trans):(mkl::transpose::nontrans)), N, N, *(&alpha_S), d_A_S_buff_ct1, N, d_B_S_buff_ct1, N, *(&beta_S), d_C_S_buff_ct1, N);
    // CHECK-NEXT: return 0;
    //CHECK-NEXT: }());
    cublasErrCheck(cublasSsyrkx(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans1, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N));


    // CHECK: /*
    // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: cublasErrCheck([&](){
    // CHECK-NEXT: auto transpose_ct3 = trans0;
    // CHECK-NEXT: auto ptr_ct8 = d_A_S;
    // CHECK-NEXT: auto ptr_ct8_buff_ct1 = dpct::get_buffer<float>(ptr_ct8);
    // CHECK-NEXT: auto ptr_ct12 = d_C_S;
    // CHECK-NEXT: auto ptr_ct12_buff_ct1 = dpct::get_buffer<float>(ptr_ct12);
    // CHECK-NEXT: auto ld_ct13 = N; auto m_ct5 = N; auto n_ct6 = N;
    // CHECK-NEXT: dpct::matrix_mem_copy(ptr_ct12, d_B_S, ld_ct13, N, m_ct5, n_ct6, dpct::device_to_device, handle);
    // CHECK-NEXT: mkl::blas::trmm(handle, (mkl::side)side0, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct3)==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct3)), (mkl::diag)diag0, m_ct5, n_ct6, *(&alpha_S), ptr_ct8_buff_ct1, N, ptr_ct12_buff_ct1, ld_ct13);
    // CHECK-NEXT: return 0;
    // CHECK-NEXT: }());
    cublasErrCheck(cublasStrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N));


    float2 *d_A_C = 0;
    float2 *d_B_C = 0;
    float2 *d_C_C = 0;
    float2 alpha_C;
    float2 beta_C;
    const float2 *x_C = 0;
    float **Aarray_S = 0;
    int *PivotArray = 0;
    int *infoArray = 0;
    int batchSize = 10;


    // CHECK: /*
    // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT:cublasErrCheck([&](){
    // CHECK-NEXT:auto transpose_ct{{[0-9]+}} = trans0;
    // CHECK-NEXT:auto transpose_ct{{[0-9]+}} = trans1;
    // CHECK-NEXT:auto d_A_C_buff_ct1 = dpct::get_buffer<std::complex<float>>(d_A_C);
    // CHECK-NEXT:auto d_B_C_buff_ct1 = dpct::get_buffer<std::complex<float>>(d_B_C);
    // CHECK-NEXT:auto d_C_C_buff_ct1 = dpct::get_buffer<std::complex<float>>(d_C_C);
    // CHECK-NEXT:mkl::blas::gemm(handle, (((int)transpose_ct{{[0-9]+}})==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct{{[0-9]+}})), (((int)transpose_ct{{[0-9]+}})==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct{{[0-9]+}})), N, N, N, std::complex<float>((&alpha_C)->x(),(&alpha_C)->y()), d_A_C_buff_ct1, N, d_B_C_buff_ct1, N, std::complex<float>((&beta_C)->x(),(&beta_C)->y()), d_C_C_buff_ct1, N);
    // CHECK-NEXT:return 0;
    // CHECK-NEXT:}());
    cublasErrCheck(cublasCgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_C, d_A_C, N, d_B_C, N, &beta_C, d_C_C, N));

    // CHECK: /*
    // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT:cublasErrCheck([&](){
    // CHECK-NEXT:auto x_C_buff_ct1 = dpct::get_buffer<std::complex<float>>(x_C);
    // CHECK-NEXT:auto result_buff_ct1 = dpct::get_buffer<int>(result);
    // CHECK-NEXT:sycl::buffer<int64_t> result_temp_buffer(sycl::range<1>(1));
    // CHECK-NEXT:mkl::blas::iamax(handle, N, x_C_buff_ct1, N, result_temp_buffer);
    // CHECK-NEXT:result_buff_ct1.get_access<sycl::access::mode::write>()[0] = (int)result_temp_buffer.get_access<sycl::access::mode::read>()[0];
    // CHECK-NEXT:return 0;
    // CHECK-NEXT:}());
    cublasErrCheck(cublasIcamax(handle, N, x_C, N, result));

    // CHECK: /*
    // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT:cublasErrCheck([&](){
    // CHECK-NEXT:auto transpose_ct{{[0-9]+}} = trans0;
    // CHECK-NEXT:auto ptr_ct{{[0-9]+}} = d_A_C;
    // CHECK-NEXT:auto ptr_ct{{[0-9]+}}_buff_ct1 = dpct::get_buffer<std::complex<float>>(ptr_ct{{[0-9]+}});
    // CHECK-NEXT:auto ptr_ct{{[0-9]+}} = d_C_C;
    // CHECK-NEXT:auto ptr_ct{{[0-9]+}}_buff_ct1 = dpct::get_buffer<std::complex<float>>(ptr_ct{{[0-9]+}});
    // CHECK-NEXT:auto ld_ct{{[0-9]+}} = N; auto m_ct{{[0-9]+}} = N; auto n_ct{{[0-9]+}} = N;
    // CHECK-NEXT:dpct::matrix_mem_copy(ptr_ct{{[0-9]+}}, d_B_C, ld_ct{{[0-9]+}}, N, m_ct{{[0-9]+}}, n_ct{{[0-9]+}}, dpct::device_to_device, handle);
    // CHECK-NEXT:mkl::blas::trmm(handle, (mkl::side)side0, (((int)fill0)==0?(mkl::uplo::lower):(mkl::uplo::upper)), (((int)transpose_ct{{[0-9]+}})==2?(mkl::transpose::conjtrans):((mkl::transpose)transpose_ct{{[0-9]+}})), (mkl::diag)diag0, m_ct{{[0-9]+}}, n_ct{{[0-9]+}}, std::complex<float>((&alpha_C)->x(),(&alpha_C)->y()), ptr_ct{{[0-9]+}}_buff_ct1, N,  ptr_ct{{[0-9]+}}_buff_ct1, ld_ct{{[0-9]+}});
    // CHECK-NEXT:return 0;
    // CHECK-NEXT:}());
    cublasErrCheck(cublasCtrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_C, d_A_C, N, d_B_C, N, d_C_C, N));

    // CHECK: /*
    // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT:cublasErrCheck([&](){
    // CHECK-NEXT:dpct::getrf_batch_wrapper(handle, N, Aarray_S, N, PivotArray, infoArray, batchSize);
    // CHECK-NEXT:return 0;
    // CHECK-NEXT:}());
    cublasErrCheck(cublasSgetrfBatched(handle, N, Aarray_S, N, PivotArray, infoArray, batchSize));



    float * __restrict__ h_data;
    //CHECK:mkl::rng::philox4x32x10 rng(dpct::get_default_queue_wait(), 1337ull);
    //CHECK-NEXT:/*
    //CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to curandCreateGenerator was replaced with 0, because the function call is redundant in DPC++.
    //CHECK-NEXT:*/
    //CHECK-NEXT:curandErrCheck(0);
    //CHECK-NEXT:/*
    //CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was replaced with 0, because the function call is redundant in DPC++.
    //CHECK-NEXT:*/
    //CHECK-NEXT:curandErrCheck(0);
    //CHECK-NEXT:/*
    //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
    //CHECK-NEXT:*/
    //CHECK-NEXT:curandErrCheck([&](){
    //CHECK-NEXT:auto h_data_buff_ct1 = dpct::get_buffer<float>(h_data);
    //CHECK-NEXT:mkl::rng::uniform<float> distr_ct1;
    //CHECK-NEXT:mkl::rng::generate(distr_ct1, rng, (100 + 1) * (200) * 4, h_data_buff_ct1);
    //CHECK-NEXT:return 0;
    //CHECK-NEXT:}());
    //CHECK-NEXT:/*
    //CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to curandDestroyGenerator was replaced with 0, because the function call is redundant in DPC++.
    //CHECK-NEXT:*/
    //CHECK-NEXT:curandErrCheck(0);
    curandGenerator_t rng;
    curandErrCheck(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(rng, 1337ull));
    curandErrCheck(curandGenerateUniform(rng, h_data, (100 + 1) * (200) * 4));
    curandErrCheck(curandDestroyGenerator(rng));

}
