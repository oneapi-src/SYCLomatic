// RUN: dpct --format-range=none --usm-level=none -out-root %T/cublas_curandInMacro %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cublas_curandInMacro/cublas_curandInMacro.dp.cpp --match-full-lines %s
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
    //CHECK:oneapi::mkl::rng::uniform<float> distr_ct{{[0-9]+}};
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

    // CHECK: dpct::queue_ptr stream1;
    // CHECK-NEXT: stream1 = dpct::get_current_device().create_queue();
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: cublasErrCheck((handle = stream1, 0));
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: cublasErrCheck((stream1 = handle, 0));
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cublasErrCheck(cublasSetStream(handle, stream1));
    cublasErrCheck(cublasGetStream(handle, &stream1));

    // CHECK: /*
    // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: cublasErrCheck([&](){
    // CHECK-NEXT: auto d_A_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_A_S);
    // CHECK-NEXT: auto d_B_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_B_S);
    // CHECK-NEXT: auto d_C_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_C_S);
    // CHECK-NEXT: oneapi::mkl::blas::column_major::gemm(*handle, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, trans1==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans1, N, N, N, alpha_S, d_A_S_buf_ct{{[0-9]+}}, N, d_B_S_buf_ct{{[0-9]+}}, N, beta_S, d_C_S_buf_ct{{[0-9]+}}, N);
    // CHECK-NEXT: return 0;
    // CHECK-NEXT: }());
    cublasErrCheck(cublasSgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N));


    // CHECK: /*
    // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: cublasErrCheck([&](){
    // CHECK-NEXT: auto x_S_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(x_S);
    // CHECK-NEXT: auto result_buf_ct{{[0-9]+}} = sycl::buffer<int>(sycl::range<1>(1));
    // CHECK-NEXT: sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
    // CHECK-NEXT: if (dpct::is_device_ptr(result)) {
    // CHECK-NEXT:   result_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(result);
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   result_buf_ct{{[0-9]+}} = sycl::buffer<int>(result, sycl::range<1>(1));
    // CHECK-NEXT: }
    // CHECK-NEXT: oneapi::mkl::blas::column_major::iamax(*handle, N, x_S_buf_ct{{[0-9]+}}, N, res_temp_buf_ct{{[0-9]+}});
    // CHECK-NEXT: result_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::write>()[0] = (int)res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0];
    // CHECK-NEXT: return 0;
    // CHECK-NEXT: }());
    cublasErrCheck(cublasIsamax(handle, N, x_S, N, result));


    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: cublasErrCheck((dpct::syrk(*handle, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans1), N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N), 0));
    cublasErrCheck(cublasSsyrkx(handle, (cublasFillMode_t)fill0, (cublasOperation_t)trans1, N, N, &alpha_S, d_A_S, N, d_B_S, N, &beta_S, d_C_S, N));


    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: cublasErrCheck((dpct::trmm(*handle, (oneapi::mkl::side)side0, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, N, N, &alpha_S, d_A_S, N, d_B_S, N, d_C_S, N), 0));
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
    // CHECK-NEXT:auto d_A_C_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(d_A_C);
    // CHECK-NEXT:auto d_B_C_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(d_B_C);
    // CHECK-NEXT:auto d_C_C_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(d_C_C);
    // CHECK-NEXT:oneapi::mkl::blas::column_major::gemm(*handle, trans0==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans0, trans1==2 ? oneapi::mkl::transpose::conjtrans : (oneapi::mkl::transpose)trans1, N, N, N, std::complex<float>(alpha_C.x(), alpha_C.y()), d_A_C_buf_ct{{[0-9]+}}, N, d_B_C_buf_ct{{[0-9]+}}, N, std::complex<float>(beta_C.x(), beta_C.y()), d_C_C_buf_ct{{[0-9]+}}, N);
    // CHECK-NEXT:return 0;
    // CHECK-NEXT:}());
    cublasErrCheck(cublasCgemm(handle, (cublasOperation_t)trans0, (cublasOperation_t)trans1, N, N, N, &alpha_C, d_A_C, N, d_B_C, N, &beta_C, d_C_C, N));

    // CHECK: /*
    // CHECK-NEXT: DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT:cublasErrCheck([&](){
    // CHECK-NEXT:auto x_C_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(x_C);
    // CHECK-NEXT:auto result_buf_ct{{[0-9]+}} = sycl::buffer<int>(sycl::range<1>(1));
    // CHECK-NEXT:sycl::buffer<int64_t> res_temp_buf_ct{{[0-9]+}}(sycl::range<1>(1));
    // CHECK-NEXT:if (dpct::is_device_ptr(result)) {
    // CHECK-NEXT:  result_buf_ct{{[0-9]+}} = dpct::get_buffer<int>(result);
    // CHECK-NEXT:} else {
    // CHECK-NEXT:  result_buf_ct{{[0-9]+}} = sycl::buffer<int>(result, sycl::range<1>(1));
    // CHECK-NEXT:}
    // CHECK-NEXT:oneapi::mkl::blas::column_major::iamax(*handle, N, x_C_buf_ct{{[0-9]+}}, N, res_temp_buf_ct{{[0-9]+}});
    // CHECK-NEXT:result_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::write>()[0] = (int)res_temp_buf_ct{{[0-9]+}}.get_access<sycl::access_mode::read>()[0];
    // CHECK-NEXT:return 0;
    // CHECK-NEXT:}());
    cublasErrCheck(cublasIcamax(handle, N, x_C, N, result));

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: cublasErrCheck((dpct::trmm(*handle, (oneapi::mkl::side)side0, fill0 == 0 ? oneapi::mkl::uplo::lower : oneapi::mkl::uplo::upper, dpct::get_transpose(trans0), (oneapi::mkl::diag)diag0, N, N, &alpha_C, d_A_C, N, d_B_C, N, d_C_C, N), 0));
    cublasErrCheck(cublasCtrmm(handle, (cublasSideMode_t)side0, (cublasFillMode_t)fill0, (cublasOperation_t)trans0, (cublasDiagType_t)diag0, N, N, &alpha_C, d_A_C, N, d_B_C, N, d_C_C, N));

    // CHECK: /*
    // CHECK-NEXT: DPCT1047:{{[0-9]+}}: The meaning of PivotArray in the dpct::getrf_batch_wrapper is different from the cublasSgetrfBatched. You may need to check the migrated code.
    // CHECK-NEXT: */
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT:cublasErrCheck((dpct::getrf_batch_wrapper(*handle, N, Aarray_S, N, PivotArray, infoArray, batchSize), 0));
    cublasErrCheck(cublasSgetrfBatched(handle, N, Aarray_S, N, PivotArray, infoArray, batchSize));



    float * __restrict__ d_data;
    //CHECK:std::shared_ptr<oneapi::mkl::rng::mcg59> rng;
    //CHECK-NEXT:curandErrCheck((rng = std::make_shared<oneapi::mkl::rng::mcg59>(dpct::get_default_queue(), 1337ull), 0));
    //CHECK-NEXT:/*
    //CHECK-NEXT:DPCT1027:{{[0-9]+}}: The call to curandSetPseudoRandomGeneratorSeed was replaced with 0 because this call is redundant in SYCL.
    //CHECK-NEXT:*/
    //CHECK-NEXT:curandErrCheck(0);
    //CHECK-NEXT:/*
    //CHECK-NEXT:DPCT1034:{{[0-9]+}}: Migrated API does not return error code. 0 is returned in the lambda. You may need to rewrite this code.
    //CHECK-NEXT:*/
    //CHECK-NEXT:curandErrCheck([&](){
    //CHECK-NEXT:auto d_data_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(d_data);
    //CHECK-NEXT:oneapi::mkl::rng::generate(distr_ct{{[0-9]+}}, *rng, (100 + 1) * (200) * 4, d_data_buf_ct{{[0-9]+}});
    //CHECK-NEXT:return 0;
    //CHECK-NEXT:}());
    //CHECK-NEXT:/*
    //CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT:*/
    //CHECK-NEXT:curandErrCheck((rng.reset(), 0));
    curandGenerator_t rng;
    curandErrCheck(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(rng, 1337ull));
    curandErrCheck(curandGenerateUniform(rng, d_data, (100 + 1) * (200) * 4));
    curandErrCheck(curandDestroyGenerator(rng));

}

