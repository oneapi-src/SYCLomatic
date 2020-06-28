// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.2, v10.0
// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusolverDnLn_cuda10-1.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>


int main(int argc, char *argv[])
{
    cusolverDnHandle_t* cusolverH = NULL;
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    status = CUSOLVER_STATUS_NOT_INITIALIZED;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    int m = 0;
    int n = 0;
    int nrhs = 0;
    float A_f = 0;
    double A_d = 0.0;
    cuComplex A_c = make_cuComplex(1,0);
    cuDoubleComplex A_z = make_cuDoubleComplex(1,0);
    float B_f = 0;
    double B_d = 0.0;
    cuComplex B_c = make_cuComplex(1,0);
    cuDoubleComplex B_z = make_cuDoubleComplex(1,0);

    const float C_f = 0;
    const double C_d = 0.0;
    const cuComplex C_c = make_cuComplex(1,0);
    const cuDoubleComplex C_z = make_cuDoubleComplex(1,0);

    int lda = 0;
    int ldb = 0;
    float workspace_f = 0;
    double workspace_d = 0;
    cuComplex workspace_c = make_cuComplex(1,0);
    cuDoubleComplex workspace_z = make_cuDoubleComplex(1,0);
    int Lwork = 0;
    int devInfo = 0;

    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::potri_scratchpad_size<float>(**cusolverH, uplo, n, lda), 0);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::potri_scratchpad_size<double>(**cusolverH, uplo, n, lda), 0);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::potri_scratchpad_size<std::complex<float>>(**cusolverH, uplo, n, lda), 0);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::potri_scratchpad_size<std::complex<double>>(**cusolverH, uplo, n, lda), 0);
    status = cusolverDnSpotri_bufferSize(*cusolverH, uplo, n, &A_f, lda, &Lwork);
    status = cusolverDnDpotri_bufferSize(*cusolverH, uplo, n, &A_d, lda, &Lwork);
    status = cusolverDnCpotri_bufferSize(*cusolverH, uplo, n, &A_c, lda, &Lwork);
    status = cusolverDnZpotri_bufferSize(*cusolverH, uplo, n, &A_z, lda, &Lwork);

    //CHECK: {
    //CHECK-NEXT: auto A_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&A_f);
    //CHECK-NEXT: auto workspace_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&workspace_f);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::potri(**cusolverH, uplo, n, A_f_buf_ct{{[0-9]+}}, lda, workspace_f_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&A_f);
    //CHECK-NEXT: auto workspace_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&workspace_f);
    //CHECK-NEXT: mkl::lapack::potri(**cusolverH, uplo, n, A_f_buf_ct{{[0-9]+}}, lda, workspace_f_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnSpotri(*cusolverH, uplo, n, &A_f, lda, &workspace_f, Lwork, &devInfo);
    cusolverDnSpotri(*cusolverH, uplo, n, &A_f, lda, &workspace_f, Lwork, &devInfo);

    //CHECK: {
    //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&A_d);
    //CHECK-NEXT: auto workspace_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&workspace_d);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::potri(**cusolverH, uplo, n, A_d_buf_ct{{[0-9]+}}, lda, workspace_d_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&A_d);
    //CHECK-NEXT: auto workspace_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&workspace_d);
    //CHECK-NEXT: mkl::lapack::potri(**cusolverH, uplo, n, A_d_buf_ct{{[0-9]+}}, lda, workspace_d_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnDpotri(*cusolverH, uplo, n, &A_d, lda, &workspace_d, Lwork, &devInfo);
    cusolverDnDpotri(*cusolverH, uplo, n, &A_d, lda, &workspace_d, Lwork, &devInfo);

    //CHECK: {
    //CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&A_c);
    //CHECK-NEXT: auto workspace_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&workspace_c);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::potri(**cusolverH, uplo, n, A_c_buf_ct{{[0-9]+}}, lda, workspace_c_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&A_c);
    //CHECK-NEXT: auto workspace_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&workspace_c);
    //CHECK-NEXT: mkl::lapack::potri(**cusolverH, uplo, n, A_c_buf_ct{{[0-9]+}}, lda, workspace_c_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnCpotri(*cusolverH, uplo, n, &A_c, lda, &workspace_c, Lwork, &devInfo);
    cusolverDnCpotri(*cusolverH, uplo, n, &A_c, lda, &workspace_c, Lwork, &devInfo);

    //CHECK: {
    //CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&A_z);
    //CHECK-NEXT: auto workspace_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&workspace_z);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::potri(**cusolverH, uplo, n, A_z_buf_ct{{[0-9]+}}, lda, workspace_z_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&A_z);
    //CHECK-NEXT: auto workspace_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&workspace_z);
    //CHECK-NEXT: mkl::lapack::potri(**cusolverH, uplo, n, A_z_buf_ct{{[0-9]+}}, lda, workspace_z_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnZpotri(*cusolverH, uplo, n, &A_z, lda, &workspace_z, Lwork, &devInfo);
    cusolverDnZpotri(*cusolverH, uplo, n, &A_z, lda, &workspace_z, Lwork, &devInfo);
}
