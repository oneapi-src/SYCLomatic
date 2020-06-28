// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusolverDnEi.dp.cpp --match-full-lines %s
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
    cublasOperation_t trans = CUBLAS_OP_N;
    cublasSideMode_t side = CUBLAS_SIDE_LEFT;
    cusolverEigMode_t jobz;

    int m = 0;
    int n = 0;
    int k = 0;
    int nrhs = 0;
    float A_f = 0;
    double A_d = 0.0;
    cuComplex A_c = make_cuComplex(1,0);
    cuDoubleComplex A_z = make_cuDoubleComplex(1,0);

    float B_f = 0;
    double B_d = 0.0;
    cuComplex B_c = make_cuComplex(1,0);
    cuDoubleComplex B_z = make_cuDoubleComplex(1,0);

    float D_f = 0;
    double D_d = 0.0;
    cuComplex D_c = make_cuComplex(1,0);
    cuDoubleComplex D_z = make_cuDoubleComplex(1,0);

    float E_f = 0;
    double E_d = 0.0;
    cuComplex E_c = make_cuComplex(1,0);
    cuDoubleComplex E_z = make_cuDoubleComplex(1,0);

    float TAU_f = 0;
    double TAU_d = 0.0;
    cuComplex TAU_c = make_cuComplex(1,0);
    cuDoubleComplex TAU_z = make_cuDoubleComplex(1,0);

    float TAUQ_f = 0;
    double TAUQ_d = 0.0;
    cuComplex TAUQ_c = make_cuComplex(1,0);
    cuDoubleComplex TAUQ_z = make_cuDoubleComplex(1,0);

    float TAUP_f = 0;
    double TAUP_d = 0.0;
    cuComplex TAUP_c = make_cuComplex(1,0);
    cuDoubleComplex TAUP_z = make_cuDoubleComplex(1,0);

    const float C_f = 0;
    const double C_d = 0.0;
    const cuComplex C_c = make_cuComplex(1,0);
    const cuDoubleComplex C_z = make_cuDoubleComplex(1,0);

    int lda = 0;
    int ldb = 0;
    const int ldc = 0;
    float workspace_f = 0;
    double workspace_d = 0;
    cuComplex workspace_c = make_cuComplex(1,0);
    cuDoubleComplex workspace_z = make_cuDoubleComplex(1,0);
    int Lwork = 0;
    int devInfo = 0;
    int devIpiv = 0;

    signed char jobu;
    signed char jobvt;

    float S_f = 0;
    double S_d = 0.0;
    cuComplex S_c = make_cuComplex(1,0);
    cuDoubleComplex S_z = make_cuDoubleComplex(1,0);

    float U_f = 0;
    double U_d = 0.0;
    cuComplex U_c = make_cuComplex(1,0);
    cuDoubleComplex U_z = make_cuDoubleComplex(1,0);
    int ldu;

    float VT_f = 0;
    double VT_d = 0.0;
    cuComplex VT_c = make_cuComplex(1,0);
    cuDoubleComplex VT_z = make_cuDoubleComplex(1,0);
    int ldvt;

    float Rwork_f = 0;
    double Rwork_d = 0.0;
    cuComplex Rwork_c = make_cuComplex(1,0);
    cuDoubleComplex Rwork_z = make_cuDoubleComplex(1,0);

    float W_f = 0;
    double W_d = 0.0;
    cuComplex W_c = make_cuComplex(1,0);
    cuDoubleComplex W_z = make_cuDoubleComplex(1,0);

    //CHECK: {
    //CHECK-NEXT: std::int64_t lda_ct;
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::gebrd_scratchpad_size<float>(**cusolverH, m, n, lda_ct), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: std::int64_t lda_ct;
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::gebrd_scratchpad_size<float>(**cusolverH, m, n, lda_ct);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&A_f);
    //CHECK-NEXT: auto D_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&D_f);
    //CHECK-NEXT: auto E_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&E_f);
    //CHECK-NEXT: auto TAUQ_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&TAUQ_f);
    //CHECK-NEXT: auto TAUP_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&TAUP_f);
    //CHECK-NEXT: auto workspace_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&workspace_f);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::gebrd(**cusolverH, m, n, A_f_buf_ct{{[0-9]+}}, lda, D_f_buf_ct{{[0-9]+}}, E_f_buf_ct{{[0-9]+}}, TAUQ_f_buf_ct{{[0-9]+}}, TAUP_f_buf_ct{{[0-9]+}}, workspace_f_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&A_f);
    //CHECK-NEXT: auto D_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&D_f);
    //CHECK-NEXT: auto E_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&E_f);
    //CHECK-NEXT: auto TAUQ_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&TAUQ_f);
    //CHECK-NEXT: auto TAUP_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&TAUP_f);
    //CHECK-NEXT: auto workspace_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&workspace_f);
    //CHECK-NEXT: mkl::lapack::gebrd(**cusolverH, m, n, A_f_buf_ct{{[0-9]+}}, lda, D_f_buf_ct{{[0-9]+}}, E_f_buf_ct{{[0-9]+}}, TAUQ_f_buf_ct{{[0-9]+}}, TAUP_f_buf_ct{{[0-9]+}}, workspace_f_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnSgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnSgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnSgebrd(*cusolverH, m, n, &A_f, lda, &D_f, &E_f, &TAUQ_f, &TAUP_f, &workspace_f, Lwork, &devInfo);
    cusolverDnSgebrd(*cusolverH, m, n, &A_f, lda, &D_f, &E_f, &TAUQ_f, &TAUP_f, &workspace_f, Lwork, &devInfo);

    //CHECK: {
    //CHECK-NEXT: std::int64_t lda_ct;
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::gebrd_scratchpad_size<double>(**cusolverH, m, n, lda_ct), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: std::int64_t lda_ct;
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::gebrd_scratchpad_size<double>(**cusolverH, m, n, lda_ct);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&A_d);
    //CHECK-NEXT: auto D_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&D_d);
    //CHECK-NEXT: auto E_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&E_d);
    //CHECK-NEXT: auto TAUQ_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&TAUQ_d);
    //CHECK-NEXT: auto TAUP_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&TAUP_d);
    //CHECK-NEXT: auto workspace_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&workspace_d);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::gebrd(**cusolverH, m, n, A_d_buf_ct{{[0-9]+}}, lda, D_d_buf_ct{{[0-9]+}}, E_d_buf_ct{{[0-9]+}}, TAUQ_d_buf_ct{{[0-9]+}}, TAUP_d_buf_ct{{[0-9]+}}, workspace_d_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&A_d);
    //CHECK-NEXT: auto D_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&D_d);
    //CHECK-NEXT: auto E_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&E_d);
    //CHECK-NEXT: auto TAUQ_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&TAUQ_d);
    //CHECK-NEXT: auto TAUP_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&TAUP_d);
    //CHECK-NEXT: auto workspace_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&workspace_d);
    //CHECK-NEXT: mkl::lapack::gebrd(**cusolverH, m, n, A_d_buf_ct{{[0-9]+}}, lda, D_d_buf_ct{{[0-9]+}}, E_d_buf_ct{{[0-9]+}}, TAUQ_d_buf_ct{{[0-9]+}}, TAUP_d_buf_ct{{[0-9]+}}, workspace_d_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnDgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnDgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnDgebrd(*cusolverH, m, n, &A_d, lda, &D_d, &E_d, &TAUQ_d, &TAUP_d, &workspace_d, Lwork, &devInfo);
    cusolverDnDgebrd(*cusolverH, m, n, &A_d, lda, &D_d, &E_d, &TAUQ_d, &TAUP_d, &workspace_d, Lwork, &devInfo);

    //CHECK: {
    //CHECK-NEXT: std::int64_t lda_ct;
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::gebrd_scratchpad_size<std::complex<float>>(**cusolverH, m, n, lda_ct), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: std::int64_t lda_ct;
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::gebrd_scratchpad_size<std::complex<float>>(**cusolverH, m, n, lda_ct);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&A_c);
    //CHECK-NEXT: auto D_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&D_f);
    //CHECK-NEXT: auto E_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&E_f);
    //CHECK-NEXT: auto TAUQ_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&TAUQ_c);
    //CHECK-NEXT: auto TAUP_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&TAUP_c);
    //CHECK-NEXT: auto workspace_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&workspace_c);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::gebrd(**cusolverH, m, n, A_c_buf_ct{{[0-9]+}}, lda, D_f_buf_ct{{[0-9]+}}, E_f_buf_ct{{[0-9]+}}, TAUQ_c_buf_ct{{[0-9]+}}, TAUP_c_buf_ct{{[0-9]+}}, workspace_c_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&A_c);
    //CHECK-NEXT: auto D_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&D_f);
    //CHECK-NEXT: auto E_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&E_f);
    //CHECK-NEXT: auto TAUQ_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&TAUQ_c);
    //CHECK-NEXT: auto TAUP_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&TAUP_c);
    //CHECK-NEXT: auto workspace_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&workspace_c);
    //CHECK-NEXT: mkl::lapack::gebrd(**cusolverH, m, n, A_c_buf_ct{{[0-9]+}}, lda, D_f_buf_ct{{[0-9]+}}, E_f_buf_ct{{[0-9]+}}, TAUQ_c_buf_ct{{[0-9]+}}, TAUP_c_buf_ct{{[0-9]+}}, workspace_c_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnCgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnCgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnCgebrd(*cusolverH, m, n, &A_c, lda, &D_f, &E_f, &TAUQ_c, &TAUP_c, &workspace_c, Lwork, &devInfo);
    cusolverDnCgebrd(*cusolverH, m, n, &A_c, lda, &D_f, &E_f, &TAUQ_c, &TAUP_c, &workspace_c, Lwork, &devInfo);

    //CHECK: {
    //CHECK-NEXT: std::int64_t lda_ct;
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::gebrd_scratchpad_size<std::complex<double>>(**cusolverH, m, n, lda_ct), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: std::int64_t lda_ct;
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::gebrd_scratchpad_size<std::complex<double>>(**cusolverH, m, n, lda_ct);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&A_z);
    //CHECK-NEXT: auto D_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&D_d);
    //CHECK-NEXT: auto E_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&E_d);
    //CHECK-NEXT: auto TAUQ_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&TAUQ_z);
    //CHECK-NEXT: auto TAUP_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&TAUP_z);
    //CHECK-NEXT: auto workspace_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&workspace_z);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::gebrd(**cusolverH, m, n, A_z_buf_ct{{[0-9]+}}, lda, D_d_buf_ct{{[0-9]+}}, E_d_buf_ct{{[0-9]+}}, TAUQ_z_buf_ct{{[0-9]+}}, TAUP_z_buf_ct{{[0-9]+}}, workspace_z_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&A_z);
    //CHECK-NEXT: auto D_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&D_d);
    //CHECK-NEXT: auto E_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&E_d);
    //CHECK-NEXT: auto TAUQ_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&TAUQ_z);
    //CHECK-NEXT: auto TAUP_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&TAUP_z);
    //CHECK-NEXT: auto workspace_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&workspace_z);
    //CHECK-NEXT: mkl::lapack::gebrd(**cusolverH, m, n, A_z_buf_ct{{[0-9]+}}, lda, D_d_buf_ct{{[0-9]+}}, E_d_buf_ct{{[0-9]+}}, TAUQ_z_buf_ct{{[0-9]+}}, TAUP_z_buf_ct{{[0-9]+}}, workspace_z_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnZgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnZgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnZgebrd(*cusolverH, m, n, &A_z, lda, &D_d, &E_d, &TAUQ_z, &TAUP_z, &workspace_z, Lwork, &devInfo);
    cusolverDnZgebrd(*cusolverH, m, n, &A_z, lda, &D_d, &E_d, &TAUQ_z, &TAUP_z, &workspace_z, Lwork, &devInfo);


    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::orgbr_scratchpad_size<float>(**cusolverH, (mkl::generate)side, m, n, k, lda), 0);
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::orgbr_scratchpad_size<float>(**cusolverH, (mkl::generate)side, m, n, k, lda);
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&A_f);
    //CHECK-NEXT: auto TAU_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&TAU_f);
    //CHECK-NEXT: auto workspace_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&workspace_f);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::orgbr(**cusolverH, (mkl::generate)side, m, n, k, A_f_buf_ct{{[0-9]+}}, lda, TAU_f_buf_ct{{[0-9]+}}, workspace_f_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&A_f);
    //CHECK-NEXT: auto TAU_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&TAU_f);
    //CHECK-NEXT: auto workspace_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&workspace_f);
    //CHECK-NEXT: mkl::lapack::orgbr(**cusolverH, (mkl::generate)side, m, n, k, A_f_buf_ct{{[0-9]+}}, lda, TAU_f_buf_ct{{[0-9]+}}, workspace_f_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnSorgbr_bufferSize(*cusolverH, side, m, n, k, &A_f, lda, &TAU_f, &Lwork);
    cusolverDnSorgbr_bufferSize(*cusolverH, side, m, n, k, &A_f, lda, &TAU_f, &Lwork);
    status = cusolverDnSorgbr(*cusolverH, side, m, n, k, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);
    cusolverDnSorgbr(*cusolverH, side, m, n, k, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);


    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::orgbr_scratchpad_size<double>(**cusolverH, (mkl::generate)side, m, n, k, lda), 0);
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::orgbr_scratchpad_size<double>(**cusolverH, (mkl::generate)side, m, n, k, lda);
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&A_d);
    //CHECK-NEXT: auto TAU_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&TAU_d);
    //CHECK-NEXT: auto workspace_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&workspace_d);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::orgbr(**cusolverH, (mkl::generate)side, m, n, k, A_d_buf_ct{{[0-9]+}}, lda, TAU_d_buf_ct{{[0-9]+}}, workspace_d_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&A_d);
    //CHECK-NEXT: auto TAU_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&TAU_d);
    //CHECK-NEXT: auto workspace_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&workspace_d);
    //CHECK-NEXT: mkl::lapack::orgbr(**cusolverH, (mkl::generate)side, m, n, k, A_d_buf_ct{{[0-9]+}}, lda, TAU_d_buf_ct{{[0-9]+}}, workspace_d_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnDorgbr_bufferSize(*cusolverH, side, m, n, k, &A_d, lda, &TAU_d, &Lwork);
    cusolverDnDorgbr_bufferSize(*cusolverH, side, m, n, k, &A_d, lda, &TAU_d, &Lwork);
    status = cusolverDnDorgbr(*cusolverH, side, m, n, k, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);
    cusolverDnDorgbr(*cusolverH, side, m, n, k, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);

    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::ungbr_scratchpad_size<std::complex<float>>(**cusolverH, (mkl::generate)side, m, n, k, lda), 0);
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::ungbr_scratchpad_size<std::complex<float>>(**cusolverH, (mkl::generate)side, m, n, k, lda);
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&A_c);
    //CHECK-NEXT: auto TAU_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&TAU_c);
    //CHECK-NEXT: auto workspace_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&workspace_c);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::ungbr(**cusolverH, (mkl::generate)side, m, n, k, A_c_buf_ct{{[0-9]+}}, lda, TAU_c_buf_ct{{[0-9]+}}, workspace_c_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&A_c);
    //CHECK-NEXT: auto TAU_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&TAU_c);
    //CHECK-NEXT: auto workspace_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&workspace_c);
    //CHECK-NEXT: mkl::lapack::ungbr(**cusolverH, (mkl::generate)side, m, n, k, A_c_buf_ct{{[0-9]+}}, lda, TAU_c_buf_ct{{[0-9]+}}, workspace_c_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnCungbr_bufferSize(*cusolverH, side, m, n, k, &A_c, lda, &TAU_c, &Lwork);
    cusolverDnCungbr_bufferSize(*cusolverH, side, m, n, k, &A_c, lda, &TAU_c, &Lwork);
    status = cusolverDnCungbr(*cusolverH, side, m, n, k, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);
    cusolverDnCungbr(*cusolverH, side, m, n, k, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);

    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::ungbr_scratchpad_size<std::complex<double>>(**cusolverH, (mkl::generate)side, m, n, k, lda), 0);
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::ungbr_scratchpad_size<std::complex<double>>(**cusolverH, (mkl::generate)side, m, n, k, lda);
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&A_z);
    //CHECK-NEXT: auto TAU_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&TAU_z);
    //CHECK-NEXT: auto workspace_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&workspace_z);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::ungbr(**cusolverH, (mkl::generate)side, m, n, k, A_z_buf_ct{{[0-9]+}}, lda, TAU_z_buf_ct{{[0-9]+}}, workspace_z_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&A_z);
    //CHECK-NEXT: auto TAU_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&TAU_z);
    //CHECK-NEXT: auto workspace_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&workspace_z);
    //CHECK-NEXT: mkl::lapack::ungbr(**cusolverH, (mkl::generate)side, m, n, k, A_z_buf_ct{{[0-9]+}}, lda, TAU_z_buf_ct{{[0-9]+}}, workspace_z_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnZungbr_bufferSize(*cusolverH, side, m, n, k, &A_z, lda, &TAU_z, &Lwork);
    cusolverDnZungbr_bufferSize(*cusolverH, side, m, n, k, &A_z, lda, &TAU_z, &Lwork);
    status = cusolverDnZungbr(*cusolverH, side, m, n, k, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);
    cusolverDnZungbr(*cusolverH, side, m, n, k, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);


    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::sytrd_scratchpad_size<float>(**cusolverH, uplo, n, lda), 0);
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::sytrd_scratchpad_size<float>(**cusolverH, uplo, n, lda);
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&A_f);
    //CHECK-NEXT: auto D_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&D_f);
    //CHECK-NEXT: auto E_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&E_f);
    //CHECK-NEXT: auto TAU_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&TAU_f);
    //CHECK-NEXT: auto workspace_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&workspace_f);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::sytrd(**cusolverH, uplo, n, A_f_buf_ct{{[0-9]+}}, lda, D_f_buf_ct{{[0-9]+}}, E_f_buf_ct{{[0-9]+}}, TAU_f_buf_ct{{[0-9]+}}, workspace_f_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&A_f);
    //CHECK-NEXT: auto D_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&D_f);
    //CHECK-NEXT: auto E_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&E_f);
    //CHECK-NEXT: auto TAU_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&TAU_f);
    //CHECK-NEXT: auto workspace_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&workspace_f);
    //CHECK-NEXT: mkl::lapack::sytrd(**cusolverH, uplo, n, A_f_buf_ct{{[0-9]+}}, lda, D_f_buf_ct{{[0-9]+}}, E_f_buf_ct{{[0-9]+}}, TAU_f_buf_ct{{[0-9]+}}, workspace_f_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnSsytrd_bufferSize(*cusolverH, uplo, n, &A_f, lda, &D_f, &E_f, &TAU_f, &Lwork);
    cusolverDnSsytrd_bufferSize(*cusolverH, uplo, n, &A_f, lda, &D_f, &E_f, &TAU_f, &Lwork);
    status = cusolverDnSsytrd(*cusolverH, uplo, n, &A_f, lda, &D_f, &E_f, &TAU_f, &workspace_f, Lwork, &devInfo);
    cusolverDnSsytrd(*cusolverH, uplo, n, &A_f, lda, &D_f, &E_f, &TAU_f, &workspace_f, Lwork, &devInfo);

    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::sytrd_scratchpad_size<double>(**cusolverH, uplo, n, lda), 0);
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::sytrd_scratchpad_size<double>(**cusolverH, uplo, n, lda);
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&A_d);
    //CHECK-NEXT: auto D_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&D_d);
    //CHECK-NEXT: auto E_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&E_d);
    //CHECK-NEXT: auto TAU_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&TAU_d);
    //CHECK-NEXT: auto workspace_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&workspace_d);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::sytrd(**cusolverH, uplo, n, A_d_buf_ct{{[0-9]+}}, lda, D_d_buf_ct{{[0-9]+}}, E_d_buf_ct{{[0-9]+}}, TAU_d_buf_ct{{[0-9]+}}, workspace_d_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&A_d);
    //CHECK-NEXT: auto D_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&D_d);
    //CHECK-NEXT: auto E_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&E_d);
    //CHECK-NEXT: auto TAU_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&TAU_d);
    //CHECK-NEXT: auto workspace_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&workspace_d);
    //CHECK-NEXT: mkl::lapack::sytrd(**cusolverH, uplo, n, A_d_buf_ct{{[0-9]+}}, lda, D_d_buf_ct{{[0-9]+}}, E_d_buf_ct{{[0-9]+}}, TAU_d_buf_ct{{[0-9]+}}, workspace_d_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnDsytrd_bufferSize(*cusolverH, uplo, n, &A_d, lda, &D_d, &E_d, &TAU_d, &Lwork);
    cusolverDnDsytrd_bufferSize(*cusolverH, uplo, n, &A_d, lda, &D_d, &E_d, &TAU_d, &Lwork);
    status = cusolverDnDsytrd(*cusolverH, uplo, n, &A_d, lda, &D_d, &E_d, &TAU_d, &workspace_d, Lwork, &devInfo);
    cusolverDnDsytrd(*cusolverH, uplo, n, &A_d, lda, &D_d, &E_d, &TAU_d, &workspace_d, Lwork, &devInfo);

    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::hetrd_scratchpad_size<std::complex<float>>(**cusolverH, uplo, n, lda), 0);
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::hetrd_scratchpad_size<std::complex<float>>(**cusolverH, uplo, n, lda);
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&A_c);
    //CHECK-NEXT: auto D_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&D_f);
    //CHECK-NEXT: auto E_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&E_f);
    //CHECK-NEXT: auto TAU_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&TAU_c);
    //CHECK-NEXT: auto workspace_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&workspace_c);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::hetrd(**cusolverH, uplo, n, A_c_buf_ct{{[0-9]+}}, lda, D_f_buf_ct{{[0-9]+}}, E_f_buf_ct{{[0-9]+}}, TAU_c_buf_ct{{[0-9]+}}, workspace_c_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&A_c);
    //CHECK-NEXT: auto D_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&D_f);
    //CHECK-NEXT: auto E_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&E_f);
    //CHECK-NEXT: auto TAU_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&TAU_c);
    //CHECK-NEXT: auto workspace_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&workspace_c);
    //CHECK-NEXT: mkl::lapack::hetrd(**cusolverH, uplo, n, A_c_buf_ct{{[0-9]+}}, lda, D_f_buf_ct{{[0-9]+}}, E_f_buf_ct{{[0-9]+}}, TAU_c_buf_ct{{[0-9]+}}, workspace_c_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnChetrd_bufferSize(*cusolverH, uplo, n, &A_c, lda, &D_f, &E_f, &TAU_c, &Lwork);
    cusolverDnChetrd_bufferSize(*cusolverH, uplo, n, &A_c, lda, &D_f, &E_f, &TAU_c, &Lwork);
    status = cusolverDnChetrd(*cusolverH, uplo, n, &A_c, lda, &D_f, &E_f, &TAU_c, &workspace_c, Lwork, &devInfo);
    cusolverDnChetrd(*cusolverH, uplo, n, &A_c, lda, &D_f, &E_f, &TAU_c, &workspace_c, Lwork, &devInfo);

    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::hetrd_scratchpad_size<std::complex<double>>(**cusolverH, uplo, n, lda), 0);
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::hetrd_scratchpad_size<std::complex<double>>(**cusolverH, uplo, n, lda);
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&A_z);
    //CHECK-NEXT: auto D_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&D_d);
    //CHECK-NEXT: auto E_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&E_d);
    //CHECK-NEXT: auto TAU_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&TAU_z);
    //CHECK-NEXT: auto workspace_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&workspace_z);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::hetrd(**cusolverH, uplo, n, A_z_buf_ct{{[0-9]+}}, lda, D_d_buf_ct{{[0-9]+}}, E_d_buf_ct{{[0-9]+}}, TAU_z_buf_ct{{[0-9]+}}, workspace_z_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&A_z);
    //CHECK-NEXT: auto D_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&D_d);
    //CHECK-NEXT: auto E_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&E_d);
    //CHECK-NEXT: auto TAU_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&TAU_z);
    //CHECK-NEXT: auto workspace_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&workspace_z);
    //CHECK-NEXT: mkl::lapack::hetrd(**cusolverH, uplo, n, A_z_buf_ct{{[0-9]+}}, lda, D_d_buf_ct{{[0-9]+}}, E_d_buf_ct{{[0-9]+}}, TAU_z_buf_ct{{[0-9]+}}, workspace_z_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnZhetrd_bufferSize(*cusolverH, uplo, n, &A_z, lda, &D_d, &E_d, &TAU_z, &Lwork);
    cusolverDnZhetrd_bufferSize(*cusolverH, uplo, n, &A_z, lda, &D_d, &E_d, &TAU_z, &Lwork);
    status = cusolverDnZhetrd(*cusolverH, uplo, n, &A_z, lda, &D_d, &E_d, &TAU_z, &workspace_z, Lwork, &devInfo);
    cusolverDnZhetrd(*cusolverH, uplo, n, &A_z, lda, &D_d, &E_d, &TAU_z, &workspace_z, Lwork, &devInfo);

    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::ormtr_scratchpad_size<float>(**cusolverH, side, uplo, trans, m, n, lda, ldb), 0);
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::ormtr_scratchpad_size<float>(**cusolverH, side, uplo, trans, m, n, lda, ldb);
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&A_f);
    //CHECK-NEXT: auto TAU_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&TAU_f);
    //CHECK-NEXT: auto B_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&B_f);
    //CHECK-NEXT: auto workspace_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&workspace_f);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::ormtr(**cusolverH, side, uplo, trans, m, n, A_f_buf_ct{{[0-9]+}}, lda, TAU_f_buf_ct{{[0-9]+}}, B_f_buf_ct{{[0-9]+}}, ldb, workspace_f_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&A_f);
    //CHECK-NEXT: auto TAU_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&TAU_f);
    //CHECK-NEXT: auto B_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&B_f);
    //CHECK-NEXT: auto workspace_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&workspace_f);
    //CHECK-NEXT: mkl::lapack::ormtr(**cusolverH, side, uplo, trans, m, n, A_f_buf_ct{{[0-9]+}}, lda, TAU_f_buf_ct{{[0-9]+}}, B_f_buf_ct{{[0-9]+}}, ldb, workspace_f_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnSormtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_f, lda, &TAU_f, &B_f, ldb, &Lwork);
    cusolverDnSormtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_f, lda, &TAU_f, &B_f, ldb, &Lwork);
    status = cusolverDnSormtr(*cusolverH, side, uplo, trans, m, n, &A_f, lda, &TAU_f, &B_f, ldb, &workspace_f, Lwork, &devInfo);
    cusolverDnSormtr(*cusolverH, side, uplo, trans, m, n, &A_f, lda, &TAU_f, &B_f, ldb, &workspace_f, Lwork, &devInfo);

    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::ormtr_scratchpad_size<double>(**cusolverH, side, uplo, trans, m, n, lda, ldb), 0);
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::ormtr_scratchpad_size<double>(**cusolverH, side, uplo, trans, m, n, lda, ldb);
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&A_d);
    //CHECK-NEXT: auto TAU_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&TAU_d);
    //CHECK-NEXT: auto B_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&B_d);
    //CHECK-NEXT: auto workspace_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&workspace_d);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::ormtr(**cusolverH, side, uplo, trans, m, n, A_d_buf_ct{{[0-9]+}}, lda, TAU_d_buf_ct{{[0-9]+}}, B_d_buf_ct{{[0-9]+}}, ldb, workspace_d_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&A_d);
    //CHECK-NEXT: auto TAU_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&TAU_d);
    //CHECK-NEXT: auto B_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&B_d);
    //CHECK-NEXT: auto workspace_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&workspace_d);
    //CHECK-NEXT: mkl::lapack::ormtr(**cusolverH, side, uplo, trans, m, n, A_d_buf_ct{{[0-9]+}}, lda, TAU_d_buf_ct{{[0-9]+}}, B_d_buf_ct{{[0-9]+}}, ldb, workspace_d_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnDormtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_d, lda, &TAU_d, &B_d, ldb, &Lwork);
    cusolverDnDormtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_d, lda, &TAU_d, &B_d, ldb, &Lwork);
    status = cusolverDnDormtr(*cusolverH, side, uplo, trans, m, n, &A_d, lda, &TAU_d, &B_d, ldb, &workspace_d, Lwork, &devInfo);
    cusolverDnDormtr(*cusolverH, side, uplo, trans, m, n, &A_d, lda, &TAU_d, &B_d, ldb, &workspace_d, Lwork, &devInfo);

    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::unmtr_scratchpad_size<std::complex<float>>(**cusolverH, side, uplo, trans, m, n, lda, ldb), 0);
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::unmtr_scratchpad_size<std::complex<float>>(**cusolverH, side, uplo, trans, m, n, lda, ldb);
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&A_c);
    //CHECK-NEXT: auto TAU_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&TAU_c);
    //CHECK-NEXT: auto B_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&B_c);
    //CHECK-NEXT: auto workspace_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&workspace_c);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::unmtr(**cusolverH, side, uplo, trans, m, n, A_c_buf_ct{{[0-9]+}}, lda, TAU_c_buf_ct{{[0-9]+}}, B_c_buf_ct{{[0-9]+}}, ldb, workspace_c_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&A_c);
    //CHECK-NEXT: auto TAU_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&TAU_c);
    //CHECK-NEXT: auto B_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&B_c);
    //CHECK-NEXT: auto workspace_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&workspace_c);
    //CHECK-NEXT: mkl::lapack::unmtr(**cusolverH, side, uplo, trans, m, n, A_c_buf_ct{{[0-9]+}}, lda, TAU_c_buf_ct{{[0-9]+}}, B_c_buf_ct{{[0-9]+}}, ldb, workspace_c_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnCunmtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_c, lda, &TAU_c, &B_c, ldb, &Lwork);
    cusolverDnCunmtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_c, lda, &TAU_c, &B_c, ldb, &Lwork);
    status = cusolverDnCunmtr(*cusolverH, side, uplo, trans, m, n, &A_c, lda, &TAU_c, &B_c, ldb, &workspace_c, Lwork, &devInfo);
    cusolverDnCunmtr(*cusolverH, side, uplo, trans, m, n, &A_c, lda, &TAU_c, &B_c, ldb, &workspace_c, Lwork, &devInfo);

    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::unmtr_scratchpad_size<std::complex<double>>(**cusolverH, side, uplo, trans, m, n, lda, ldb), 0);
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::unmtr_scratchpad_size<std::complex<double>>(**cusolverH, side, uplo, trans, m, n, lda, ldb);
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&A_z);
    //CHECK-NEXT: auto TAU_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&TAU_z);
    //CHECK-NEXT: auto B_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&B_z);
    //CHECK-NEXT: auto workspace_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&workspace_z);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::unmtr(**cusolverH, side, uplo, trans, m, n, A_z_buf_ct{{[0-9]+}}, lda, TAU_z_buf_ct{{[0-9]+}}, B_z_buf_ct{{[0-9]+}}, ldb, workspace_z_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&A_z);
    //CHECK-NEXT: auto TAU_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&TAU_z);
    //CHECK-NEXT: auto B_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&B_z);
    //CHECK-NEXT: auto workspace_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&workspace_z);
    //CHECK-NEXT: mkl::lapack::unmtr(**cusolverH, side, uplo, trans, m, n, A_z_buf_ct{{[0-9]+}}, lda, TAU_z_buf_ct{{[0-9]+}}, B_z_buf_ct{{[0-9]+}}, ldb, workspace_z_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnZunmtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_z, lda, &TAU_z, &B_z, ldb, &Lwork);
    cusolverDnZunmtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_z, lda, &TAU_z, &B_z, ldb, &Lwork);
    status = cusolverDnZunmtr(*cusolverH, side, uplo, trans, m, n, &A_z, lda, &TAU_z, &B_z, ldb, &workspace_z, Lwork, &devInfo);
    cusolverDnZunmtr(*cusolverH, side, uplo, trans, m, n, &A_z, lda, &TAU_z, &B_z, ldb, &workspace_z, Lwork, &devInfo);

    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::orgtr_scratchpad_size<float>(**cusolverH, uplo, n, lda), 0);
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::orgtr_scratchpad_size<float>(**cusolverH, uplo, n, lda);
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&A_f);
    //CHECK-NEXT: auto TAU_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&TAU_f);
    //CHECK-NEXT: auto workspace_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&workspace_f);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::orgtr(**cusolverH, uplo, n, A_f_buf_ct{{[0-9]+}}, lda, TAU_f_buf_ct{{[0-9]+}}, workspace_f_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&A_f);
    //CHECK-NEXT: auto TAU_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&TAU_f);
    //CHECK-NEXT: auto workspace_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&workspace_f);
    //CHECK-NEXT: mkl::lapack::orgtr(**cusolverH, uplo, n, A_f_buf_ct{{[0-9]+}}, lda, TAU_f_buf_ct{{[0-9]+}}, workspace_f_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnSorgtr_bufferSize(*cusolverH, uplo, n, &A_f, lda, &TAU_f, &Lwork);
    cusolverDnSorgtr_bufferSize(*cusolverH, uplo, n, &A_f, lda, &TAU_f, &Lwork);
    status = cusolverDnSorgtr(*cusolverH, uplo, n, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);
    cusolverDnSorgtr(*cusolverH, uplo, n, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);

    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::orgtr_scratchpad_size<double>(**cusolverH, uplo, n, lda), 0);
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::orgtr_scratchpad_size<double>(**cusolverH, uplo, n, lda);
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&A_d);
    //CHECK-NEXT: auto TAU_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&TAU_d);
    //CHECK-NEXT: auto workspace_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&workspace_d);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::orgtr(**cusolverH, uplo, n, A_d_buf_ct{{[0-9]+}}, lda, TAU_d_buf_ct{{[0-9]+}}, workspace_d_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&A_d);
    //CHECK-NEXT: auto TAU_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&TAU_d);
    //CHECK-NEXT: auto workspace_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&workspace_d);
    //CHECK-NEXT: mkl::lapack::orgtr(**cusolverH, uplo, n, A_d_buf_ct{{[0-9]+}}, lda, TAU_d_buf_ct{{[0-9]+}}, workspace_d_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnDorgtr_bufferSize(*cusolverH, uplo, n, &A_d, lda, &TAU_d, &Lwork);
    cusolverDnDorgtr_bufferSize(*cusolverH, uplo, n, &A_d, lda, &TAU_d, &Lwork);
    status = cusolverDnDorgtr(*cusolverH, uplo, n, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);
    cusolverDnDorgtr(*cusolverH, uplo, n, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);

    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::ungtr_scratchpad_size<std::complex<float>>(**cusolverH, uplo, n, lda), 0);
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::ungtr_scratchpad_size<std::complex<float>>(**cusolverH, uplo, n, lda);
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&A_c);
    //CHECK-NEXT: auto TAU_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&TAU_c);
    //CHECK-NEXT: auto workspace_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&workspace_c);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::ungtr(**cusolverH, uplo, n, A_c_buf_ct{{[0-9]+}}, lda, TAU_c_buf_ct{{[0-9]+}}, workspace_c_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&A_c);
    //CHECK-NEXT: auto TAU_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&TAU_c);
    //CHECK-NEXT: auto workspace_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&workspace_c);
    //CHECK-NEXT: mkl::lapack::ungtr(**cusolverH, uplo, n, A_c_buf_ct{{[0-9]+}}, lda, TAU_c_buf_ct{{[0-9]+}}, workspace_c_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnCungtr_bufferSize(*cusolverH, uplo, n, &A_c, lda, &TAU_c, &Lwork);
    cusolverDnCungtr_bufferSize(*cusolverH, uplo, n, &A_c, lda, &TAU_c, &Lwork);
    status = cusolverDnCungtr(*cusolverH, uplo, n, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);
    cusolverDnCungtr(*cusolverH, uplo, n, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);

    //CHECK: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::ungtr_scratchpad_size<std::complex<double>>(**cusolverH, uplo, n, lda), 0);
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::ungtr_scratchpad_size<std::complex<double>>(**cusolverH, uplo, n, lda);
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&A_z);
    //CHECK-NEXT: auto TAU_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&TAU_z);
    //CHECK-NEXT: auto workspace_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&workspace_z);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::ungtr(**cusolverH, uplo, n, A_z_buf_ct{{[0-9]+}}, lda, TAU_z_buf_ct{{[0-9]+}}, workspace_z_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&A_z);
    //CHECK-NEXT: auto TAU_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&TAU_z);
    //CHECK-NEXT: auto workspace_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&workspace_z);
    //CHECK-NEXT: mkl::lapack::ungtr(**cusolverH, uplo, n, A_z_buf_ct{{[0-9]+}}, lda, TAU_z_buf_ct{{[0-9]+}}, workspace_z_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnZungtr_bufferSize(*cusolverH, uplo, n, &A_z, lda, &TAU_z, &Lwork);
    cusolverDnZungtr_bufferSize(*cusolverH, uplo, n, &A_z, lda, &TAU_z, &Lwork);
    status = cusolverDnZungtr(*cusolverH, uplo, n, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);
    cusolverDnZungtr(*cusolverH, uplo, n, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);


    //CHECK: {
    //CHECK-NEXT: mkl::job job_ct_mkl_jobu;
    //CHECK-NEXT: mkl::job job_ct_mkl_jobvt;
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::gesvd_scratchpad_size<float>(**cusolverH, job_ct_mkl_jobu, job_ct_mkl_jobvt, m, n, m, m, n), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: mkl::job job_ct_mkl_jobu;
    //CHECK-NEXT: mkl::job job_ct_mkl_jobvt;
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::gesvd_scratchpad_size<float>(**cusolverH, job_ct_mkl_jobu, job_ct_mkl_jobvt, m, n, m, m, n);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&A_f);
    //CHECK-NEXT: auto S_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&S_f);
    //CHECK-NEXT: auto U_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&U_f);
    //CHECK-NEXT: auto VT_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&VT_f);
    //CHECK-NEXT: auto workspace_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&workspace_f);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::gesvd (**cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, A_f_buf_ct{{[0-9]+}}, lda, S_f_buf_ct{{[0-9]+}}, U_f_buf_ct{{[0-9]+}}, ldu, VT_f_buf_ct{{[0-9]+}}, ldvt, workspace_f_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&A_f);
    //CHECK-NEXT: auto S_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&S_f);
    //CHECK-NEXT: auto U_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&U_f);
    //CHECK-NEXT: auto VT_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&VT_f);
    //CHECK-NEXT: auto workspace_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&workspace_f);
    //CHECK-NEXT: mkl::lapack::gesvd (**cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, A_f_buf_ct{{[0-9]+}}, lda, S_f_buf_ct{{[0-9]+}}, U_f_buf_ct{{[0-9]+}}, ldu, VT_f_buf_ct{{[0-9]+}}, ldvt, workspace_f_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnSgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnSgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnSgesvd (*cusolverH, jobu, jobvt, m, n, &A_f, lda, &S_f, &U_f, ldu, &VT_f, ldvt, &workspace_f, Lwork, &Rwork_f, &devInfo);
    cusolverDnSgesvd (*cusolverH, jobu, jobvt, m, n, &A_f, lda, &S_f, &U_f, ldu, &VT_f, ldvt, &workspace_f, Lwork, &Rwork_f, &devInfo);

    //CHECK: {
    //CHECK-NEXT: mkl::job job_ct_mkl_jobu;
    //CHECK-NEXT: mkl::job job_ct_mkl_jobvt;
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::gesvd_scratchpad_size<double>(**cusolverH, job_ct_mkl_jobu, job_ct_mkl_jobvt, m, n, m, m, n), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: mkl::job job_ct_mkl_jobu;
    //CHECK-NEXT: mkl::job job_ct_mkl_jobvt;
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::gesvd_scratchpad_size<double>(**cusolverH, job_ct_mkl_jobu, job_ct_mkl_jobvt, m, n, m, m, n);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&A_d);
    //CHECK-NEXT: auto S_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&S_d);
    //CHECK-NEXT: auto U_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&U_d);
    //CHECK-NEXT: auto VT_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&VT_d);
    //CHECK-NEXT: auto workspace_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&workspace_d);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::gesvd (**cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, A_d_buf_ct{{[0-9]+}}, lda, S_d_buf_ct{{[0-9]+}}, U_d_buf_ct{{[0-9]+}}, ldu, VT_d_buf_ct{{[0-9]+}}, ldvt, workspace_d_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&A_d);
    //CHECK-NEXT: auto S_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&S_d);
    //CHECK-NEXT: auto U_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&U_d);
    //CHECK-NEXT: auto VT_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&VT_d);
    //CHECK-NEXT: auto workspace_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&workspace_d);
    //CHECK-NEXT: mkl::lapack::gesvd (**cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, A_d_buf_ct{{[0-9]+}}, lda, S_d_buf_ct{{[0-9]+}}, U_d_buf_ct{{[0-9]+}}, ldu, VT_d_buf_ct{{[0-9]+}}, ldvt, workspace_d_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnDgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnDgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnDgesvd (*cusolverH, jobu, jobvt, m, n, &A_d, lda, &S_d, &U_d, ldu, &VT_d, ldvt, &workspace_d, Lwork, &Rwork_d, &devInfo);
    cusolverDnDgesvd (*cusolverH, jobu, jobvt, m, n, &A_d, lda, &S_d, &U_d, ldu, &VT_d, ldvt, &workspace_d, Lwork, &Rwork_d, &devInfo);

    //CHECK: {
    //CHECK-NEXT: mkl::job job_ct_mkl_jobu;
    //CHECK-NEXT: mkl::job job_ct_mkl_jobvt;
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::gesvd_scratchpad_size<std::complex<float>>(**cusolverH, job_ct_mkl_jobu, job_ct_mkl_jobvt, m, n, m, m, n), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: mkl::job job_ct_mkl_jobu;
    //CHECK-NEXT: mkl::job job_ct_mkl_jobvt;
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::gesvd_scratchpad_size<std::complex<float>>(**cusolverH, job_ct_mkl_jobu, job_ct_mkl_jobvt, m, n, m, m, n);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&A_c);
    //CHECK-NEXT: auto S_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&S_f);
    //CHECK-NEXT: auto U_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&U_c);
    //CHECK-NEXT: auto VT_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&VT_c);
    //CHECK-NEXT: auto workspace_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&workspace_c);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::gesvd (**cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, A_c_buf_ct{{[0-9]+}}, lda, S_f_buf_ct{{[0-9]+}}, U_c_buf_ct{{[0-9]+}}, ldu, VT_c_buf_ct{{[0-9]+}}, ldvt, workspace_c_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&A_c);
    //CHECK-NEXT: auto S_f_buf_ct{{[0-9]+}} = dpct::get_buffer<float>(&S_f);
    //CHECK-NEXT: auto U_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&U_c);
    //CHECK-NEXT: auto VT_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&VT_c);
    //CHECK-NEXT: auto workspace_c_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<float>>(&workspace_c);
    //CHECK-NEXT: mkl::lapack::gesvd (**cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, A_c_buf_ct{{[0-9]+}}, lda, S_f_buf_ct{{[0-9]+}}, U_c_buf_ct{{[0-9]+}}, ldu, VT_c_buf_ct{{[0-9]+}}, ldvt, workspace_c_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnCgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnCgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnCgesvd (*cusolverH, jobu, jobvt, m, n, &A_c, lda, &S_f, &U_c, ldu, &VT_c, ldvt, &workspace_c, Lwork, &Rwork_f, &devInfo);
    cusolverDnCgesvd (*cusolverH, jobu, jobvt, m, n, &A_c, lda, &S_f, &U_c, ldu, &VT_c, ldvt, &workspace_c, Lwork, &Rwork_f, &devInfo);

    //CHECK: {
    //CHECK-NEXT: mkl::job job_ct_mkl_jobu;
    //CHECK-NEXT: mkl::job job_ct_mkl_jobvt;
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (*(&Lwork) = mkl::lapack::gesvd_scratchpad_size<std::complex<double>>(**cusolverH, job_ct_mkl_jobu, job_ct_mkl_jobvt, m, n, m, m, n), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: mkl::job job_ct_mkl_jobu;
    //CHECK-NEXT: mkl::job job_ct_mkl_jobvt;
    //CHECK-NEXT: *(&Lwork) = mkl::lapack::gesvd_scratchpad_size<std::complex<double>>(**cusolverH, job_ct_mkl_jobu, job_ct_mkl_jobvt, m, n, m, m, n);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&A_z);
    //CHECK-NEXT: auto S_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&S_d);
    //CHECK-NEXT: auto U_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&U_z);
    //CHECK-NEXT: auto VT_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&VT_z);
    //CHECK-NEXT: auto workspace_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&workspace_z);
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = (mkl::lapack::gesvd (**cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, A_z_buf_ct{{[0-9]+}}, lda, S_d_buf_ct{{[0-9]+}}, U_z_buf_ct{{[0-9]+}}, ldu, VT_z_buf_ct{{[0-9]+}}, ldvt, workspace_z_buf_ct{{[0-9]+}}, Lwork), 0);
    //CHECK-NEXT: }
    //CHECK-NEXT: {
    //CHECK-NEXT: auto A_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&A_z);
    //CHECK-NEXT: auto S_d_buf_ct{{[0-9]+}} = dpct::get_buffer<double>(&S_d);
    //CHECK-NEXT: auto U_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&U_z);
    //CHECK-NEXT: auto VT_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&VT_z);
    //CHECK-NEXT: auto workspace_z_buf_ct{{[0-9]+}} = dpct::get_buffer<std::complex<double>>(&workspace_z);
    //CHECK-NEXT: mkl::lapack::gesvd (**cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, A_z_buf_ct{{[0-9]+}}, lda, S_d_buf_ct{{[0-9]+}}, U_z_buf_ct{{[0-9]+}}, ldu, VT_z_buf_ct{{[0-9]+}}, ldvt, workspace_z_buf_ct{{[0-9]+}}, Lwork);
    //CHECK-NEXT: }
    status = cusolverDnZgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnZgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnZgesvd (*cusolverH, jobu, jobvt, m, n, &A_z, lda, &S_d, &U_z, ldu, &VT_z, ldvt, &workspace_z, Lwork, &Rwork_d, &devInfo);
    cusolverDnZgesvd (*cusolverH, jobu, jobvt, m, n, &A_z, lda, &S_d, &U_z, ldu, &VT_z, ldvt, &workspace_z, Lwork, &Rwork_d, &devInfo);

}
