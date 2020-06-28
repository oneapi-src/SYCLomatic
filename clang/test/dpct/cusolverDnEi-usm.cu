// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusolverDnEi-usm.dp.cpp --match-full-lines %s
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


    //CHECK: mkl::lapack::gebrd(**cusolverH, m, n, (float*)&A_f, lda, (float*)&D_f, (float*)&E_f, (float*)&TAUQ_f, (float*)&TAUP_f, (float*)&workspace_f, Lwork);
    //CHECK-NEXT: mkl::lapack::gebrd(**cusolverH, m, n, (double*)&A_d, lda, (double*)&D_d, (double*)&E_d, (double*)&TAUQ_d, (double*)&TAUP_d, (double*)&workspace_d, Lwork);
    //CHECK-NEXT: mkl::lapack::gebrd(**cusolverH, m, n, (std::complex<float>*)&A_c, lda, (float*)&D_f, (float*)&E_f, (std::complex<float>*)&TAUQ_c, (std::complex<float>*)&TAUP_c, (std::complex<float>*)&workspace_c, Lwork);
    //CHECK-NEXT: mkl::lapack::gebrd(**cusolverH, m, n, (std::complex<double>*)&A_z, lda, (double*)&D_d, (double*)&E_d, (std::complex<double>*)&TAUQ_z, (std::complex<double>*)&TAUP_z, (std::complex<double>*)&workspace_z, Lwork);
    cusolverDnSgebrd(*cusolverH, m, n, &A_f, lda, &D_f, &E_f, &TAUQ_f, &TAUP_f, &workspace_f, Lwork, &devInfo);
    cusolverDnDgebrd(*cusolverH, m, n, &A_d, lda, &D_d, &E_d, &TAUQ_d, &TAUP_d, &workspace_d, Lwork, &devInfo);
    cusolverDnCgebrd(*cusolverH, m, n, &A_c, lda, &D_f, &E_f, &TAUQ_c, &TAUP_c, &workspace_c, Lwork, &devInfo);
    cusolverDnZgebrd(*cusolverH, m, n, &A_z, lda, &D_d, &E_d, &TAUQ_z, &TAUP_z, &workspace_z, Lwork, &devInfo);

    //CHECK: mkl::lapack::orgbr(**cusolverH, (mkl::generate)side, m, n, k, (float*)&A_f, lda, (float*)&TAU_f, (float*)&workspace_f, Lwork);
    //CHECK-NEXT: mkl::lapack::orgbr(**cusolverH, (mkl::generate)side, m, n, k, (double*)&A_d, lda, (double*)&TAU_d, (double*)&workspace_d, Lwork);
    //CHECK-NEXT: mkl::lapack::ungbr(**cusolverH, (mkl::generate)side, m, n, k, (std::complex<float>*)&A_c, lda, (std::complex<float>*)&TAU_c, (std::complex<float>*)&workspace_c, Lwork);
    //CHECK-NEXT: mkl::lapack::ungbr(**cusolverH, (mkl::generate)side, m, n, k, (std::complex<double>*)&A_z, lda, (std::complex<double>*)&TAU_z, (std::complex<double>*)&workspace_z, Lwork);
    cusolverDnSorgbr(*cusolverH, side, m, n, k, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);
    cusolverDnDorgbr(*cusolverH, side, m, n, k, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);
    cusolverDnCungbr(*cusolverH, side, m, n, k, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);
    cusolverDnZungbr(*cusolverH, side, m, n, k, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);

    //CHECK: mkl::lapack::sytrd(**cusolverH, uplo, n, (float*)&A_f, lda, (float*)&D_f, (float*)&E_f, (float*)&TAU_f, (float*)&workspace_f, Lwork);
    //CHECK-NEXT: mkl::lapack::sytrd(**cusolverH, uplo, n, (double*)&A_d, lda, (double*)&D_d, (double*)&E_d, (double*)&TAU_d, (double*)&workspace_d, Lwork);
    //CHECK-NEXT: mkl::lapack::hetrd(**cusolverH, uplo, n, (std::complex<float>*)&A_c, lda, (float*)&D_f, (float*)&E_f, (std::complex<float>*)&TAU_c, (std::complex<float>*)&workspace_c, Lwork);
    //CHECK-NEXT: mkl::lapack::hetrd(**cusolverH, uplo, n, (std::complex<double>*)&A_z, lda, (double*)&D_d, (double*)&E_d, (std::complex<double>*)&TAU_z, (std::complex<double>*)&workspace_z, Lwork);
    cusolverDnSsytrd(*cusolverH, uplo, n, &A_f, lda, &D_f, &E_f, &TAU_f, &workspace_f, Lwork, &devInfo);
    cusolverDnDsytrd(*cusolverH, uplo, n, &A_d, lda, &D_d, &E_d, &TAU_d, &workspace_d, Lwork, &devInfo);
    cusolverDnChetrd(*cusolverH, uplo, n, &A_c, lda, &D_f, &E_f, &TAU_c, &workspace_c, Lwork, &devInfo);
    cusolverDnZhetrd(*cusolverH, uplo, n, &A_z, lda, &D_d, &E_d, &TAU_z, &workspace_z, Lwork, &devInfo);

    //CHECK: mkl::lapack::ormtr(**cusolverH, side, uplo, trans, m, n, (float*)&A_f, lda, (float*)&TAU_f, (float*)&B_f, ldb, (float*)&workspace_f, Lwork);
    //CHECK-NEXT: mkl::lapack::ormtr(**cusolverH, side, uplo, trans, m, n, (double*)&A_d, lda, (double*)&TAU_d, (double*)&B_d, ldb, (double*)&workspace_d, Lwork);
    //CHECK-NEXT: mkl::lapack::unmtr(**cusolverH, side, uplo, trans, m, n, (std::complex<float>*)&A_c, lda, (std::complex<float>*)&TAU_c, (std::complex<float>*)&B_c, ldb, (std::complex<float>*)&workspace_c, Lwork);
    //CHECK-NEXT: mkl::lapack::unmtr(**cusolverH, side, uplo, trans, m, n, (std::complex<double>*)&A_z, lda, (std::complex<double>*)&TAU_z, (std::complex<double>*)&B_z, ldb, (std::complex<double>*)&workspace_z, Lwork);
    cusolverDnSormtr(*cusolverH, side, uplo, trans, m, n, &A_f, lda, &TAU_f, &B_f, ldb, &workspace_f, Lwork, &devInfo);
    cusolverDnDormtr(*cusolverH, side, uplo, trans, m, n, &A_d, lda, &TAU_d, &B_d, ldb, &workspace_d, Lwork, &devInfo);
    cusolverDnCunmtr(*cusolverH, side, uplo, trans, m, n, &A_c, lda, &TAU_c, &B_c, ldb, &workspace_c, Lwork, &devInfo);
    cusolverDnZunmtr(*cusolverH, side, uplo, trans, m, n, &A_z, lda, &TAU_z, &B_z, ldb, &workspace_z, Lwork, &devInfo);

    //CHECK: mkl::lapack::orgtr(**cusolverH, uplo, n, (float*)&A_f, lda, (float*)&TAU_f, (float*)&workspace_f, Lwork);
    //CHECK-NEXT: mkl::lapack::orgtr(**cusolverH, uplo, n, (double*)&A_d, lda, (double*)&TAU_d, (double*)&workspace_d, Lwork);
    //CHECK-NEXT: mkl::lapack::ungtr(**cusolverH, uplo, n, (std::complex<float>*)&A_c, lda, (std::complex<float>*)&TAU_c, (std::complex<float>*)&workspace_c, Lwork);
    //CHECK-NEXT: mkl::lapack::ungtr(**cusolverH, uplo, n, (std::complex<double>*)&A_z, lda, (std::complex<double>*)&TAU_z, (std::complex<double>*)&workspace_z, Lwork);
    cusolverDnSorgtr(*cusolverH, uplo, n, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);
    cusolverDnDorgtr(*cusolverH, uplo, n, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);
    cusolverDnCungtr(*cusolverH, uplo, n, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);
    cusolverDnZungtr(*cusolverH, uplo, n, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);

    //CHECK: mkl::lapack::gesvd (**cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, (float*)&A_f, lda, (float*)&S_f, (float*)&U_f, ldu, (float*)&VT_f, ldvt, (float*)&workspace_f, Lwork);
    //CHECK-NEXT: mkl::lapack::gesvd (**cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, (double*)&A_d, lda, (double*)&S_d, (double*)&U_d, ldu, (double*)&VT_d, ldvt, (double*)&workspace_d, Lwork);
    //CHECK-NEXT: mkl::lapack::gesvd (**cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, (std::complex<float>*)&A_c, lda, (float*)&S_f, (std::complex<float>*)&U_c, ldu, (std::complex<float>*)&VT_c, ldvt, (std::complex<float>*)&workspace_c, Lwork);
    //CHECK-NEXT: mkl::lapack::gesvd (**cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, (std::complex<double>*)&A_z, lda, (double*)&S_d, (std::complex<double>*)&U_z, ldu, (std::complex<double>*)&VT_z, ldvt, (std::complex<double>*)&workspace_z, Lwork);
    cusolverDnSgesvd (*cusolverH, jobu, jobvt, m, n, &A_f, lda, &S_f, &U_f, ldu, &VT_f, ldvt, &workspace_f, Lwork, &Rwork_f, &devInfo);
    cusolverDnDgesvd (*cusolverH, jobu, jobvt, m, n, &A_d, lda, &S_d, &U_d, ldu, &VT_d, ldvt, &workspace_d, Lwork, &Rwork_d, &devInfo);
    cusolverDnCgesvd (*cusolverH, jobu, jobvt, m, n, &A_c, lda, &S_f, &U_c, ldu, &VT_c, ldvt, &workspace_c, Lwork, &Rwork_f, &devInfo);
    cusolverDnZgesvd (*cusolverH, jobu, jobvt, m, n, &A_z, lda, &S_d, &U_z, ldu, &VT_z, ldvt, &workspace_z, Lwork, &Rwork_d, &devInfo);

}
