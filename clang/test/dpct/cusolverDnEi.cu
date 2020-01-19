// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
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

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::gebrd_get_lwork<float>((*cusolverH).get_device(), m, n, m, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::gebrd_get_lwork<float>((*cusolverH).get_device(), m, n, m, lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct3 = allocation_ct3.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct3.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&D_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct5 = allocation_ct5.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct5.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&E_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct6 = allocation_ct6.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct6.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAUQ_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct7.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&TAUP_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct8 = allocation_ct8.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct8.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&workspace_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct9 = allocation_ct9.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct9.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct11 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct11 = allocation_ct11.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct11.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer11(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::gebrd(*cusolverH, m, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, buffer_ct7, buffer_ct8, buffer_ct9, Lwork, result_temp_buffer11), 0);
    // CHECK-NEXT: buffer_ct11.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer11.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct3 = allocation_ct3.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct3.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&D_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct5 = allocation_ct5.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct5.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&E_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct6 = allocation_ct6.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct6.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAUQ_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct7.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&TAUP_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct8 = allocation_ct8.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct8.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&workspace_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct9 = allocation_ct9.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct9.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct11 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct11 = allocation_ct11.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct11.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer11(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::gebrd(*cusolverH, m, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, buffer_ct7, buffer_ct8, buffer_ct9, Lwork, result_temp_buffer11);
    // CHECK-NEXT: buffer_ct11.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer11.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnSgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnSgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnSgebrd(*cusolverH, m, n, &A_f, lda, &D_f, &E_f, &TAUQ_f, &TAUP_f, &workspace_f, Lwork, &devInfo);
    cusolverDnSgebrd(*cusolverH, m, n, &A_f, lda, &D_f, &E_f, &TAUQ_f, &TAUP_f, &workspace_f, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::gebrd_get_lwork<double>((*cusolverH).get_device(), m, n, m, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::gebrd_get_lwork<double>((*cusolverH).get_device(), m, n, m, lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct3 = allocation_ct3.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct3.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&D_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct5 = allocation_ct5.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct5.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&E_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct6 = allocation_ct6.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct6.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAUQ_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct7 = allocation_ct7.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct7.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&TAUP_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct8 = allocation_ct8.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct8.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&workspace_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct9 = allocation_ct9.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct9.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct11 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct11 = allocation_ct11.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct11.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer11(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::gebrd(*cusolverH, m, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, buffer_ct7, buffer_ct8, buffer_ct9, Lwork, result_temp_buffer11), 0);
    // CHECK-NEXT: buffer_ct11.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer11.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct3 = allocation_ct3.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct3.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&D_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct5 = allocation_ct5.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct5.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&E_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct6 = allocation_ct6.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct6.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAUQ_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct7 = allocation_ct7.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct7.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&TAUP_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct8 = allocation_ct8.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct8.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&workspace_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct9 = allocation_ct9.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct9.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct11 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct11 = allocation_ct11.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct11.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer11(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::gebrd(*cusolverH, m, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, buffer_ct7, buffer_ct8, buffer_ct9, Lwork, result_temp_buffer11);
    // CHECK-NEXT: buffer_ct11.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer11.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnDgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnDgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnDgebrd(*cusolverH, m, n, &A_d, lda, &D_d, &E_d, &TAUQ_d, &TAUP_d, &workspace_d, Lwork, &devInfo);
    cusolverDnDgebrd(*cusolverH, m, n, &A_d, lda, &D_d, &E_d, &TAUQ_d, &TAUP_d, &workspace_d, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::gebrd_get_lwork<std::complex<float>>((*cusolverH).get_device(), m, n, m, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::gebrd_get_lwork<std::complex<float>>((*cusolverH).get_device(), m, n, m, lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&D_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct5 = allocation_ct5.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct5.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&E_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct6 = allocation_ct6.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct6.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAUQ_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&TAUP_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&workspace_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct9 = allocation_ct9.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct9.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct11 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct11 = allocation_ct11.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct11.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer11(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::gebrd(*cusolverH, m, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, buffer_ct7, buffer_ct8, buffer_ct9, Lwork, result_temp_buffer11), 0);
    // CHECK-NEXT: buffer_ct11.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer11.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&D_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct5 = allocation_ct5.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct5.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&E_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct6 = allocation_ct6.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct6.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAUQ_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&TAUP_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&workspace_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct9 = allocation_ct9.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct9.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct11 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct11 = allocation_ct11.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct11.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer11(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::gebrd(*cusolverH, m, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, buffer_ct7, buffer_ct8, buffer_ct9, Lwork, result_temp_buffer11);
    // CHECK-NEXT: buffer_ct11.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer11.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnCgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnCgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnCgebrd(*cusolverH, m, n, &A_c, lda, &D_f, &E_f, &TAUQ_c, &TAUP_c, &workspace_c, Lwork, &devInfo);
    cusolverDnCgebrd(*cusolverH, m, n, &A_c, lda, &D_f, &E_f, &TAUQ_c, &TAUP_c, &workspace_c, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::gebrd_get_lwork<std::complex<double>>((*cusolverH).get_device(), m, n, m, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::gebrd_get_lwork<std::complex<double>>((*cusolverH).get_device(), m, n, m, lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&D_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct5 = allocation_ct5.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct5.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&E_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct6 = allocation_ct6.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct6.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAUQ_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&TAUP_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&workspace_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct9 = allocation_ct9.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct9.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct11 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct11 = allocation_ct11.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct11.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer11(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::gebrd(*cusolverH, m, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, buffer_ct7, buffer_ct8, buffer_ct9, Lwork, result_temp_buffer11), 0);
    // CHECK-NEXT: buffer_ct11.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer11.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&D_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct5 = allocation_ct5.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct5.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&E_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct6 = allocation_ct6.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct6.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAUQ_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&TAUP_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&workspace_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct9 = allocation_ct9.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct9.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct11 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct11 = allocation_ct11.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct11.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer11(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::gebrd(*cusolverH, m, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, buffer_ct7, buffer_ct8, buffer_ct9, Lwork, result_temp_buffer11);
    // CHECK-NEXT: buffer_ct11.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer11.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnZgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnZgebrd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnZgebrd(*cusolverH, m, n, &A_z, lda, &D_d, &E_d, &TAUQ_z, &TAUP_z, &workspace_z, Lwork, &devInfo);
    cusolverDnZgebrd(*cusolverH, m, n, &A_z, lda, &D_d, &E_d, &TAUQ_z, &TAUP_z, &workspace_z, Lwork, &devInfo);


    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::orgbr_get_lwork<float>((*cusolverH).get_device(), (mkl::generate)side, m, n, k,  lda,  lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::orgbr_get_lwork<float>((*cusolverH).get_device(), (mkl::generate)side, m, n, k,  lda,  lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct5 = allocation_ct5.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct5.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAU_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct7.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&workspace_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct8 = allocation_ct8.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct8.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct10 = allocation_ct10.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct10.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer10(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::orgbr(*cusolverH, (mkl::generate)side, m, n, k, buffer_ct5, lda, buffer_ct7, buffer_ct8, Lwork, result_temp_buffer10), 0);
    // CHECK-NEXT: buffer_ct10.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer10.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct5 = allocation_ct5.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct5.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAU_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct7.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&workspace_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct8 = allocation_ct8.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct8.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct10 = allocation_ct10.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct10.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer10(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::orgbr(*cusolverH, (mkl::generate)side, m, n, k, buffer_ct5, lda, buffer_ct7, buffer_ct8, Lwork, result_temp_buffer10);
    // CHECK-NEXT: buffer_ct10.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer10.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnSorgbr_bufferSize(*cusolverH, side, m, n, k, &A_f, lda, &TAU_f, &Lwork);
    cusolverDnSorgbr_bufferSize(*cusolverH, side, m, n, k, &A_f, lda, &TAU_f, &Lwork);
    status = cusolverDnSorgbr(*cusolverH, side, m, n, k, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);
    cusolverDnSorgbr(*cusolverH, side, m, n, k, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);


    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::orgbr_get_lwork<double>((*cusolverH).get_device(), (mkl::generate)side, m, n, k,  lda,  lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::orgbr_get_lwork<double>((*cusolverH).get_device(), (mkl::generate)side, m, n, k,  lda,  lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct5 = allocation_ct5.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct5.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAU_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct7 = allocation_ct7.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct7.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&workspace_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct8 = allocation_ct8.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct8.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct10 = allocation_ct10.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct10.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer10(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::orgbr(*cusolverH, (mkl::generate)side, m, n, k, buffer_ct5, lda, buffer_ct7, buffer_ct8, Lwork, result_temp_buffer10), 0);
    // CHECK-NEXT: buffer_ct10.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer10.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct5 = allocation_ct5.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct5.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAU_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct7 = allocation_ct7.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct7.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&workspace_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct8 = allocation_ct8.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct8.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct10 = allocation_ct10.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct10.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer10(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::orgbr(*cusolverH, (mkl::generate)side, m, n, k, buffer_ct5, lda, buffer_ct7, buffer_ct8, Lwork, result_temp_buffer10);
    // CHECK-NEXT: buffer_ct10.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer10.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnDorgbr_bufferSize(*cusolverH, side, m, n, k, &A_d, lda, &TAU_d, &Lwork);
    cusolverDnDorgbr_bufferSize(*cusolverH, side, m, n, k, &A_d, lda, &TAU_d, &Lwork);
    status = cusolverDnDorgbr(*cusolverH, side, m, n, k, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);
    cusolverDnDorgbr(*cusolverH, side, m, n, k, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::ungbr_get_lwork<std::complex<float>>((*cusolverH).get_device(), (mkl::generate)side, m, n, k,  lda,  lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::ungbr_get_lwork<std::complex<float>>((*cusolverH).get_device(), (mkl::generate)side, m, n, k,  lda,  lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct5 = allocation_ct5.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct5.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAU_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&workspace_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct10 = allocation_ct10.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct10.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer10(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::ungbr(*cusolverH, (mkl::generate)side, m, n, k, buffer_ct5, lda, buffer_ct7, buffer_ct8, Lwork, result_temp_buffer10), 0);
    // CHECK-NEXT: buffer_ct10.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer10.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct5 = allocation_ct5.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct5.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAU_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&workspace_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct10 = allocation_ct10.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct10.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer10(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::ungbr(*cusolverH, (mkl::generate)side, m, n, k, buffer_ct5, lda, buffer_ct7, buffer_ct8, Lwork, result_temp_buffer10);
    // CHECK-NEXT: buffer_ct10.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer10.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnCungbr_bufferSize(*cusolverH, side, m, n, k, &A_c, lda, &TAU_c, &Lwork);
    cusolverDnCungbr_bufferSize(*cusolverH, side, m, n, k, &A_c, lda, &TAU_c, &Lwork);
    status = cusolverDnCungbr(*cusolverH, side, m, n, k, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);
    cusolverDnCungbr(*cusolverH, side, m, n, k, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::ungbr_get_lwork<std::complex<double>>((*cusolverH).get_device(), (mkl::generate)side, m, n, k,  lda,  lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::ungbr_get_lwork<std::complex<double>>((*cusolverH).get_device(), (mkl::generate)side, m, n, k,  lda,  lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct5 = allocation_ct5.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct5.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAU_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&workspace_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct10 = allocation_ct10.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct10.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer10(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::ungbr(*cusolverH, (mkl::generate)side, m, n, k, buffer_ct5, lda, buffer_ct7, buffer_ct8, Lwork, result_temp_buffer10), 0);
    // CHECK-NEXT: buffer_ct10.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer10.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct5 = allocation_ct5.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct5.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAU_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&workspace_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct10 = allocation_ct10.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct10.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer10(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::ungbr(*cusolverH, (mkl::generate)side, m, n, k, buffer_ct5, lda, buffer_ct7, buffer_ct8, Lwork, result_temp_buffer10);
    // CHECK-NEXT: buffer_ct10.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer10.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnZungbr_bufferSize(*cusolverH, side, m, n, k, &A_z, lda, &TAU_z, &Lwork);
    cusolverDnZungbr_bufferSize(*cusolverH, side, m, n, k, &A_z, lda, &TAU_z, &Lwork);
    status = cusolverDnZungbr(*cusolverH, side, m, n, k, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);
    cusolverDnZungbr(*cusolverH, side, m, n, k, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);


    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::sytrd_get_lwork<float>((*cusolverH).get_device(), uplo, n,  lda,    lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::sytrd_get_lwork<float>((*cusolverH).get_device(), uplo, n,  lda,    lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct3 = allocation_ct3.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct3.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&D_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct5 = allocation_ct5.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct5.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&E_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct6 = allocation_ct6.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct6.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAU_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct7.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&workspace_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct8 = allocation_ct8.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct8.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct10 = allocation_ct10.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct10.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer10(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::sytrd(*cusolverH, uplo, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, buffer_ct7, buffer_ct8, Lwork, result_temp_buffer10), 0);
    // CHECK-NEXT: buffer_ct10.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer10.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct3 = allocation_ct3.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct3.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&D_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct5 = allocation_ct5.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct5.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&E_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct6 = allocation_ct6.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct6.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAU_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct7.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&workspace_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct8 = allocation_ct8.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct8.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct10 = allocation_ct10.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct10.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer10(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::sytrd(*cusolverH, uplo, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, buffer_ct7, buffer_ct8, Lwork, result_temp_buffer10);
    // CHECK-NEXT: buffer_ct10.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer10.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnSsytrd_bufferSize(*cusolverH, uplo, n, &A_f, lda, &D_f, &E_f, &TAU_f, &Lwork);
    cusolverDnSsytrd_bufferSize(*cusolverH, uplo, n, &A_f, lda, &D_f, &E_f, &TAU_f, &Lwork);
    status = cusolverDnSsytrd(*cusolverH, uplo, n, &A_f, lda, &D_f, &E_f, &TAU_f, &workspace_f, Lwork, &devInfo);
    cusolverDnSsytrd(*cusolverH, uplo, n, &A_f, lda, &D_f, &E_f, &TAU_f, &workspace_f, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::sytrd_get_lwork<double>((*cusolverH).get_device(), uplo, n,  lda,    lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::sytrd_get_lwork<double>((*cusolverH).get_device(), uplo, n,  lda,    lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct3 = allocation_ct3.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct3.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&D_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct5 = allocation_ct5.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct5.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&E_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct6 = allocation_ct6.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct6.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAU_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct7 = allocation_ct7.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct7.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&workspace_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct8 = allocation_ct8.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct8.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct10 = allocation_ct10.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct10.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer10(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::sytrd(*cusolverH, uplo, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, buffer_ct7, buffer_ct8, Lwork, result_temp_buffer10), 0);
    // CHECK-NEXT: buffer_ct10.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer10.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct3 = allocation_ct3.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct3.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&D_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct5 = allocation_ct5.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct5.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&E_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct6 = allocation_ct6.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct6.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAU_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct7 = allocation_ct7.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct7.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&workspace_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct8 = allocation_ct8.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct8.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct10 = allocation_ct10.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct10.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer10(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::sytrd(*cusolverH, uplo, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, buffer_ct7, buffer_ct8, Lwork, result_temp_buffer10);
    // CHECK-NEXT: buffer_ct10.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer10.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnDsytrd_bufferSize(*cusolverH, uplo, n, &A_d, lda, &D_d, &E_d, &TAU_d, &Lwork);
    cusolverDnDsytrd_bufferSize(*cusolverH, uplo, n, &A_d, lda, &D_d, &E_d, &TAU_d, &Lwork);
    status = cusolverDnDsytrd(*cusolverH, uplo, n, &A_d, lda, &D_d, &E_d, &TAU_d, &workspace_d, Lwork, &devInfo);
    cusolverDnDsytrd(*cusolverH, uplo, n, &A_d, lda, &D_d, &E_d, &TAU_d, &workspace_d, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::hetrd_get_lwork<std::complex<float>>((*cusolverH).get_device(), uplo, n,  lda,    lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::hetrd_get_lwork<std::complex<float>>((*cusolverH).get_device(), uplo, n,  lda,    lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&D_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct5 = allocation_ct5.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct5.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&E_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct6 = allocation_ct6.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct6.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAU_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&workspace_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct10 = allocation_ct10.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct10.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer10(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::hetrd(*cusolverH, uplo, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, buffer_ct7, buffer_ct8, Lwork, result_temp_buffer10), 0);
    // CHECK-NEXT: buffer_ct10.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer10.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&D_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct5 = allocation_ct5.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct5.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&E_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct6 = allocation_ct6.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct6.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAU_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&workspace_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct10 = allocation_ct10.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct10.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer10(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::hetrd(*cusolverH, uplo, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, buffer_ct7, buffer_ct8, Lwork, result_temp_buffer10);
    // CHECK-NEXT: buffer_ct10.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer10.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnChetrd_bufferSize(*cusolverH, uplo, n, &A_c, lda, &D_f, &E_f, &TAU_c, &Lwork);
    cusolverDnChetrd_bufferSize(*cusolverH, uplo, n, &A_c, lda, &D_f, &E_f, &TAU_c, &Lwork);
    status = cusolverDnChetrd(*cusolverH, uplo, n, &A_c, lda, &D_f, &E_f, &TAU_c, &workspace_c, Lwork, &devInfo);
    cusolverDnChetrd(*cusolverH, uplo, n, &A_c, lda, &D_f, &E_f, &TAU_c, &workspace_c, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::hetrd_get_lwork<std::complex<double>>((*cusolverH).get_device(), uplo, n,  lda,    lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::hetrd_get_lwork<std::complex<double>>((*cusolverH).get_device(), uplo, n,  lda,    lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&D_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct5 = allocation_ct5.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct5.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&E_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct6 = allocation_ct6.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct6.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAU_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&workspace_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct10 = allocation_ct10.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct10.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer10(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::hetrd(*cusolverH, uplo, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, buffer_ct7, buffer_ct8, Lwork, result_temp_buffer10), 0);
    // CHECK-NEXT: buffer_ct10.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer10.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&D_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct5 = allocation_ct5.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct5.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&E_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct6 = allocation_ct6.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct6.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&TAU_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&workspace_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct10 = allocation_ct10.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct10.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer10(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::hetrd(*cusolverH, uplo, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, buffer_ct7, buffer_ct8, Lwork, result_temp_buffer10);
    // CHECK-NEXT: buffer_ct10.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer10.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnZhetrd_bufferSize(*cusolverH, uplo, n, &A_z, lda, &D_d, &E_d, &TAU_z, &Lwork);
    cusolverDnZhetrd_bufferSize(*cusolverH, uplo, n, &A_z, lda, &D_d, &E_d, &TAU_z, &Lwork);
    status = cusolverDnZhetrd(*cusolverH, uplo, n, &A_z, lda, &D_d, &E_d, &TAU_z, &workspace_z, Lwork, &devInfo);
    cusolverDnZhetrd(*cusolverH, uplo, n, &A_z, lda, &D_d, &E_d, &TAU_z, &workspace_z, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::ormtr_get_lwork<float>((*cusolverH).get_device(), side, uplo, trans, m, n,  lda,   ldb, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::ormtr_get_lwork<float>((*cusolverH).get_device(), side, uplo, trans, m, n,  lda,   ldb, lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct6 = allocation_ct6.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct6.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&TAU_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct8 = allocation_ct8.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct8.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&B_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct9 = allocation_ct9.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct9.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct11 = dpct::mem_mgr::instance().translate_ptr(&workspace_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct11 = allocation_ct11.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct11.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct13 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct13 = allocation_ct13.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct13.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer13(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::ormtr(*cusolverH, side, uplo, trans, m, n, buffer_ct6, lda, buffer_ct8, buffer_ct9, ldb, buffer_ct11, Lwork, result_temp_buffer13), 0);
    // CHECK-NEXT: buffer_ct13.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer13.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct6 = allocation_ct6.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct6.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&TAU_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct8 = allocation_ct8.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct8.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&B_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct9 = allocation_ct9.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct9.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct11 = dpct::mem_mgr::instance().translate_ptr(&workspace_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct11 = allocation_ct11.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct11.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct13 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct13 = allocation_ct13.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct13.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer13(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::ormtr(*cusolverH, side, uplo, trans, m, n, buffer_ct6, lda, buffer_ct8, buffer_ct9, ldb, buffer_ct11, Lwork, result_temp_buffer13);
    // CHECK-NEXT: buffer_ct13.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer13.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnSormtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_f, lda, &TAU_f, &B_f, ldb, &Lwork);
    cusolverDnSormtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_f, lda, &TAU_f, &B_f, ldb, &Lwork);
    status = cusolverDnSormtr(*cusolverH, side, uplo, trans, m, n, &A_f, lda, &TAU_f, &B_f, ldb, &workspace_f, Lwork, &devInfo);
    cusolverDnSormtr(*cusolverH, side, uplo, trans, m, n, &A_f, lda, &TAU_f, &B_f, ldb, &workspace_f, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::ormtr_get_lwork<double>((*cusolverH).get_device(), side, uplo, trans, m, n,  lda,   ldb, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::ormtr_get_lwork<double>((*cusolverH).get_device(), side, uplo, trans, m, n,  lda,   ldb, lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct6 = allocation_ct6.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct6.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&TAU_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct8 = allocation_ct8.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct8.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&B_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct9 = allocation_ct9.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct9.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct11 = dpct::mem_mgr::instance().translate_ptr(&workspace_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct11 = allocation_ct11.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct11.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct13 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct13 = allocation_ct13.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct13.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer13(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::ormtr(*cusolverH, side, uplo, trans, m, n, buffer_ct6, lda, buffer_ct8, buffer_ct9, ldb, buffer_ct11, Lwork, result_temp_buffer13), 0);
    // CHECK-NEXT: buffer_ct13.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer13.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct6 = allocation_ct6.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct6.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&TAU_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct8 = allocation_ct8.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct8.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&B_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct9 = allocation_ct9.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct9.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct11 = dpct::mem_mgr::instance().translate_ptr(&workspace_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct11 = allocation_ct11.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct11.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct13 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct13 = allocation_ct13.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct13.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer13(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::ormtr(*cusolverH, side, uplo, trans, m, n, buffer_ct6, lda, buffer_ct8, buffer_ct9, ldb, buffer_ct11, Lwork, result_temp_buffer13);
    // CHECK-NEXT: buffer_ct13.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer13.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnDormtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_d, lda, &TAU_d, &B_d, ldb, &Lwork);
    cusolverDnDormtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_d, lda, &TAU_d, &B_d, ldb, &Lwork);
    status = cusolverDnDormtr(*cusolverH, side, uplo, trans, m, n, &A_d, lda, &TAU_d, &B_d, ldb, &workspace_d, Lwork, &devInfo);
    cusolverDnDormtr(*cusolverH, side, uplo, trans, m, n, &A_d, lda, &TAU_d, &B_d, ldb, &workspace_d, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::unmtr_get_lwork<std::complex<float>>((*cusolverH).get_device(), side, uplo, trans, m, n,  lda,   ldb, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::unmtr_get_lwork<std::complex<float>>((*cusolverH).get_device(), side, uplo, trans, m, n,  lda,   ldb, lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&TAU_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&B_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct9 = allocation_ct9.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct9.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct11 = dpct::mem_mgr::instance().translate_ptr(&workspace_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct11 = allocation_ct11.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct11.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct13 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct13 = allocation_ct13.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct13.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer13(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::unmtr(*cusolverH, side, uplo, trans, m, n, buffer_ct6, lda, buffer_ct8, buffer_ct9, ldb, buffer_ct11, Lwork, result_temp_buffer13), 0);
    // CHECK-NEXT: buffer_ct13.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer13.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&TAU_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&B_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct9 = allocation_ct9.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct9.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct11 = dpct::mem_mgr::instance().translate_ptr(&workspace_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct11 = allocation_ct11.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct11.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct13 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct13 = allocation_ct13.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct13.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer13(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::unmtr(*cusolverH, side, uplo, trans, m, n, buffer_ct6, lda, buffer_ct8, buffer_ct9, ldb, buffer_ct11, Lwork, result_temp_buffer13);
    // CHECK-NEXT: buffer_ct13.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer13.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnCunmtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_c, lda, &TAU_c, &B_c, ldb, &Lwork);
    cusolverDnCunmtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_c, lda, &TAU_c, &B_c, ldb, &Lwork);
    status = cusolverDnCunmtr(*cusolverH, side, uplo, trans, m, n, &A_c, lda, &TAU_c, &B_c, ldb, &workspace_c, Lwork, &devInfo);
    cusolverDnCunmtr(*cusolverH, side, uplo, trans, m, n, &A_c, lda, &TAU_c, &B_c, ldb, &workspace_c, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::unmtr_get_lwork<std::complex<double>>((*cusolverH).get_device(), side, uplo, trans, m, n,  lda,   ldb, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::unmtr_get_lwork<std::complex<double>>((*cusolverH).get_device(), side, uplo, trans, m, n,  lda,   ldb, lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&TAU_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&B_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct9 = allocation_ct9.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct9.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct11 = dpct::mem_mgr::instance().translate_ptr(&workspace_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct11 = allocation_ct11.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct11.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct13 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct13 = allocation_ct13.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct13.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer13(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::unmtr(*cusolverH, side, uplo, trans, m, n, buffer_ct6, lda, buffer_ct8, buffer_ct9, ldb, buffer_ct11, Lwork, result_temp_buffer13), 0);
    // CHECK-NEXT: buffer_ct13.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer13.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&TAU_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&B_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct9 = allocation_ct9.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct9.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct11 = dpct::mem_mgr::instance().translate_ptr(&workspace_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct11 = allocation_ct11.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct11.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct13 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct13 = allocation_ct13.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct13.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer13(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::unmtr(*cusolverH, side, uplo, trans, m, n, buffer_ct6, lda, buffer_ct8, buffer_ct9, ldb, buffer_ct11, Lwork, result_temp_buffer13);
    // CHECK-NEXT: buffer_ct13.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer13.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnZunmtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_z, lda, &TAU_z, &B_z, ldb, &Lwork);
    cusolverDnZunmtr_bufferSize(*cusolverH, side, uplo, trans, m, n, &A_z, lda, &TAU_z, &B_z, ldb, &Lwork);
    status = cusolverDnZunmtr(*cusolverH, side, uplo, trans, m, n, &A_z, lda, &TAU_z, &B_z, ldb, &workspace_z, Lwork, &devInfo);
    cusolverDnZunmtr(*cusolverH, side, uplo, trans, m, n, &A_z, lda, &TAU_z, &B_z, ldb, &workspace_z, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::orgtr_get_lwork<float>((*cusolverH).get_device(), uplo, n,  lda,  lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::orgtr_get_lwork<float>((*cusolverH).get_device(), uplo, n,  lda,  lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct3 = allocation_ct3.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct3.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&TAU_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct5 = allocation_ct5.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct5.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&workspace_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct6 = allocation_ct6.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct6.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::orgtr(*cusolverH, uplo, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, Lwork, result_temp_buffer8), 0);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct3 = allocation_ct3.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct3.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&TAU_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct5 = allocation_ct5.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct5.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&workspace_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct6 = allocation_ct6.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct6.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::orgtr(*cusolverH, uplo, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, Lwork, result_temp_buffer8);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnSorgtr_bufferSize(*cusolverH, uplo, n, &A_f, lda, &TAU_f, &Lwork);
    cusolverDnSorgtr_bufferSize(*cusolverH, uplo, n, &A_f, lda, &TAU_f, &Lwork);
    status = cusolverDnSorgtr(*cusolverH, uplo, n, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);
    cusolverDnSorgtr(*cusolverH, uplo, n, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::orgtr_get_lwork<double>((*cusolverH).get_device(), uplo, n,  lda,  lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::orgtr_get_lwork<double>((*cusolverH).get_device(), uplo, n,  lda,  lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct3 = allocation_ct3.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct3.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&TAU_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct5 = allocation_ct5.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct5.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&workspace_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct6 = allocation_ct6.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct6.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::orgtr(*cusolverH, uplo, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, Lwork, result_temp_buffer8), 0);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct3 = allocation_ct3.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct3.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&TAU_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct5 = allocation_ct5.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct5.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&workspace_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct6 = allocation_ct6.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct6.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::orgtr(*cusolverH, uplo, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, Lwork, result_temp_buffer8);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnDorgtr_bufferSize(*cusolverH, uplo, n, &A_d, lda, &TAU_d, &Lwork);
    cusolverDnDorgtr_bufferSize(*cusolverH, uplo, n, &A_d, lda, &TAU_d, &Lwork);
    status = cusolverDnDorgtr(*cusolverH, uplo, n, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);
    cusolverDnDorgtr(*cusolverH, uplo, n, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::ungtr_get_lwork<std::complex<float>>((*cusolverH).get_device(), uplo, n,  lda,  lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::ungtr_get_lwork<std::complex<float>>((*cusolverH).get_device(), uplo, n,  lda,  lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&TAU_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct5 = allocation_ct5.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct5.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&workspace_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::ungtr(*cusolverH, uplo, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, Lwork, result_temp_buffer8), 0);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&TAU_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct5 = allocation_ct5.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct5.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&workspace_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::ungtr(*cusolverH, uplo, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, Lwork, result_temp_buffer8);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnCungtr_bufferSize(*cusolverH, uplo, n, &A_c, lda, &TAU_c, &Lwork);
    cusolverDnCungtr_bufferSize(*cusolverH, uplo, n, &A_c, lda, &TAU_c, &Lwork);
    status = cusolverDnCungtr(*cusolverH, uplo, n, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);
    cusolverDnCungtr(*cusolverH, uplo, n, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::ungtr_get_lwork<std::complex<double>>((*cusolverH).get_device(), uplo, n,  lda,  lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::ungtr_get_lwork<std::complex<double>>((*cusolverH).get_device(), uplo, n,  lda,  lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&TAU_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct5 = allocation_ct5.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct5.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&workspace_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::ungtr(*cusolverH, uplo, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, Lwork, result_temp_buffer8), 0);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&TAU_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct5 = allocation_ct5.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct5.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&workspace_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::ungtr(*cusolverH, uplo, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, Lwork, result_temp_buffer8);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnZungtr_bufferSize(*cusolverH, uplo, n, &A_z, lda, &TAU_z, &Lwork);
    cusolverDnZungtr_bufferSize(*cusolverH, uplo, n, &A_z, lda, &TAU_z, &Lwork);
    status = cusolverDnZungtr(*cusolverH, uplo, n, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);
    cusolverDnZungtr(*cusolverH, uplo, n, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);


    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::job job_ct_mkl_jobu;
    // CHECK-NEXT: mkl::job job_ct_mkl_jobvt;
    // CHECK-NEXT: status = (mkl::lapack::gesvd_get_lwork<float>((*cusolverH).get_device(), job_ct_mkl_jobu, job_ct_mkl_jobvt, m, n, m, m, n, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::job job_ct_mkl_jobu;
    // CHECK-NEXT: mkl::job job_ct_mkl_jobvt;
    // CHECK-NEXT: mkl::lapack::gesvd_get_lwork<float>((*cusolverH).get_device(), job_ct_mkl_jobu, job_ct_mkl_jobvt, m, n, m, m, n, lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct5 = allocation_ct5.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct5.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&S_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct7.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&U_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct8 = allocation_ct8.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct8.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&VT_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct10 = allocation_ct10.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct10.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct12 = dpct::mem_mgr::instance().translate_ptr(&workspace_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct12 = allocation_ct12.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct12.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct15 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct15 = allocation_ct15.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct15.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer15(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::gesvd (*cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, buffer_ct5, lda, buffer_ct7, buffer_ct8, ldu, buffer_ct10, ldvt, buffer_ct12, Lwork,  result_temp_buffer15), 0);
    // CHECK-NEXT: buffer_ct15.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer15.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct5 = allocation_ct5.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct5.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&S_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct7.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&U_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct8 = allocation_ct8.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct8.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&VT_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct10 = allocation_ct10.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct10.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct12 = dpct::mem_mgr::instance().translate_ptr(&workspace_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct12 = allocation_ct12.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct12.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct15 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct15 = allocation_ct15.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct15.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer15(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::gesvd (*cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, buffer_ct5, lda, buffer_ct7, buffer_ct8, ldu, buffer_ct10, ldvt, buffer_ct12, Lwork,  result_temp_buffer15);
    // CHECK-NEXT: buffer_ct15.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer15.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnSgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnSgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnSgesvd (*cusolverH, jobu, jobvt, m, n, &A_f, lda, &S_f, &U_f, ldu, &VT_f, ldvt, &workspace_f, Lwork, &Rwork_f, &devInfo);
    cusolverDnSgesvd (*cusolverH, jobu, jobvt, m, n, &A_f, lda, &S_f, &U_f, ldu, &VT_f, ldvt, &workspace_f, Lwork, &Rwork_f, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::job job_ct_mkl_jobu;
    // CHECK-NEXT: mkl::job job_ct_mkl_jobvt;
    // CHECK-NEXT: status = (mkl::lapack::gesvd_get_lwork<double>((*cusolverH).get_device(), job_ct_mkl_jobu, job_ct_mkl_jobvt, m, n, m, m, n, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::job job_ct_mkl_jobu;
    // CHECK-NEXT: mkl::job job_ct_mkl_jobvt;
    // CHECK-NEXT: mkl::lapack::gesvd_get_lwork<double>((*cusolverH).get_device(), job_ct_mkl_jobu, job_ct_mkl_jobvt, m, n, m, m, n, lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct5 = allocation_ct5.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct5.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&S_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct7 = allocation_ct7.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct7.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&U_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct8 = allocation_ct8.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct8.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&VT_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct10 = allocation_ct10.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct10.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct12 = dpct::mem_mgr::instance().translate_ptr(&workspace_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct12 = allocation_ct12.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct12.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct15 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct15 = allocation_ct15.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct15.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer15(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::gesvd (*cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, buffer_ct5, lda, buffer_ct7, buffer_ct8, ldu, buffer_ct10, ldvt, buffer_ct12, Lwork,  result_temp_buffer15), 0);
    // CHECK-NEXT: buffer_ct15.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer15.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct5 = allocation_ct5.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct5.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&S_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct7 = allocation_ct7.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct7.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&U_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct8 = allocation_ct8.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct8.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&VT_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct10 = allocation_ct10.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct10.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct12 = dpct::mem_mgr::instance().translate_ptr(&workspace_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct12 = allocation_ct12.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct12.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct15 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct15 = allocation_ct15.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct15.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer15(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::gesvd (*cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, buffer_ct5, lda, buffer_ct7, buffer_ct8, ldu, buffer_ct10, ldvt, buffer_ct12, Lwork,  result_temp_buffer15);
    // CHECK-NEXT: buffer_ct15.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer15.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnDgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnDgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnDgesvd (*cusolverH, jobu, jobvt, m, n, &A_d, lda, &S_d, &U_d, ldu, &VT_d, ldvt, &workspace_d, Lwork, &Rwork_d, &devInfo);
    cusolverDnDgesvd (*cusolverH, jobu, jobvt, m, n, &A_d, lda, &S_d, &U_d, ldu, &VT_d, ldvt, &workspace_d, Lwork, &Rwork_d, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::job job_ct_mkl_jobu;
    // CHECK-NEXT: mkl::job job_ct_mkl_jobvt;
    // CHECK-NEXT: status = (mkl::lapack::gesvd_get_lwork<std::complex<float>>((*cusolverH).get_device(), job_ct_mkl_jobu, job_ct_mkl_jobvt, m, n, m, m, n, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::job job_ct_mkl_jobu;
    // CHECK-NEXT: mkl::job job_ct_mkl_jobvt;
    // CHECK-NEXT: mkl::lapack::gesvd_get_lwork<std::complex<float>>((*cusolverH).get_device(), job_ct_mkl_jobu, job_ct_mkl_jobvt, m, n, m, m, n, lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct5 = allocation_ct5.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct5.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&S_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct7.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&U_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&VT_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct10 = allocation_ct10.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct10.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct12 = dpct::mem_mgr::instance().translate_ptr(&workspace_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct12 = allocation_ct12.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct12.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct14 = dpct::mem_mgr::instance().translate_ptr(&Rwork_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct14 = allocation_ct14.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct14.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct15 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct15 = allocation_ct15.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct15.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer15(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::gesvd (*cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, buffer_ct5, lda, buffer_ct7, buffer_ct8, ldu, buffer_ct10, ldvt, buffer_ct12, Lwork, buffer_ct14, result_temp_buffer15), 0);
    // CHECK-NEXT: buffer_ct15.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer15.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct5 = allocation_ct5.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct5.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&S_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct7.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&U_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&VT_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct10 = allocation_ct10.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct10.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct12 = dpct::mem_mgr::instance().translate_ptr(&workspace_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct12 = allocation_ct12.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct12.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct14 = dpct::mem_mgr::instance().translate_ptr(&Rwork_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct14 = allocation_ct14.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct14.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct15 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct15 = allocation_ct15.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct15.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer15(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::gesvd (*cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, buffer_ct5, lda, buffer_ct7, buffer_ct8, ldu, buffer_ct10, ldvt, buffer_ct12, Lwork, buffer_ct14, result_temp_buffer15);
    // CHECK-NEXT: buffer_ct15.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer15.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnCgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnCgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnCgesvd (*cusolverH, jobu, jobvt, m, n, &A_c, lda, &S_f, &U_c, ldu, &VT_c, ldvt, &workspace_c, Lwork, &Rwork_f, &devInfo);
    cusolverDnCgesvd (*cusolverH, jobu, jobvt, m, n, &A_c, lda, &S_f, &U_c, ldu, &VT_c, ldvt, &workspace_c, Lwork, &Rwork_f, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::job job_ct_mkl_jobu;
    // CHECK-NEXT: mkl::job job_ct_mkl_jobvt;
    // CHECK-NEXT: status = (mkl::lapack::gesvd_get_lwork<std::complex<double>>((*cusolverH).get_device(), job_ct_mkl_jobu, job_ct_mkl_jobvt, m, n, m, m, n, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::job job_ct_mkl_jobu;
    // CHECK-NEXT: mkl::job job_ct_mkl_jobvt;
    // CHECK-NEXT: mkl::lapack::gesvd_get_lwork<std::complex<double>>((*cusolverH).get_device(), job_ct_mkl_jobu, job_ct_mkl_jobvt, m, n, m, m, n, lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct5 = allocation_ct5.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct5.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&S_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct7 = allocation_ct7.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct7.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&U_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&VT_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct10 = allocation_ct10.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct10.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct12 = dpct::mem_mgr::instance().translate_ptr(&workspace_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct12 = allocation_ct12.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct12.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct14 = dpct::mem_mgr::instance().translate_ptr(&Rwork_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct14 = allocation_ct14.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct14.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct15 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct15 = allocation_ct15.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct15.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer15(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::gesvd (*cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, buffer_ct5, lda, buffer_ct7, buffer_ct8, ldu, buffer_ct10, ldvt, buffer_ct12, Lwork, buffer_ct14, result_temp_buffer15), 0);
    // CHECK-NEXT: buffer_ct15.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer15.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct5 = allocation_ct5.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct5.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&S_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct7 = allocation_ct7.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct7.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&U_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct8 = allocation_ct8.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct8.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct10 = dpct::mem_mgr::instance().translate_ptr(&VT_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct10 = allocation_ct10.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct10.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct12 = dpct::mem_mgr::instance().translate_ptr(&workspace_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct12 = allocation_ct12.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct12.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct14 = dpct::mem_mgr::instance().translate_ptr(&Rwork_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct14 = allocation_ct14.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct14.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct15 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct15 = allocation_ct15.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct15.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer15(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::gesvd (*cusolverH, (mkl::job)jobu, (mkl::job)jobvt, m, n, buffer_ct5, lda, buffer_ct7, buffer_ct8, ldu, buffer_ct10, ldvt, buffer_ct12, Lwork, buffer_ct14, result_temp_buffer15);
    // CHECK-NEXT: buffer_ct15.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer15.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnZgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    cusolverDnZgesvd_bufferSize(*cusolverH, m, n, &Lwork);
    status = cusolverDnZgesvd (*cusolverH, jobu, jobvt, m, n, &A_z, lda, &S_d, &U_z, ldu, &VT_z, ldvt, &workspace_z, Lwork, &Rwork_d, &devInfo);
    cusolverDnZgesvd (*cusolverH, jobu, jobvt, m, n, &A_z, lda, &S_d, &U_z, ldu, &VT_z, ldvt, &workspace_z, Lwork, &Rwork_d, &devInfo);

}
