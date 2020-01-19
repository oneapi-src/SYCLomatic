// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cusolverDnLn.dp.cpp --match-full-lines %s
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

    //CHECK: /*
    //CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cusolverDnSpotrf_bufferSize was replaced with 0, because Function call is redundant in DPC++.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = 0;
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cusolverDnDpotrf_bufferSize was replaced with 0, because Function call is redundant in DPC++.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = 0;
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cusolverDnCpotrf_bufferSize was replaced with 0, because Function call is redundant in DPC++.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = 0;
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cusolverDnZpotrf_bufferSize was replaced with 0, because Function call is redundant in DPC++.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = 0;
    status = cusolverDnSpotrf_bufferSize(*cusolverH, uplo, n, &A_f, lda, &Lwork);
    status = cusolverDnDpotrf_bufferSize(*cusolverH, uplo, n, &A_d, lda, &Lwork);
    status = cusolverDnCpotrf_bufferSize(*cusolverH, uplo, n, &A_c, lda, &Lwork);
    status = cusolverDnZpotrf_bufferSize(*cusolverH, uplo, n, &A_z, lda, &Lwork);

    // CHECK: /*
    // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cusolverDnSgetrf_bufferSize was replaced with 0, because Function call is redundant in DPC++.
    // CHECK-NEXT: */
    // CHECK-NEXT: status = 0;
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cusolverDnDgetrf_bufferSize was replaced with 0, because Function call is redundant in DPC++.
    // CHECK-NEXT: */
    // CHECK-NEXT: status = 0;
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cusolverDnCgetrf_bufferSize was replaced with 0, because Function call is redundant in DPC++.
    // CHECK-NEXT: */
    // CHECK-NEXT: status = 0;
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cusolverDnZgetrf_bufferSize was replaced with 0, because Function call is redundant in DPC++.
    // CHECK-NEXT: */
    // CHECK-NEXT: status = 0;
    status = cusolverDnSgetrf_bufferSize(*cusolverH, m, n, &A_f, lda, &Lwork);
    status = cusolverDnDgetrf_bufferSize(*cusolverH, m, n, &A_d, lda, &Lwork);
    status = cusolverDnCgetrf_bufferSize(*cusolverH, m, n, &A_c, lda, &Lwork);
    status = cusolverDnZgetrf_bufferSize(*cusolverH, m, n, &A_z, lda, &Lwork);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct3 = allocation_ct3.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct3.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct7 = allocation_ct7.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::potrf(*cusolverH, uplo, n, buffer_ct3, lda,   result_temp_buffer7), 0);
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct3 = allocation_ct3.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct3.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct7 = allocation_ct7.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::potrf(*cusolverH, uplo, n, buffer_ct3, lda,   result_temp_buffer7);
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnSpotrf(*cusolverH, uplo, n, &A_f, lda, &workspace_f, Lwork, &devInfo);
    cusolverDnSpotrf(*cusolverH, uplo, n, &A_f, lda, &workspace_f, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct3 = allocation_ct3.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct3.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct7 = allocation_ct7.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::potrf(*cusolverH, uplo, n, buffer_ct3, lda,   result_temp_buffer7), 0);
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct3 = allocation_ct3.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct3.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct7 = allocation_ct7.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::potrf(*cusolverH, uplo, n, buffer_ct3, lda,   result_temp_buffer7);
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnDpotrf(*cusolverH, uplo, n, &A_d, lda, &workspace_d, Lwork, &devInfo);
    cusolverDnDpotrf(*cusolverH, uplo, n, &A_d, lda, &workspace_d, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct7 = allocation_ct7.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::potrf(*cusolverH, uplo, n, buffer_ct3, lda,   result_temp_buffer7), 0);
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct7 = allocation_ct7.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::potrf(*cusolverH, uplo, n, buffer_ct3, lda,   result_temp_buffer7);
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnCpotrf(*cusolverH, uplo, n, &A_c, lda, &workspace_c, Lwork, &devInfo);
    cusolverDnCpotrf(*cusolverH, uplo, n, &A_c, lda, &workspace_c, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct7 = allocation_ct7.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::potrf(*cusolverH, uplo, n, buffer_ct3, lda,   result_temp_buffer7), 0);
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct7 = allocation_ct7.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::potrf(*cusolverH, uplo, n, buffer_ct3, lda,   result_temp_buffer7);
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnZpotrf(*cusolverH, uplo, n, &A_z, lda, &workspace_z, Lwork, &devInfo);
    cusolverDnZpotrf(*cusolverH, uplo, n, &A_z, lda, &workspace_z, Lwork, &devInfo);


    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&C_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct4 = allocation_ct4.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct4.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&B_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct6 = allocation_ct6.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct6.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::potrs(*cusolverH, uplo, n, nrhs, buffer_ct4, lda, buffer_ct6, ldb, result_temp_buffer8), 0);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&C_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct4 = allocation_ct4.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct4.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&B_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct6 = allocation_ct6.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct6.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::potrs(*cusolverH, uplo, n, nrhs, buffer_ct4, lda, buffer_ct6, ldb, result_temp_buffer8);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnSpotrs(*cusolverH, uplo, n, nrhs, &C_f, lda, &B_f, ldb, &devInfo);
    cusolverDnSpotrs(*cusolverH, uplo, n, nrhs, &C_f, lda, &B_f, ldb, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&C_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct4 = allocation_ct4.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct4.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&B_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct6 = allocation_ct6.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct6.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::potrs(*cusolverH, uplo, n, nrhs, buffer_ct4, lda, buffer_ct6, ldb, result_temp_buffer8), 0);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&C_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct4 = allocation_ct4.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct4.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&B_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct6 = allocation_ct6.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct6.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::potrs(*cusolverH, uplo, n, nrhs, buffer_ct4, lda, buffer_ct6, ldb, result_temp_buffer8);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnDpotrs(*cusolverH, uplo, n, nrhs, &C_d, lda, &B_d, ldb, &devInfo);
    cusolverDnDpotrs(*cusolverH, uplo, n, nrhs, &C_d, lda, &B_d, ldb, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&C_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct4 = allocation_ct4.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct4.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&B_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::potrs(*cusolverH, uplo, n, nrhs, buffer_ct4, lda, buffer_ct6, ldb, result_temp_buffer8), 0);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&C_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct4 = allocation_ct4.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct4.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&B_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::potrs(*cusolverH, uplo, n, nrhs, buffer_ct4, lda, buffer_ct6, ldb, result_temp_buffer8);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnCpotrs(*cusolverH, uplo, n, nrhs, &C_c, lda, &B_c, ldb, &devInfo);
    cusolverDnCpotrs(*cusolverH, uplo, n, nrhs, &C_c, lda, &B_c, ldb, &devInfo);


    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&C_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct4 = allocation_ct4.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct4.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&B_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::potrs(*cusolverH, uplo, n, nrhs, buffer_ct4, lda, buffer_ct6, ldb, result_temp_buffer8), 0);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&C_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct4 = allocation_ct4.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct4.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&B_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::potrs(*cusolverH, uplo, n, nrhs, buffer_ct4, lda, buffer_ct6, ldb, result_temp_buffer8);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnZpotrs(*cusolverH, uplo, n, nrhs, &C_z, lda, &B_z, ldb, &devInfo);
    cusolverDnZpotrs(*cusolverH, uplo, n, nrhs, &C_z, lda, &B_z, ldb, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct3 = allocation_ct3.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct3.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct6 = allocation_ct6.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct6.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer6(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct7 = allocation_ct7.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::getrf(*cusolverH, m, n, buffer_ct3, lda,  result_temp_buffer6, result_temp_buffer7), 0);
    // CHECK-NEXT: buffer_ct6.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer6.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct3 = allocation_ct3.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct3.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct6 = allocation_ct6.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct6.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer6(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct7 = allocation_ct7.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::getrf(*cusolverH, m, n, buffer_ct3, lda,  result_temp_buffer6, result_temp_buffer7);
    // CHECK-NEXT: buffer_ct6.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer6.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnSgetrf(*cusolverH, m, n, &A_f, lda, &workspace_f, &devIpiv, &devInfo);
    cusolverDnSgetrf(*cusolverH, m, n, &A_f, lda, &workspace_f, &devIpiv, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct3 = allocation_ct3.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct3.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct6 = allocation_ct6.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct6.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer6(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct7 = allocation_ct7.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::getrf(*cusolverH, m, n, buffer_ct3, lda,  result_temp_buffer6, result_temp_buffer7), 0);
    // CHECK-NEXT: buffer_ct6.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer6.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct3 = allocation_ct3.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct3.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct6 = allocation_ct6.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct6.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer6(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct7 = allocation_ct7.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::getrf(*cusolverH, m, n, buffer_ct3, lda,  result_temp_buffer6, result_temp_buffer7);
    // CHECK-NEXT: buffer_ct6.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer6.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnDgetrf(*cusolverH, m, n, &A_d, lda, &workspace_d, &devIpiv, &devInfo);
    cusolverDnDgetrf(*cusolverH, m, n, &A_d, lda, &workspace_d, &devIpiv, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct6 = allocation_ct6.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct6.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer6(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct7 = allocation_ct7.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::getrf(*cusolverH, m, n, buffer_ct3, lda,  result_temp_buffer6, result_temp_buffer7), 0);
    // CHECK-NEXT: buffer_ct6.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer6.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct6 = allocation_ct6.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct6.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer6(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct7 = allocation_ct7.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::getrf(*cusolverH, m, n, buffer_ct3, lda,  result_temp_buffer6, result_temp_buffer7);
    // CHECK-NEXT: buffer_ct6.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer6.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnCgetrf(*cusolverH, m, n, &A_c, lda, &workspace_c, &devIpiv, &devInfo);
    cusolverDnCgetrf(*cusolverH, m, n, &A_c, lda, &workspace_c, &devIpiv, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct6 = allocation_ct6.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct6.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer6(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct7 = allocation_ct7.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::getrf(*cusolverH, m, n, buffer_ct3, lda,  result_temp_buffer6, result_temp_buffer7), 0);
    // CHECK-NEXT: buffer_ct6.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer6.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct6 = allocation_ct6.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct6.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer6(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct7 = allocation_ct7.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::getrf(*cusolverH, m, n, buffer_ct3, lda,  result_temp_buffer6, result_temp_buffer7);
    // CHECK-NEXT: buffer_ct6.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer6.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnZgetrf(*cusolverH, m, n, &A_z, lda, &workspace_z, &devIpiv, &devInfo);
    cusolverDnZgetrf(*cusolverH, m, n, &A_z, lda, &workspace_z, &devIpiv, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct6 = allocation_ct6.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct6.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer6(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct7 = allocation_ct7.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::getrf(*cusolverH, m, n, buffer_ct3, lda,  result_temp_buffer6, result_temp_buffer7), 0);
    // CHECK-NEXT: buffer_ct6.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer6.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct6 = allocation_ct6.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct6.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer6(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct7 = allocation_ct7.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::getrf(*cusolverH, m, n, buffer_ct3, lda,  result_temp_buffer6, result_temp_buffer7);
    // CHECK-NEXT: buffer_ct6.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer6.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnZgetrf(*cusolverH, m, n, &A_z, lda, &workspace_z, &devIpiv, &devInfo);
    cusolverDnZgetrf(*cusolverH, m, n, &A_z, lda, &workspace_z, &devIpiv, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct4 = allocation_ct4.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct4.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct6 = allocation_ct6.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct6.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer6(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&B_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct7.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct9 = allocation_ct9.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct9.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer9(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::getrs(*cusolverH, trans, n, nrhs, buffer_ct4, lda, result_temp_buffer6, buffer_ct7, ldb, result_temp_buffer9), 0);
    // CHECK-NEXT: buffer_ct6.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer6.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct9.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer9.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct4 = allocation_ct4.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct4.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct6 = allocation_ct6.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct6.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer6(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&B_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct7.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct9 = allocation_ct9.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct9.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer9(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::getrs(*cusolverH, trans, n, nrhs, buffer_ct4, lda, result_temp_buffer6, buffer_ct7, ldb, result_temp_buffer9);
    // CHECK-NEXT: buffer_ct6.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer6.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct9.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer9.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnSgetrs(*cusolverH, trans, n, nrhs, &A_f, lda, &devIpiv, &B_f, ldb, &devInfo);
    cusolverDnSgetrs(*cusolverH, trans, n, nrhs, &A_f, lda, &devIpiv, &B_f, ldb, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct4 = allocation_ct4.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct4.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct6 = allocation_ct6.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct6.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer6(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&B_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct7 = allocation_ct7.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct7.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct9 = allocation_ct9.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct9.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer9(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::getrs(*cusolverH, trans, n, nrhs, buffer_ct4, lda, result_temp_buffer6, buffer_ct7, ldb, result_temp_buffer9), 0);
    // CHECK-NEXT: buffer_ct6.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer6.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct9.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer9.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct4 = allocation_ct4.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct4.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct6 = allocation_ct6.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct6.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer6(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&B_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct7 = allocation_ct7.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct7.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct9 = allocation_ct9.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct9.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer9(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::getrs(*cusolverH, trans, n, nrhs, buffer_ct4, lda, result_temp_buffer6, buffer_ct7, ldb, result_temp_buffer9);
    // CHECK-NEXT: buffer_ct6.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer6.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct9.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer9.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnDgetrs(*cusolverH, trans, n, nrhs, &A_d, lda, &devIpiv, &B_d, ldb, &devInfo);
    cusolverDnDgetrs(*cusolverH, trans, n, nrhs, &A_d, lda, &devIpiv, &B_d, ldb, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct4 = allocation_ct4.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct4.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct6 = allocation_ct6.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct6.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer6(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&B_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct9 = allocation_ct9.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct9.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer9(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::getrs(*cusolverH, trans, n, nrhs, buffer_ct4, lda, result_temp_buffer6, buffer_ct7, ldb, result_temp_buffer9), 0);
    // CHECK-NEXT: buffer_ct6.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer6.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct9.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer9.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct4 = allocation_ct4.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct4.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct6 = allocation_ct6.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct6.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer6(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&B_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct9 = allocation_ct9.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct9.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer9(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::getrs(*cusolverH, trans, n, nrhs, buffer_ct4, lda, result_temp_buffer6, buffer_ct7, ldb, result_temp_buffer9);
    // CHECK-NEXT: buffer_ct6.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer6.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct9.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer9.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnCgetrs(*cusolverH, trans, n, nrhs, &A_c, lda, &devIpiv, &B_c, ldb, &devInfo);
    cusolverDnCgetrs(*cusolverH, trans, n, nrhs, &A_c, lda, &devIpiv, &B_c, ldb, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct4 = allocation_ct4.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct4.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct6 = allocation_ct6.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct6.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer6(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&B_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct9 = allocation_ct9.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct9.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer9(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::getrs(*cusolverH, trans, n, nrhs, buffer_ct4, lda, result_temp_buffer6, buffer_ct7, ldb, result_temp_buffer9), 0);
    // CHECK-NEXT: buffer_ct6.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer6.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct9.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer9.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct4 = allocation_ct4.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct4.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct6 = allocation_ct6.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct6.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer6(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&B_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct9 = allocation_ct9.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct9.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer9(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::getrs(*cusolverH, trans, n, nrhs, buffer_ct4, lda, result_temp_buffer6, buffer_ct7, ldb, result_temp_buffer9);
    // CHECK-NEXT: buffer_ct6.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer6.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct9.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer9.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnZgetrs(*cusolverH, trans, n, nrhs, &A_z, lda, &devIpiv, &B_z, ldb, &devInfo);
    cusolverDnZgetrs(*cusolverH, trans, n, nrhs, &A_z, lda, &devIpiv, &B_z, ldb, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::geqrf_get_lwork<float>((*cusolverH).get_device(), m, n,  lda, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::geqrf_get_lwork<float>((*cusolverH).get_device(), m, n,  lda, lwork64);
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
    // CHECK-NEXT: status = (mkl::lapack::geqrf(*cusolverH, m, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, Lwork, result_temp_buffer8), 0);
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
    // CHECK-NEXT: mkl::lapack::geqrf(*cusolverH, m, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, Lwork, result_temp_buffer8);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnSgeqrf_bufferSize(*cusolverH, m, n, &A_f, lda, &Lwork);
    cusolverDnSgeqrf_bufferSize(*cusolverH, m, n, &A_f, lda, &Lwork);
    status = cusolverDnSgeqrf(*cusolverH, m, n, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);
    cusolverDnSgeqrf(*cusolverH, m, n, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::geqrf_get_lwork<double>((*cusolverH).get_device(), m, n,  lda, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::geqrf_get_lwork<double>((*cusolverH).get_device(), m, n,  lda, lwork64);
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
    // CHECK-NEXT: status = (mkl::lapack::geqrf(*cusolverH, m, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, Lwork, result_temp_buffer8), 0);
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
    // CHECK-NEXT: mkl::lapack::geqrf(*cusolverH, m, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, Lwork, result_temp_buffer8);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnDgeqrf_bufferSize(*cusolverH, m, n, &A_d, lda, &Lwork);
    cusolverDnDgeqrf_bufferSize(*cusolverH, m, n, &A_d, lda, &Lwork);
    status = cusolverDnDgeqrf(*cusolverH, m, n, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);
    cusolverDnDgeqrf(*cusolverH, m, n, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::geqrf_get_lwork<std::complex<float>>((*cusolverH).get_device(), m, n,  lda, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::geqrf_get_lwork<std::complex<float>>((*cusolverH).get_device(), m, n,  lda, lwork64);
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
    // CHECK-NEXT: status = (mkl::lapack::geqrf(*cusolverH, m, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, Lwork, result_temp_buffer8), 0);
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
    // CHECK-NEXT: mkl::lapack::geqrf(*cusolverH, m, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, Lwork, result_temp_buffer8);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnCgeqrf_bufferSize(*cusolverH, m, n, &A_c, lda, &Lwork);
    cusolverDnCgeqrf_bufferSize(*cusolverH, m, n, &A_c, lda, &Lwork);
    status = cusolverDnCgeqrf(*cusolverH, m, n, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);
    cusolverDnCgeqrf(*cusolverH, m, n, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::geqrf_get_lwork<std::complex<double>>((*cusolverH).get_device(), m, n,  lda, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::geqrf_get_lwork<std::complex<double>>((*cusolverH).get_device(), m, n,  lda, lwork64);
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
    // CHECK-NEXT: status = (mkl::lapack::geqrf(*cusolverH, m, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, Lwork, result_temp_buffer8), 0);
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
    // CHECK-NEXT: mkl::lapack::geqrf(*cusolverH, m, n, buffer_ct3, lda, buffer_ct5, buffer_ct6, Lwork, result_temp_buffer8);
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnZgeqrf_bufferSize(*cusolverH, m, n, &A_z, lda, &Lwork);
    cusolverDnZgeqrf_bufferSize(*cusolverH, m, n, &A_z, lda, &Lwork);
    status = cusolverDnZgeqrf(*cusolverH, m, n, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);
    cusolverDnZgeqrf(*cusolverH, m, n, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::ormqr_get_lwork<float>((*cusolverH).get_device(), side, trans, m, n, k,  lda,   ldc, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::ormqr_get_lwork<float>((*cusolverH).get_device(), side, trans, m, n, k,  lda,   ldc, lwork64);
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
    // CHECK-NEXT: status = (mkl::lapack::ormqr(*cusolverH, side, trans, m, n, k, buffer_ct6, lda, buffer_ct8, buffer_ct9, ldb, buffer_ct11, Lwork, result_temp_buffer13), 0);
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
    // CHECK-NEXT: mkl::lapack::ormqr(*cusolverH, side, trans, m, n, k, buffer_ct6, lda, buffer_ct8, buffer_ct9, ldb, buffer_ct11, Lwork, result_temp_buffer13);
    // CHECK-NEXT: buffer_ct13.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer13.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnSormqr_bufferSize(*cusolverH, side, trans, m, n, k, &A_f, lda, &TAU_f, &C_f, ldc, &Lwork);
    cusolverDnSormqr_bufferSize(*cusolverH, side, trans, m, n, k, &A_f, lda, &TAU_f, &C_f, ldc, &Lwork);
    status = cusolverDnSormqr(*cusolverH, side, trans, m, n, k, &A_f, lda, &TAU_f, &B_f, ldb, &workspace_f, Lwork, &devInfo);
    cusolverDnSormqr(*cusolverH, side, trans, m, n, k, &A_f, lda, &TAU_f, &B_f, ldb, &workspace_f, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::ormqr_get_lwork<double>((*cusolverH).get_device(), side, trans, m, n, k,  lda,   ldc, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::ormqr_get_lwork<double>((*cusolverH).get_device(), side, trans, m, n, k,  lda,   ldc, lwork64);
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
    // CHECK-NEXT: status = (mkl::lapack::ormqr(*cusolverH, side, trans, m, n, k, buffer_ct6, lda, buffer_ct8, buffer_ct9, ldb, buffer_ct11, Lwork, result_temp_buffer13), 0);
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
    // CHECK-NEXT: mkl::lapack::ormqr(*cusolverH, side, trans, m, n, k, buffer_ct6, lda, buffer_ct8, buffer_ct9, ldb, buffer_ct11, Lwork, result_temp_buffer13);
    // CHECK-NEXT: buffer_ct13.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer13.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnDormqr_bufferSize(*cusolverH, side, trans, m, n, k, &A_d, lda, &TAU_d, &C_d, ldc, &Lwork);
    cusolverDnDormqr_bufferSize(*cusolverH, side, trans, m, n, k, &A_d, lda, &TAU_d, &C_d, ldc, &Lwork);
    status = cusolverDnDormqr(*cusolverH, side, trans, m, n, k, &A_d, lda, &TAU_d, &B_d, ldb, &workspace_d, Lwork, &devInfo);
    cusolverDnDormqr(*cusolverH, side, trans, m, n, k, &A_d, lda, &TAU_d, &B_d, ldb, &workspace_d, Lwork, &devInfo);


    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::unmqr_get_lwork<std::complex<float>>((*cusolverH).get_device(), side, trans, m, n, k,  lda,   ldc, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::unmqr_get_lwork<std::complex<float>>((*cusolverH).get_device(), side, trans, m, n, k,  lda,   ldc, lwork64);
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
    // CHECK-NEXT: status = (mkl::lapack::unmqr(*cusolverH, side, trans, m, n, k, buffer_ct6, lda, buffer_ct8, buffer_ct9, ldb, buffer_ct11, Lwork, result_temp_buffer13), 0);
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
    // CHECK-NEXT: mkl::lapack::unmqr(*cusolverH, side, trans, m, n, k, buffer_ct6, lda, buffer_ct8, buffer_ct9, ldb, buffer_ct11, Lwork, result_temp_buffer13);
    // CHECK-NEXT: buffer_ct13.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer13.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnCunmqr_bufferSize(*cusolverH, side, trans, m, n, k, &A_c, lda, &TAU_c, &C_c, ldc, &Lwork);
    cusolverDnCunmqr_bufferSize(*cusolverH, side, trans, m, n, k, &A_c, lda, &TAU_c, &C_c, ldc, &Lwork);
    status = cusolverDnCunmqr(*cusolverH, side, trans, m, n, k, &A_c, lda, &TAU_c, &B_c, ldb, &workspace_c, Lwork, &devInfo);
    cusolverDnCunmqr(*cusolverH, side, trans, m, n, k, &A_c, lda, &TAU_c, &B_c, ldb, &workspace_c, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::unmqr_get_lwork<std::complex<double>>((*cusolverH).get_device(), side, trans, m, n, k,  lda,   ldc, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::unmqr_get_lwork<std::complex<double>>((*cusolverH).get_device(), side, trans, m, n, k,  lda,   ldc, lwork64);
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
    // CHECK-NEXT: status = (mkl::lapack::unmqr(*cusolverH, side, trans, m, n, k, buffer_ct6, lda, buffer_ct8, buffer_ct9, ldb, buffer_ct11, Lwork, result_temp_buffer13), 0);
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
    // CHECK-NEXT: mkl::lapack::unmqr(*cusolverH, side, trans, m, n, k, buffer_ct6, lda, buffer_ct8, buffer_ct9, ldb, buffer_ct11, Lwork, result_temp_buffer13);
    // CHECK-NEXT: buffer_ct13.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer13.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnZunmqr_bufferSize(*cusolverH, side, trans, m, n, k, &A_z, lda, &TAU_z, &C_z, ldc, &Lwork);
    cusolverDnZunmqr_bufferSize(*cusolverH, side, trans, m, n, k, &A_z, lda, &TAU_z, &C_z, ldc, &Lwork);
    status = cusolverDnZunmqr(*cusolverH, side, trans, m, n, k, &A_z, lda, &TAU_z, &B_z, ldb, &workspace_z, Lwork, &devInfo);
    cusolverDnZunmqr(*cusolverH, side, trans, m, n, k, &A_z, lda, &TAU_z, &B_z, ldb, &workspace_z, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::orgqr_get_lwork<float>((*cusolverH).get_device(), m, n, k,  lda,  lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::orgqr_get_lwork<float>((*cusolverH).get_device(), m, n, k,  lda,  lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct4 = allocation_ct4.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct4.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&TAU_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct6 = allocation_ct6.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct6.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&workspace_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct7.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct9 = allocation_ct9.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct9.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer9(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::orgqr(*cusolverH, m, n, k, buffer_ct4, lda, buffer_ct6, buffer_ct7, Lwork, result_temp_buffer9), 0);
    // CHECK-NEXT: buffer_ct9.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer9.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct4 = allocation_ct4.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct4.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&TAU_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct6 = allocation_ct6.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct6.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&workspace_f);
    // CHECK-NEXT: cl::sycl::buffer<float> buffer_ct7 = allocation_ct7.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct7.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct9 = allocation_ct9.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct9.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer9(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::orgqr(*cusolverH, m, n, k, buffer_ct4, lda, buffer_ct6, buffer_ct7, Lwork, result_temp_buffer9);
    // CHECK-NEXT: buffer_ct9.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer9.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnSorgqr_bufferSize(*cusolverH, m, n, k, &A_f, lda, &TAU_f, &Lwork);
    cusolverDnSorgqr_bufferSize(*cusolverH, m, n, k, &A_f, lda, &TAU_f, &Lwork);
    status = cusolverDnSorgqr(*cusolverH, m, n, k, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);
    cusolverDnSorgqr(*cusolverH, m, n, k, &A_f, lda, &TAU_f, &workspace_f, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::orgqr_get_lwork<double>((*cusolverH).get_device(), m, n, k,  lda,  lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::orgqr_get_lwork<double>((*cusolverH).get_device(), m, n, k,  lda,  lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct4 = allocation_ct4.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct4.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&TAU_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct6 = allocation_ct6.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct6.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&workspace_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct7 = allocation_ct7.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct7.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct9 = allocation_ct9.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct9.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer9(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::orgqr(*cusolverH, m, n, k, buffer_ct4, lda, buffer_ct6, buffer_ct7, Lwork, result_temp_buffer9), 0);
    // CHECK-NEXT: buffer_ct9.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer9.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct4 = allocation_ct4.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct4.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&TAU_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct6 = allocation_ct6.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct6.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&workspace_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct7 = allocation_ct7.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct7.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct9 = allocation_ct9.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct9.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer9(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::orgqr(*cusolverH, m, n, k, buffer_ct4, lda, buffer_ct6, buffer_ct7, Lwork, result_temp_buffer9);
    // CHECK-NEXT: buffer_ct9.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer9.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnDorgqr_bufferSize(*cusolverH, m, n, k, &A_d, lda, &TAU_d, &Lwork);
    cusolverDnDorgqr_bufferSize(*cusolverH, m, n, k, &A_d, lda, &TAU_d, &Lwork);
    status = cusolverDnDorgqr(*cusolverH, m, n, k, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);
    cusolverDnDorgqr(*cusolverH, m, n, k, &A_d, lda, &TAU_d, &workspace_d, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::ungqr_get_lwork<std::complex<float>>((*cusolverH).get_device(), m, n, k,  lda,  lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::ungqr_get_lwork<std::complex<float>>((*cusolverH).get_device(), m, n, k,  lda,  lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct4 = allocation_ct4.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct4.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&TAU_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&workspace_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct9 = allocation_ct9.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct9.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer9(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::ungqr(*cusolverH, m, n, k, buffer_ct4, lda, buffer_ct6, buffer_ct7, Lwork, result_temp_buffer9), 0);
    // CHECK-NEXT: buffer_ct9.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer9.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct4 = allocation_ct4.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct4.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&TAU_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&workspace_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct9 = allocation_ct9.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct9.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer9(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::ungqr(*cusolverH, m, n, k, buffer_ct4, lda, buffer_ct6, buffer_ct7, Lwork, result_temp_buffer9);
    // CHECK-NEXT: buffer_ct9.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer9.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnCungqr_bufferSize(*cusolverH, m, n, k, &A_c, lda, &TAU_c, &Lwork);
    cusolverDnCungqr_bufferSize(*cusolverH, m, n, k, &A_c, lda, &TAU_c, &Lwork);
    status = cusolverDnCungqr(*cusolverH, m, n, k, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);
    cusolverDnCungqr(*cusolverH, m, n, k, &A_c, lda, &TAU_c, &workspace_c, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: status = (mkl::lapack::ungqr_get_lwork<std::complex<double>>((*cusolverH).get_device(), m, n, k,  lda,  lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::lapack::ungqr_get_lwork<std::complex<double>>((*cusolverH).get_device(), m, n, k,  lda,  lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct4 = allocation_ct4.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct4.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&TAU_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&workspace_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct9 = allocation_ct9.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct9.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer9(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::ungqr(*cusolverH, m, n, k, buffer_ct4, lda, buffer_ct6, buffer_ct7, Lwork, result_temp_buffer9), 0);
    // CHECK-NEXT: buffer_ct9.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer9.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct4 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct4 = allocation_ct4.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct4.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&TAU_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::mem_mgr::instance().translate_ptr(&workspace_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct7 = allocation_ct7.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct7.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct9 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct9 = allocation_ct9.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct9.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer9(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::ungqr(*cusolverH, m, n, k, buffer_ct4, lda, buffer_ct6, buffer_ct7, Lwork, result_temp_buffer9);
    // CHECK-NEXT: buffer_ct9.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer9.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnZungqr_bufferSize(*cusolverH, m, n, k, &A_z, lda, &TAU_z, &Lwork);
    cusolverDnZungqr_bufferSize(*cusolverH, m, n, k, &A_z, lda, &TAU_z, &Lwork);
    status = cusolverDnZungqr(*cusolverH, m, n, k, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);
    cusolverDnZungqr(*cusolverH, m, n, k, &A_z, lda, &TAU_z, &workspace_z, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::uplo uplo_ct_mkl_upper_lower;
    // CHECK-NEXT: status = (mkl::lapack::sytrf_get_lwork<float>((*cusolverH).get_device(), uplo_ct_mkl_upper_lower, n,  lda, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::uplo uplo_ct_mkl_upper_lower;
    // CHECK-NEXT: mkl::lapack::sytrf_get_lwork<float>((*cusolverH).get_device(), uplo_ct_mkl_upper_lower, n,  lda, lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT:/*
    // CHECK-NEXT:DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT:*/
    // CHECK-NEXT:{
    // CHECK-NEXT:auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT:cl::sycl::buffer<float> buffer_ct3 = allocation_ct3.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct3.size/sizeof(float)));
    // CHECK-NEXT:auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT:cl::sycl::buffer<int> buffer_ct5 = allocation_ct5.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct5.size/sizeof(int)));
    // CHECK-NEXT:cl::sycl::buffer<int64_t> result_temp_buffer5(cl::sycl::range<1>(1));
    // CHECK-NEXT:auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&workspace_f);
    // CHECK-NEXT:cl::sycl::buffer<float> buffer_ct6 = allocation_ct6.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct6.size/sizeof(float)));
    // CHECK-NEXT:auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT:cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT:cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT:status = (mkl::lapack::sytrf(*cusolverH, uplo, n, buffer_ct3, lda, result_temp_buffer5, buffer_ct6, Lwork, result_temp_buffer8), 0);
    // CHECK-NEXT:buffer_ct5.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer5.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT:buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT:}
    // CHECK-NEXT:{
    // CHECK-NEXT:auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_f);
    // CHECK-NEXT:cl::sycl::buffer<float> buffer_ct3 = allocation_ct3.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct3.size/sizeof(float)));
    // CHECK-NEXT:auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT:cl::sycl::buffer<int> buffer_ct5 = allocation_ct5.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct5.size/sizeof(int)));
    // CHECK-NEXT:cl::sycl::buffer<int64_t> result_temp_buffer5(cl::sycl::range<1>(1));
    // CHECK-NEXT:auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&workspace_f);
    // CHECK-NEXT:cl::sycl::buffer<float> buffer_ct6 = allocation_ct6.buffer.reinterpret<float>(cl::sycl::range<1>(allocation_ct6.size/sizeof(float)));
    // CHECK-NEXT:auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT:cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT:cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT:mkl::lapack::sytrf(*cusolverH, uplo, n, buffer_ct3, lda, result_temp_buffer5, buffer_ct6, Lwork, result_temp_buffer8);
    // CHECK-NEXT:buffer_ct5.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer5.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT:buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT:}
    status = cusolverDnSsytrf_bufferSize(*cusolverH, n, &A_f, lda, &Lwork);
    cusolverDnSsytrf_bufferSize(*cusolverH, n, &A_f, lda, &Lwork);
    status = cusolverDnSsytrf(*cusolverH, uplo, n, &A_f, lda, &devIpiv, &workspace_f, Lwork, &devInfo);
    cusolverDnSsytrf(*cusolverH, uplo, n, &A_f, lda, &devIpiv, &workspace_f, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::uplo uplo_ct_mkl_upper_lower;
    // CHECK-NEXT: status = (mkl::lapack::sytrf_get_lwork<double>((*cusolverH).get_device(), uplo_ct_mkl_upper_lower, n,  lda, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::uplo uplo_ct_mkl_upper_lower;
    // CHECK-NEXT: mkl::lapack::sytrf_get_lwork<double>((*cusolverH).get_device(), uplo_ct_mkl_upper_lower, n,  lda, lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct3 = allocation_ct3.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct3.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct5 = allocation_ct5.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct5.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer5(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&workspace_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct6 = allocation_ct6.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct6.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::sytrf(*cusolverH, uplo, n, buffer_ct3, lda, result_temp_buffer5, buffer_ct6, Lwork, result_temp_buffer8), 0);
    // CHECK-NEXT: buffer_ct5.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer5.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct3 = allocation_ct3.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct3.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct5 = allocation_ct5.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct5.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer5(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&workspace_d);
    // CHECK-NEXT: cl::sycl::buffer<double> buffer_ct6 = allocation_ct6.buffer.reinterpret<double>(cl::sycl::range<1>(allocation_ct6.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::sytrf(*cusolverH, uplo, n, buffer_ct3, lda, result_temp_buffer5, buffer_ct6, Lwork, result_temp_buffer8);
    // CHECK-NEXT: buffer_ct5.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer5.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnDsytrf_bufferSize(*cusolverH, n, &A_d, lda, &Lwork);
    cusolverDnDsytrf_bufferSize(*cusolverH, n, &A_d, lda, &Lwork);
    status = cusolverDnDsytrf(*cusolverH, uplo, n, &A_d, lda, &devIpiv, &workspace_d, Lwork, &devInfo);
    cusolverDnDsytrf(*cusolverH, uplo, n, &A_d, lda, &devIpiv, &workspace_d, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::uplo uplo_ct_mkl_upper_lower;
    // CHECK-NEXT: status = (mkl::lapack::sytrf_get_lwork<std::complex<float>>((*cusolverH).get_device(), uplo_ct_mkl_upper_lower, n,  lda, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::uplo uplo_ct_mkl_upper_lower;
    // CHECK-NEXT: mkl::lapack::sytrf_get_lwork<std::complex<float>>((*cusolverH).get_device(), uplo_ct_mkl_upper_lower, n,  lda, lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct5 = allocation_ct5.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct5.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer5(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&workspace_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::sytrf(*cusolverH, uplo, n, buffer_ct3, lda, result_temp_buffer5, buffer_ct6, Lwork, result_temp_buffer8), 0);
    // CHECK-NEXT: buffer_ct5.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer5.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct5 = allocation_ct5.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct5.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer5(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&workspace_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<float>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::sytrf(*cusolverH, uplo, n, buffer_ct3, lda, result_temp_buffer5, buffer_ct6, Lwork, result_temp_buffer8);
    // CHECK-NEXT: buffer_ct5.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer5.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnCsytrf_bufferSize(*cusolverH, n, &A_c, lda, &Lwork);
    cusolverDnCsytrf_bufferSize(*cusolverH, n, &A_c, lda, &Lwork);
    status = cusolverDnCsytrf(*cusolverH, uplo, n, &A_c, lda, &devIpiv, &workspace_c, Lwork, &devInfo);
    cusolverDnCsytrf(*cusolverH, uplo, n, &A_c, lda, &devIpiv, &workspace_c, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::uplo uplo_ct_mkl_upper_lower;
    // CHECK-NEXT: status = (mkl::lapack::sytrf_get_lwork<std::complex<double>>((*cusolverH).get_device(), uplo_ct_mkl_upper_lower, n,  lda, lwork64), 0);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: int64_t lwork64 = *(&Lwork);
    // CHECK-NEXT: mkl::uplo uplo_ct_mkl_upper_lower;
    // CHECK-NEXT: mkl::lapack::sytrf_get_lwork<std::complex<double>>((*cusolverH).get_device(), uplo_ct_mkl_upper_lower, n,  lda, lwork64);
    // CHECK-NEXT: *(&Lwork) = lwork64;
    // CHECK-NEXT: }
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct5 = allocation_ct5.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct5.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer5(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&workspace_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::sytrf(*cusolverH, uplo, n, buffer_ct3, lda, result_temp_buffer5, buffer_ct6, Lwork, result_temp_buffer8), 0);
    // CHECK-NEXT: buffer_ct5.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer5.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::mem_mgr::instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct5 = dpct::mem_mgr::instance().translate_ptr(&devIpiv);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct5 = allocation_ct5.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct5.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer5(cl::sycl::range<1>(1));
    // CHECK-NEXT: auto allocation_ct6 = dpct::mem_mgr::instance().translate_ptr(&workspace_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>> buffer_ct6 = allocation_ct6.buffer.reinterpret<std::complex<double>>(cl::sycl::range<1>(allocation_ct6.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct8 = dpct::mem_mgr::instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int> buffer_ct8 = allocation_ct8.buffer.reinterpret<int>(cl::sycl::range<1>(allocation_ct8.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t> result_temp_buffer8(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::sytrf(*cusolverH, uplo, n, buffer_ct3, lda, result_temp_buffer5, buffer_ct6, Lwork, result_temp_buffer8);
    // CHECK-NEXT: buffer_ct5.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer5.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: buffer_ct8.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer8.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnZsytrf_bufferSize(*cusolverH, n, &A_z, lda, &Lwork);
    cusolverDnZsytrf_bufferSize(*cusolverH, n, &A_z, lda, &Lwork);
    status = cusolverDnZsytrf(*cusolverH, uplo, n, &A_z, lda, &devIpiv, &workspace_z, Lwork, &devInfo);
    cusolverDnZsytrf(*cusolverH, uplo, n, &A_z, lda, &devIpiv, &workspace_z, Lwork, &devInfo);
}
