// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.2, cuda-10.0
// UNSUPPORTED: v8.0, v9.0, v9.2, v10.0
// RUN: dpct --format-range=none -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
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
    //CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cusolverDnSpotri_bufferSize was replaced with 0, because Function call is redundant in DPC++.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = 0;
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cusolverDnDpotri_bufferSize was replaced with 0, because Function call is redundant in DPC++.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = 0;
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cusolverDnCpotri_bufferSize was replaced with 0, because Function call is redundant in DPC++.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = 0;
    //CHECK-NEXT: /*
    //CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cusolverDnZpotri_bufferSize was replaced with 0, because Function call is redundant in DPC++.
    //CHECK-NEXT: */
    //CHECK-NEXT: status = 0;
    status = cusolverDnSpotri_bufferSize(*cusolverH, uplo, n, &A_f, lda, &Lwork);
    status = cusolverDnDpotri_bufferSize(*cusolverH, uplo, n, &A_d, lda, &Lwork);
    status = cusolverDnCpotri_bufferSize(*cusolverH, uplo, n, &A_c, lda, &Lwork);
    status = cusolverDnZpotri_bufferSize(*cusolverH, uplo, n, &A_z, lda, &Lwork);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::memory_manager::get_instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct3 = allocation_ct3.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct3.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::memory_manager::get_instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int,1> buffer_ct7 = allocation_ct7.buffer.reinterpret<int, 1>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::potri(*cusolverH, uplo, n, buffer_ct3, lda,   result_temp_buffer7), 0);
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::memory_manager::get_instance().translate_ptr(&A_f);
    // CHECK-NEXT: cl::sycl::buffer<float,1> buffer_ct3 = allocation_ct3.buffer.reinterpret<float, 1>(cl::sycl::range<1>(allocation_ct3.size/sizeof(float)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::memory_manager::get_instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int,1> buffer_ct7 = allocation_ct7.buffer.reinterpret<int, 1>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::potri(*cusolverH, uplo, n, buffer_ct3, lda,   result_temp_buffer7);
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }

    status = cusolverDnSpotri(*cusolverH, uplo, n, &A_f, lda, &workspace_f, Lwork, &devInfo);
    cusolverDnSpotri(*cusolverH, uplo, n, &A_f, lda, &workspace_f, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::memory_manager::get_instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct3 = allocation_ct3.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct3.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::memory_manager::get_instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int,1> buffer_ct7 = allocation_ct7.buffer.reinterpret<int, 1>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::potri(*cusolverH, uplo, n, buffer_ct3, lda,   result_temp_buffer7), 0);
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::memory_manager::get_instance().translate_ptr(&A_d);
    // CHECK-NEXT: cl::sycl::buffer<double,1> buffer_ct3 = allocation_ct3.buffer.reinterpret<double, 1>(cl::sycl::range<1>(allocation_ct3.size/sizeof(double)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::memory_manager::get_instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int,1> buffer_ct7 = allocation_ct7.buffer.reinterpret<int, 1>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::potri(*cusolverH, uplo, n, buffer_ct3, lda,   result_temp_buffer7);
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnDpotri(*cusolverH, uplo, n, &A_d, lda, &workspace_d, Lwork, &devInfo);
    cusolverDnDpotri(*cusolverH, uplo, n, &A_d, lda, &workspace_d, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::memory_manager::get_instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::memory_manager::get_instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int,1> buffer_ct7 = allocation_ct7.buffer.reinterpret<int, 1>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::potri(*cusolverH, uplo, n, buffer_ct3, lda,   result_temp_buffer7), 0);
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::memory_manager::get_instance().translate_ptr(&A_c);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<float>,1> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<float>, 1>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<float>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::memory_manager::get_instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int,1> buffer_ct7 = allocation_ct7.buffer.reinterpret<int, 1>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::potri(*cusolverH, uplo, n, buffer_ct3, lda,   result_temp_buffer7);
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnCpotri(*cusolverH, uplo, n, &A_c, lda, &workspace_c, Lwork, &devInfo);
    cusolverDnCpotri(*cusolverH, uplo, n, &A_c, lda, &workspace_c, Lwork, &devInfo);

    // CHECK: /*
    // CHECK-NEXT: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code.
    // CHECK-NEXT: */
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::memory_manager::get_instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::memory_manager::get_instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int,1> buffer_ct7 = allocation_ct7.buffer.reinterpret<int, 1>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: status = (mkl::lapack::potri(*cusolverH, uplo, n, buffer_ct3, lda,   result_temp_buffer7), 0);
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    // CHECK-NEXT: {
    // CHECK-NEXT: auto allocation_ct3 = dpct::memory_manager::get_instance().translate_ptr(&A_z);
    // CHECK-NEXT: cl::sycl::buffer<std::complex<double>,1> buffer_ct3 = allocation_ct3.buffer.reinterpret<std::complex<double>, 1>(cl::sycl::range<1>(allocation_ct3.size/sizeof(std::complex<double>)));
    // CHECK-NEXT: auto allocation_ct7 = dpct::memory_manager::get_instance().translate_ptr(&devInfo);
    // CHECK-NEXT: cl::sycl::buffer<int,1> buffer_ct7 = allocation_ct7.buffer.reinterpret<int, 1>(cl::sycl::range<1>(allocation_ct7.size/sizeof(int)));
    // CHECK-NEXT: cl::sycl::buffer<int64_t,1> result_temp_buffer7(cl::sycl::range<1>(1));
    // CHECK-NEXT: mkl::lapack::potri(*cusolverH, uplo, n, buffer_ct3, lda,   result_temp_buffer7);
    // CHECK-NEXT: buffer_ct7.get_access<cl::sycl::access::mode::write>()[0] = (int)result_temp_buffer7.get_access<cl::sycl::access::mode::read>()[0];
    // CHECK-NEXT: }
    status = cusolverDnZpotri(*cusolverH, uplo, n, &A_z, lda, &workspace_z, Lwork, &devInfo);
    cusolverDnZpotri(*cusolverH, uplo, n, &A_z, lda, &workspace_z, Lwork, &devInfo);
}
