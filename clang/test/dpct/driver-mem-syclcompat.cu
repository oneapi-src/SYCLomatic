// RUN: dpct --format-range=none -use-syclcompat -out-root %T/driver-mem-syclcompat %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/driver-mem-syclcompat/driver-mem-syclcompat.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/driver-mem-syclcompat/driver-mem-syclcompat.dp.cpp -o %T/driver-mem/driver-mem-syclcompat.dp.o %}

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

#define CALL(x) x

void cuCheckError(CUresult err) {
}

int main(){
    size_t result1, result2;
    int size = 32;
    size_t count = 32;
    float* f_A;
    CUresult r;
    // CHECK: f_A = (float *)sycl::malloc_host(size, q_ct1);
    cuMemHostAlloc((void **)&f_A, size, CU_MEMHOSTALLOC_DEVICEMAP);


    CUdeviceptr p1;
    [&p1]() {
        //CHECK:p1 = 0;
        p1 = 0;
    }();

    CUdeviceptr p2;
    [&]() {
        //CHECK:p2 = 0;
        p2 = 0;
    }();

    // CHECK: syclcompat::device_ptr f_D = 0;
    CUdeviceptr f_D = 0;
    // CHECK: syclcompat::device_ptr f_D2 = 0;
    CUdeviceptr f_D2 = 0;
    // CHECK: int c1, c2;
    CUcontext c1, c2;
    // CHECK: f_D = (syclcompat::device_ptr)sycl::malloc_device(size, q_ct1);
    cuMemAlloc(&f_D, size);

    // CHECK: syclcompat::queue_ptr stream;
    CUstream stream;
    // CHECK: stream->memcpy(f_D, f_A, size);
    cuMemcpyHtoDAsync(f_D, f_A, size, stream);
    // CHECK: q_ct1.memcpy(f_D, f_A, size);
    cuMemcpyHtoDAsync(f_D, f_A, size, 0);
    // CHECK: CALL(q_ct1.memcpy(f_D, f_A, size).wait());
    CALL(cuMemcpyHtoD(f_D, f_A, size));

    // CHECK: stream->memcpy(f_A, f_D, size);
    cuMemcpyDtoHAsync(f_A, f_D, size, stream);
    // CHECK: q_ct1.memcpy(f_A, f_D, size);
    cuMemcpyDtoHAsync(f_A, f_D, size, 0);
    // CHECK: q_ct1.memcpy(f_A, f_D, size).wait();
    cuMemcpyDtoH(f_A, f_D, size);

    // CHECK: stream->memcpy(f_D, f_D2, size);
    cuMemcpyDtoDAsync(f_D, f_D2, size, stream);
    // CHECK: r = SYCLCOMPAT_CHECK_ERROR(stream->memcpy(f_D, f_D2, size));
    r = cuMemcpyDtoDAsync(f_D, f_D2, size, stream);

    // CHECK: q_ct1.memcpy(f_D, f_D2, size);
    cuMemcpyDtoDAsync(f_D, f_D2, size, 0);
    // CHECK: r = SYCLCOMPAT_CHECK_ERROR(q_ct1.memcpy(f_D, f_D2, size));
    r = cuMemcpyDtoDAsync(f_D, f_D2, size, 0);

    // CHECK: q_ct1.memcpy(f_D, f_D2, size).wait();
    cuMemcpyDtoD(f_D, f_D2, size);
    // CHECK: r = SYCLCOMPAT_CHECK_ERROR(q_ct1.memcpy(f_D, f_D2, size).wait());
    r = cuMemcpyDtoD(f_D, f_D2, size);

    // CHECK: q_ct1.memcpy(f_D, f_D2, size).wait();
    cuMemcpy(f_D, f_D2, size);
    // CHECK: CALL(q_ct1.memcpy(f_D, f_D2, size).wait());
    CALL(cuMemcpy(f_D, f_D2, size));
    // CHECK: r = SYCLCOMPAT_CHECK_ERROR(q_ct1.memcpy(f_D, f_D2, size).wait());
    r = cuMemcpy(f_D, f_D2, size);

    // CHECK: stream->memcpy(f_D, f_D2, size);
    cuMemcpyAsync(f_D, f_D2, size, stream);
    // CHECK: CALL(stream->memcpy(f_D, f_D2, size));
    CALL(cuMemcpyAsync(f_D, f_D2, size, stream));
    // CHECK: r = SYCLCOMPAT_CHECK_ERROR(stream->memcpy(f_D, f_D2, size));
    r = cuMemcpyAsync(f_D, f_D2, size, stream);

    // CHECK: q_ct1.memcpy(f_D, f_D2, size);
    cuMemcpyAsync(f_D, f_D2, size, 0);
    // CHECK: CALL(q_ct1.memcpy(f_D, f_D2, size));
    CALL(cuMemcpyAsync(f_D, f_D2, size, 0));
    // CHECK: r = SYCLCOMPAT_CHECK_ERROR(q_ct1.memcpy(f_D, f_D2, size));
    r = cuMemcpyAsync(f_D, f_D2, size, 0);

#ifndef NO_BUILD_TEST
    // CHECK: DPCT1131:{{[0-9]+}}: The migration of "cuMemcpyPeer" is not currently supported with SYCLcompat. Please adjust the code manually.
    cuMemcpyPeer(f_D, c1, f_D2, c2, size);
    // CHECK: DPCT1131:{{[0-9]+}}: The migration of "cuMemcpyPeerAsync" is not currently supported with SYCLcompat. Please adjust the code manually.
    cuMemcpyPeerAsync(f_D, c1, f_D2, c2, size, stream);
#endif

    unsigned int v32 = 50000;
    unsigned short v16 = 20000;
    unsigned char v8 = (unsigned char) 200;
    //CHECK: syclcompat::memset_d32(f_D, v32, size);
    //CHECK-NEXT: syclcompat::memset_d16(f_D, v16, size * 2);
    //CHECK-NEXT: syclcompat::memset(f_D, v8, size * 4);
    //CHECK-NEXT: syclcompat::memset_d32_async(f_D, v32, size, *stream);
    //CHECK-NEXT: syclcompat::memset_d16_async(f_D, v16, size * 2, *stream);
    //CHECK-NEXT: syclcompat::memset_async(f_D, v8, size * 4, *stream);
    //CHECK-NEXT: syclcompat::memset_d32(f_D, 1, v32, 4, 6);
    //CHECK-NEXT: syclcompat::memset_d16(f_D, 1, v16, 4 * 2, 6);
    //CHECK-NEXT: syclcompat::memset(f_D, 1, v8, 4 * 4, 6);
    //CHECK-NEXT: syclcompat::memset_d32_async(f_D, 1, v32, 4, 6, *stream);
    //CHECK-NEXT: syclcompat::memset_d16_async(f_D, 1, v16, 4 * 2, 6, *stream);
    //CHECK-NEXT: syclcompat::memset_async(f_D, 1, v8, 4 * 4, 6, *stream);
    cuMemsetD32(f_D, v32, size);
    cuMemsetD16(f_D, v16, size * 2);
    cuMemsetD8(f_D, v8, size * 4);
    cuMemsetD32Async(f_D, v32, size, stream);
    cuMemsetD16Async(f_D, v16, size * 2, stream);
    cuMemsetD8Async(f_D, v8, size * 4, stream);
    cuMemsetD2D32(f_D, 1, v32, 4, 6);
    cuMemsetD2D16(f_D, 1, v16, 4 * 2, 6);
    cuMemsetD2D8(f_D, 1, v8, 4 * 4, 6);
    cuMemsetD2D32Async(f_D, 1, v32, 4, 6, stream);
    cuMemsetD2D16Async(f_D, 1, v16, 4 * 2, 6, stream);
    cuMemsetD2D8Async(f_D, 1, v8, 4 * 4, 6, stream);

    // CHECK: syclcompat::experimental::memcpy_parameter cpy;
    // CHECK-NEXT: cpy.to.pitched.set_data_ptr(f_A);
    // CHECK-NEXT: cpy.to.pitched.set_pitch(20);
    // CHECK-NEXT: cpy.to.pos[1] = 10;
    // CHECK-NEXT: cpy.to.pos[0] = 15;
    // CHECK-EMPTY:
    // CHECK-NEXT: cpy.from.pitched.set_data_ptr(f_D);
    // CHECK-NEXT: cpy.from.pitched.set_pitch(20);
    // CHECK-NEXT: cpy.from.pos[1] = 10;
    // CHECK-NEXT: cpy.from.pos[0] = 15;
    // CHECK-EMPTY:
    // CHECK-NEXT: cpy.size[0] = 4;
    // CHECK-NEXT: cpy.size[1] = 7;
    CUDA_MEMCPY2D cpy;
    cpy.dstMemoryType = CU_MEMORYTYPE_HOST;
    cpy.dstHost = f_A;
    cpy.dstPitch = 20;
    cpy.dstY = 10;
    cpy.dstXInBytes = 15;

    cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cpy.srcDevice = f_D;
    cpy.srcPitch = 20;
    cpy.srcY = 10;
    cpy.srcXInBytes = 15;

    cpy.WidthInBytes = 4;
    cpy.Height = 7;

    // CHECK: syclcompat::experimental::memcpy(cpy);
    cuMemcpy2D(&cpy);
    // CHECK: syclcompat::experimental::memcpy_async(cpy, *stream);
    cuMemcpy2DAsync(&cpy, stream);

    CUdeviceptr devicePtr;

    CUresult cu_err;

    CUdeviceptr cuDevPtr;

    CUdevice cudevice;
    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: int advise = 0;
    CUmem_advise advise = CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION;

    // CHECK: syclcompat::get_device(cudevice).default_queue()->mem_advise(devicePtr, count, advise);
    cuMemAdvise(devicePtr, count, advise, cudevice);

    // CHECK: cuCheckError(SYCLCOMPAT_CHECK_ERROR(syclcompat::get_device(cudevice).default_queue()->mem_advise(devicePtr, count, advise)));
    cuCheckError(cuMemAdvise(devicePtr, count, advise, cudevice));

    // CHECK: cu_err = SYCLCOMPAT_CHECK_ERROR(syclcompat::get_device(cudevice).default_queue()->mem_advise(devicePtr, count, advise));
    cu_err = cuMemAdvise(devicePtr, count, advise, cudevice);

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: syclcompat::get_device(cudevice).default_queue()->mem_advise(devicePtr, count, 0);
    cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, cudevice);

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError(SYCLCOMPAT_CHECK_ERROR(syclcompat::get_device(cudevice).default_queue()->mem_advise(devicePtr, count, 0)));
    cuCheckError(cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, cudevice));

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError(SYCLCOMPAT_CHECK_ERROR(syclcompat::get_device(cudevice).default_queue()->mem_advise(devicePtr, count, 0)));
    cuCheckError(cuMemAdvise(devicePtr, count, (CUmem_advise)1, cudevice));

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError(SYCLCOMPAT_CHECK_ERROR(syclcompat::get_device(cudevice).default_queue()->mem_advise(devicePtr, count, 0)));
    cuCheckError(cuMemAdvise(devicePtr, count, CUmem_advise(1), cudevice));

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError(SYCLCOMPAT_CHECK_ERROR(syclcompat::get_device(cudevice).default_queue()->mem_advise(devicePtr, count, 0)));
    cuCheckError(cuMemAdvise(devicePtr, count, static_cast<CUmem_advise>(1), cudevice));

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: cu_err = SYCLCOMPAT_CHECK_ERROR(syclcompat::get_device(cudevice).default_queue()->mem_advise(devicePtr, count, 0));
    cu_err = cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, cudevice);

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: syclcompat::get_device(cudevice).default_queue()->mem_advise(devicePtr, count, 0);
    cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, cudevice);

    // CHECK: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: syclcompat::cpu_device().default_queue()->mem_advise(devicePtr, count, 0);
    cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, CU_DEVICE_CPU);


    CUdeviceptr devPtr;
    CUresult curesult;
    // CHECK: stream->prefetch(devPtr, 100);
    cuMemPrefetchAsync (devPtr, 100, cudevice, stream);
    // CHECK: (*&stream)->prefetch(devPtr, 100);
    cuMemPrefetchAsync (devPtr, 100, cudevice, *&stream);
    // CHECK: curesult = SYCLCOMPAT_CHECK_ERROR(syclcompat::get_device(cudevice).default_queue()->prefetch(devPtr, 100));
    curesult = cuMemPrefetchAsync (devPtr, 100, cudevice, NULL);
    // CHECK: syclcompat::get_device(cudevice).default_queue()->prefetch(devPtr, 100);
    cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamPerThread);
    // CHECK: curesult = SYCLCOMPAT_CHECK_ERROR(syclcompat::get_device(cudevice).default_queue()->prefetch(devPtr, 100));
    curesult = cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamDefault);
    // CHECK: curesult = SYCLCOMPAT_CHECK_ERROR(syclcompat::get_device(cudevice).default_queue()->prefetch(devPtr, 100));
    curesult = cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamLegacy);
    // CHECK: curesult = SYCLCOMPAT_CHECK_ERROR(syclcompat::get_device(cudevice).default_queue()->prefetch(devPtr, 100));
    curesult = cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamPerThread);
    // CHECK: cuCheckError(SYCLCOMPAT_CHECK_ERROR(syclcompat::get_device(cudevice).default_queue()->prefetch(devPtr, 100)));
    cuCheckError(cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamDefault));
    // CHECK: cuCheckError(SYCLCOMPAT_CHECK_ERROR(syclcompat::get_device(cudevice).default_queue()->prefetch(devPtr, 100)));
    cuCheckError(cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamLegacy));
    // CHECK: cuCheckError(SYCLCOMPAT_CHECK_ERROR(syclcompat::get_device(cudevice).default_queue()->prefetch(devPtr, 100)));
    cuCheckError(cuMemPrefetchAsync (devPtr, 100, cudevice, cudaStreamPerThread));

    // CHECK: syclcompat::experimental::memcpy_parameter cpy2;
    // CHECK: /*
    // CHECK-NEXT: DPCT1131:{{[0-9]+}}: The migration of "CUarray" is not currently supported with SYCLcompat. Please adjust the code manually.
    // CHECK-NEXT: */
    // CHECK-NEXT: CUarray ca;
    // CHECK-NEXT: cpy2.to.image = ca;
    // CHECK-NEXT: cpy2.to.pitched.set_pitch(5);
    // CHECK-NEXT: cpy2.to.pitched.set_y(4);
    // CHECK-NEXT: cpy2.to.pos[1] = 3;
    // CHECK-NEXT: cpy2.to.pos[2] = 2;
    // CHECK-NEXT: cpy2.to.pos[0] = 1;
    // CHECK: cpy2.from.pitched.set_data_ptr(f_A);
    // CHECK-NEXT: cpy2.from.pitched.set_pitch(5);
    // CHECK-NEXT: cpy2.from.pitched.set_y(4);
    // CHECK-NEXT: cpy2.from.pos[1] = 3;
    // CHECK-NEXT: cpy2.from.pos[2] = 2;
    // CHECK-NEXT: cpy2.from.pos[0] = 1;
    // CHECK-EMPTY:
    // CHECK-NEXT: cpy2.size[0] = 3;
    // CHECK-NEXT: cpy2.size[1] = 2;
    // CHECK-NEXT: cpy2.size[2] = 1;
    CUDA_MEMCPY3D cpy2;
#ifndef NO_BUILD_TEST
    CUarray ca;
    cpy2.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    cpy2.dstArray = ca;
    cpy2.dstPitch = 5;
    cpy2.dstHeight = 4;
    cpy2.dstY = 3;
    cpy2.dstZ = 2;
    cpy2.dstXInBytes = 1;
    cpy2.dstLOD = 0;
#endif

    cpy2.srcMemoryType = CU_MEMORYTYPE_HOST;
    cpy2.srcHost = f_A;
    cpy2.srcPitch = 5;
    cpy2.srcHeight = 4;
    cpy2.srcY = 3;
    cpy2.srcZ = 2;
    cpy2.srcXInBytes = 1;
    cpy2.srcLOD = 0;

    cpy2.WidthInBytes = 3;
    cpy2.Height = 2;
    cpy2.Depth = 1;

    // CHECK: syclcompat::experimental::memcpy(cpy2);
    cuMemcpy3D(&cpy2);

    CUstream cs;
    // CHECK: syclcompat::experimental::memcpy_async(cpy2, *cs);
    cuMemcpy3DAsync(&cpy2, cs);

    float *h_A = (float *)malloc(100);
    // CHECK:sycl::free(h_A, q_ct1);
    cuMemFreeHost(h_A);
    // CHECK:sycl::free(f_D, q_ct1);
    cuMemFree(f_D);

    unsigned int flags;
    int host;


    // CHECK: flags = 0;
    cuMemHostGetFlags(&flags, &host);
    // CHECK: cuCheckError(SYCLCOMPAT_CHECK_ERROR(flags = 0));
    cuCheckError(cuMemHostGetFlags(&flags, &host));

    // CHECK:  /*
    // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuMemHostRegister was removed because SYCL currently does not support registering of existing host memory for use by device. Use USM to allocate memory for use by host and device.
    // CHECK-NEXT: */
    cuMemHostRegister(h_A, count, flags);
    // CHECK:  /*
    // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cuMemHostRegister was replaced with 0 because SYCL currently does not support registering of existing host memory for use by device. Use USM to allocate memory for use by host and device.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError(0);
    cuCheckError(cuMemHostRegister(h_A, count, flags));


    // CHECK:  /*
    // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuMemHostUnregister was removed because SYCL currently does not support registering of existing host memory for use by device. Use USM to allocate memory for use by host and device.
    // CHECK-NEXT: */
    cuMemHostUnregister(h_A);

    // CHECK:  /*
    // CHECK-NEXT: DPCT1027:{{[0-9]+}}: The call to cuMemHostUnregister was replaced with 0 because SYCL currently does not support registering of existing host memory for use by device. Use USM to allocate memory for use by host and device.
    // CHECK-NEXT: */
    // CHECK-NEXT:cuCheckError(0);
    cuCheckError(cuMemHostUnregister(h_A));
    return 0;
}
