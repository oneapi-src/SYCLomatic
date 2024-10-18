// RUN: dpct --usm-level=none --format-range=none -out-root %T/driver-mem-usm-none %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/driver-mem-usm-none/driver-mem-usm-none.dp.cpp %s

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

#define CALL(x) x

void cuCheckError(CUresult err) {
}

int main(){
    size_t result1, result2;
    int size = 32;
    float* f_A;
    CUresult r;
    // CHECK: f_A = (float *)malloc(size);
    cuMemHostAlloc((void **)&f_A, size, CU_MEMHOSTALLOC_DEVICEMAP);

    // CHECK: dpct::device_ptr f_D = 0;
    CUdeviceptr f_D = 0;
    // CHECK: dpct::device_ptr f_D2 = 0;
    CUdeviceptr f_D2 = 0;
    // CHECK: f_D = (dpct::device_ptr)dpct::dpct_malloc(size);
    cuMemAlloc(&f_D, size);

    // CHECK: dpct::queue_ptr stream;
    CUstream stream;
    // CHECK: dpct::async_dpct_memcpy(f_D, f_A, size, dpct::automatic, *stream);
    cuMemcpyHtoDAsync(f_D, f_A, size, stream);
    // CHECK: dpct::async_dpct_memcpy(f_D, f_A, size, dpct::automatic);
    cuMemcpyHtoDAsync(f_D, f_A, size, 0);
    // CHECK: CALL(dpct::dpct_memcpy(f_D, f_A, size, dpct::automatic));
    CALL(cuMemcpyHtoD(f_D, f_A, size));

    // CHECK: dpct::async_dpct_memcpy(f_A, f_D, size, dpct::automatic, *stream);
    cuMemcpyDtoHAsync(f_A, f_D, size, stream);
    // CHECK: dpct::async_dpct_memcpy(f_A, f_D, size, dpct::automatic);
    cuMemcpyDtoHAsync(f_A, f_D, size, 0);
    // CHECK: dpct::dpct_memcpy(f_A, f_D, size, dpct::automatic);
    cuMemcpyDtoH(f_A, f_D, size);

    // CHECK: dpct::async_dpct_memcpy(f_D, f_D2, size, dpct::automatic, *stream);
    cuMemcpyDtoDAsync(f_D, f_D2, size, stream);
    // CHECK: dpct::async_dpct_memcpy(f_D, f_D2, size, dpct::automatic);
    cuMemcpyDtoDAsync(f_D, f_D2, size, 0);
    // CHECK: dpct::dpct_memcpy(f_D, f_D2, size, dpct::automatic);
    cuMemcpyDtoD(f_D, f_D2, size);

    // CHECK: dpct::dpct_memcpy(f_D, f_D2, size, dpct::automatic);
    cuMemcpy(f_D, f_D2, size);
    // CHECK: CALL(dpct::dpct_memcpy(f_D, f_D2, size, dpct::automatic));
    CALL(cuMemcpy(f_D, f_D2, size));
    // CHECK: r = DPCT_CHECK_ERROR(dpct::dpct_memcpy(f_D, f_D2, size, dpct::automatic));
    r = cuMemcpy(f_D, f_D2, size);

    // CHECK: dpct::async_dpct_memcpy(f_D, f_D2, size, dpct::automatic, *stream);
    cuMemcpyAsync(f_D, f_D2, size, stream);
    // CHECK: CALL(dpct::async_dpct_memcpy(f_D, f_D2, size, dpct::automatic, *stream));
    CALL(cuMemcpyAsync(f_D, f_D2, size, stream));
    // CHECK: r = DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(f_D, f_D2, size, dpct::automatic, *stream));
    r = cuMemcpyAsync(f_D, f_D2, size, stream);

    // CHECK: dpct::async_dpct_memcpy(f_D, f_D2, size, dpct::automatic);
    cuMemcpyAsync(f_D, f_D2, size, 0);
    // CHECK: CALL(dpct::async_dpct_memcpy(f_D, f_D2, size, dpct::automatic));
    CALL(cuMemcpyAsync(f_D, f_D2, size, 0));
    // CHECK: r = DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(f_D, f_D2, size, dpct::automatic));
    r = cuMemcpyAsync(f_D, f_D2, size, 0);

    // CHECK: dpct::memcpy_parameter cpy;
    // CHECK-NEXT: ;
    // CHECK-NEXT: cpy.to.pitched.set_data_ptr(f_A);
    // CHECK-NEXT: cpy.to.pitched.set_pitch(20);
    // CHECK-NEXT: cpy.to.pos[1] = 10;
    // CHECK-NEXT: cpy.to.pos_x_in_bytes = 15;
    // CHECK-EMPTY:
    // CHECK-NEXT: ;
    // CHECK-NEXT: cpy.from.pitched.set_data_ptr(f_D);
    // CHECK-NEXT: cpy.from.pitched.set_pitch(20);
    // CHECK-NEXT: cpy.from.pos[1] = 10;
    // CHECK-NEXT: cpy.from.pos_x_in_bytes = 15;
    // CHECK-EMPTY:
    // CHECK-NEXT: cpy.size_x_in_bytes = 4;
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

    // CHECK: dpct::dpct_memcpy(cpy);
    cuMemcpy2D(&cpy);
    // CHECK: dpct::async_dpct_memcpy(cpy, *stream);
    cuMemcpy2DAsync(&cpy, stream);

    CUdeviceptr devicePtr;
    size_t count = 32;
    // CHECK: int advise = 0;
    CUmem_advise advise = CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION;
    CUdevice cudevice =0;
    CUresult cu_err;

    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cuMemAdvise is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuMemAdvise(devicePtr, count, advise, cudevice);
    cuMemAdvise(devicePtr, count, advise, cudevice);

    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cuMemAdvise is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError(cuMemAdvise(devicePtr, count, advise, cudevice));
    cuCheckError(cuMemAdvise(devicePtr, count, advise, cudevice));

    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cuMemAdvise is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: cu_err = cuMemAdvise(devicePtr, count, advise, cudevice);
    cu_err = cuMemAdvise(devicePtr, count, advise, cudevice);

    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cuMemAdvise is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuMemAdvise(devicePtr, count, 0, cudevice);
    cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, cudevice);

    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cuMemAdvise is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError(cuMemAdvise(devicePtr, count, 0, cudevice));
    cuCheckError(cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, cudevice));

    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cuMemAdvise is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError(cuMemAdvise(devicePtr, count, (int)1, cudevice));
    cuCheckError(cuMemAdvise(devicePtr, count, (CUmem_advise)1, cudevice));

    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cuMemAdvise is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError(cuMemAdvise(devicePtr, count, int(1), cudevice));
    cuCheckError(cuMemAdvise(devicePtr, count, CUmem_advise(1), cudevice));

    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cuMemAdvise is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuCheckError(cuMemAdvise(devicePtr, count, static_cast<int>(1), cudevice));
    cuCheckError(cuMemAdvise(devicePtr, count, static_cast<CUmem_advise>(1), cudevice));

    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cuMemAdvise is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: cu_err = cuMemAdvise(devicePtr, count, 0, cudevice);
    cu_err = cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, cudevice);

    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of cuMemAdvise is not supported.
    // CHECK-NEXT: */
    // CHECK-NEXT: /*
    // CHECK-NEXT: DPCT1063:{{[0-9]+}}: Advice parameter is device-defined and was set to 0. You may need to adjust it.
    // CHECK-NEXT: */
    // CHECK-NEXT: cuMemAdvise(devicePtr, count, 0, cudevice);
    cuMemAdvise(devicePtr, count, CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION, cudevice);

    unsigned int v32 = 50000;
    unsigned short v16 = 20000;
    unsigned char v8 = (unsigned char) 200;
    //CHECK: dpct::dpct_memset_d32(f_D, v32, size);
    //CHECK-NEXT: dpct::dpct_memset_d16(f_D, v16, size * 2);
    //CHECK-NEXT: dpct::dpct_memset(f_D, v8, size * 4);
    //CHECK-NEXT: dpct::async_dpct_memset_d32(f_D, v32, size, *stream);
    //CHECK-NEXT: dpct::async_dpct_memset_d16(f_D, v16, size * 2, *stream);
    //CHECK-NEXT: dpct::async_dpct_memset(f_D, v8, size * 4, *stream);
    //CHECK-NEXT: dpct::dpct_memset_d32(f_D, 1, v32, 4, 6);
    //CHECK-NEXT: dpct::dpct_memset_d16(f_D, 1, v16, 4 * 2, 6);
    //CHECK-NEXT: dpct::dpct_memset(f_D, 1, v8, 4 * 4, 6);
    //CHECK-NEXT: dpct::async_dpct_memset_d32(f_D, 1, v32, 4, 6, *stream);
    //CHECK-NEXT: dpct::async_dpct_memset_d16(f_D, 1, v16, 4 * 2, 6, *stream);
    //CHECK-NEXT: dpct::async_dpct_memset(f_D, 1, v8, 4 * 4, 6, *stream);
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

    // CHECK: dpct::memcpy_parameter cpy2;
    // CHECK-EMPTY:
    // CHECK-NEXT: dpct::image_matrix_p ca;
    // CHECK-NEXT: ;
    // CHECK-NEXT: cpy2.to.image = ca;
    // CHECK-NEXT: cpy2.to.pitched.set_pitch(5);
    // CHECK-NEXT: cpy2.to.pitched.set_y(4);
    // CHECK-NEXT: cpy2.to.pos[1] = 3;
    // CHECK-NEXT: cpy2.to.pos[2] = 2;
    // CHECK-NEXT: cpy2.to.pos_x_in_bytes = 1;
    // CHECK-NEXT: ;
    // CHECK-EMPTY:
    // CHECK-NEXT: ;
    // CHECK-NEXT: cpy2.from.pitched.set_data_ptr(f_A);
    // CHECK-NEXT: cpy2.from.pitched.set_pitch(5);
    // CHECK-NEXT: cpy2.from.pitched.set_y(4);
    // CHECK-NEXT: cpy2.from.pos[1] = 3;
    // CHECK-NEXT: cpy2.from.pos[2] = 2;
    // CHECK-NEXT: cpy2.from.pos_x_in_bytes = 1;
    // CHECK-NEXT: ;
    // CHECK-EMPTY:
    // CHECK-NEXT: cpy2.size_x_in_bytes = 3;
    // CHECK-NEXT: cpy2.size[1] = 2;
    // CHECK-NEXT: cpy2.size[2] = 1;
    CUDA_MEMCPY3D cpy2;

    CUarray ca;
    cpy2.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    cpy2.dstArray = ca;
    cpy2.dstPitch = 5;
    cpy2.dstHeight = 4;
    cpy2.dstY = 3;
    cpy2.dstZ = 2;
    cpy2.dstXInBytes = 1;
    cpy2.dstLOD = 0;

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

    // CHECK: dpct::dpct_memcpy(cpy2);
    cuMemcpy3D(&cpy2);

    CUstream cs;
    // CHECK: dpct::async_dpct_memcpy(cpy2, *cs);
    cuMemcpy3DAsync(&cpy2, cs);

    // CHECK: dpct::dpct_free(f_D);
    cuMemFree(f_D);
    unsigned int flags;
    int host;
    float *h_A = (float *)malloc(100);

    // CHECK: flags = 0;
    cuMemHostGetFlags(&flags, &host);
    // CHECK: cuCheckError(DPCT_CHECK_ERROR(flags = 0));
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
