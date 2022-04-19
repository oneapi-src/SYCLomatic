// RUN: c2s --usm-level=none --format-range=none -out-root %T/driver-mem-usm-none %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/driver-mem-usm-none/driver-mem-usm-none.dp.cpp %s

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
int main(){
    size_t result1, result2;
    int size = 32;
    float* f_A;
    // CHECK: f_A = (float *)malloc(size);
    cuMemHostAlloc((void **)&f_A, size, CU_MEMHOSTALLOC_DEVICEMAP);


    // CHECK: void * f_D = 0;
    CUdeviceptr f_D = 0;
    // CHECK: void * f_D2 = 0;
    CUdeviceptr f_D2 = 0;
    // CHECK: f_D = c2s::c2s_malloc(size);
    cuMemAlloc(&f_D, size);

    // CHECK: sycl::queue * stream;
    CUstream stream;
    // CHECK: c2s::async_c2s_memcpy(f_D, f_A, size, c2s::automatic, *stream);
    cuMemcpyHtoDAsync(f_D, f_A, size, stream);
    // CHECK: c2s::async_c2s_memcpy(f_D, f_A, size, c2s::automatic);
    cuMemcpyHtoDAsync(f_D, f_A, size, 0);
    // CHECK: c2s::c2s_memcpy(f_D, f_A, size, c2s::automatic);
    cuMemcpyHtoD(f_D, f_A, size);

    // CHECK: c2s::async_c2s_memcpy(f_A, f_D, size, c2s::automatic, *stream);
    cuMemcpyDtoHAsync(f_A, f_D, size, stream);
    // CHECK: c2s::async_c2s_memcpy(f_A, f_D, size, c2s::automatic);
    cuMemcpyDtoHAsync(f_A, f_D, size, 0);
    // CHECK: c2s::c2s_memcpy(f_A, f_D, size, c2s::automatic);
    cuMemcpyDtoH(f_A, f_D, size);

    // CHECK: c2s::async_c2s_memcpy(f_D, f_D2, size, c2s::automatic, *stream);
    cuMemcpyDtoDAsync(f_D, f_D2, size, stream);
    // CHECK: c2s::async_c2s_memcpy(f_D, f_D2, size, c2s::automatic);
    cuMemcpyDtoDAsync(f_D, f_D2, size, 0);
    // CHECK: c2s::c2s_memcpy(f_D, f_D2, size, c2s::automatic);
    cuMemcpyDtoD(f_D, f_D2, size);

    // CHECK: c2s::pitched_data cpy_from_data_ct1, cpy_to_data_ct1;
    // CHECK: sycl::id<3> cpy_from_pos_ct1(0, 0, 0), cpy_to_pos_ct1(0, 0, 0);
    // CHECK: sycl::range<3> cpy_size_ct1(1, 1, 1);
    CUDA_MEMCPY2D cpy;
    //
    cpy.dstMemoryType = CU_MEMORYTYPE_HOST;
    // CHECK: cpy_to_data_ct1.set_data_ptr(f_A);
    cpy.dstHost = f_A;
    // CHECK: cpy_to_data_ct1.set_pitch(20);
    cpy.dstPitch = 20;
    // CHECK: cpy_to_pos_ct1[1] = 10;
    cpy.dstY = 10;
    // CHECK: cpy_to_pos_ct1[0] = 15;
    cpy.dstXInBytes = 15;

    //
    cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    // CHECK: cpy_from_data_ct1.set_data_ptr(f_D);
    cpy.srcDevice = f_D;
    // CHECK: cpy_from_data_ct1.set_pitch(20);
    cpy.srcPitch = 20;
    // CHECK: cpy_from_pos_ct1[1] = 10;
    cpy.srcY = 10;
    // CHECK: cpy_from_pos_ct1[0] = 15;
    cpy.srcXInBytes = 15;

    // CHECK: cpy_size_ct1[0] = 4;
    cpy.WidthInBytes = 4;
    // CHECK: cpy_size_ct1[1] = 7;
    cpy.Height = 7;

    // CHECK: c2s::c2s_memcpy(cpy_to_data_ct1, cpy_to_pos_ct1, cpy_from_data_ct1, cpy_from_pos_ct1, cpy_size_ct1);
    cuMemcpy2D(&cpy);
    // CHECK: c2s::async_c2s_memcpy(cpy_to_data_ct1, cpy_to_pos_ct1, cpy_from_data_ct1, cpy_from_pos_ct1, cpy_size_ct1, c2s::automatic, *stream);
    cuMemcpy2DAsync(&cpy, stream);

    // CHECK: c2s::pitched_data cpy2_from_data_ct1, cpy2_to_data_ct1;
    // CHECK: sycl::id<3> cpy2_from_pos_ct1(0, 0, 0), cpy2_to_pos_ct1(0, 0, 0);
    // CHECK: sycl::range<3> cpy2_size_ct1(1, 1, 1);
    CUDA_MEMCPY3D cpy2;

    CUarray ca;
    //
    cpy2.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    // CHECK: cpy2_to_data_ct1 = ca->to_pitched_data();
    cpy2.dstArray = ca;
    // CHECK: cpy2_to_data_ct1.set_pitch(5);
    cpy2.dstPitch = 5;
    // CHECK: cpy2_to_data_ct1.set_y(4);
    cpy2.dstHeight = 4;
    // CHECK: cpy2_to_pos_ct1[1] = 3;
    cpy2.dstY = 3;
    // CHECK: cpy2_to_pos_ct1[2] = 2;
    cpy2.dstZ = 2;
    // CHECK: cpy2_to_pos_ct1[0] = 1;
    cpy2.dstXInBytes = 1;
    //
    cpy2.dstLOD = 0;

    //
    cpy2.srcMemoryType = CU_MEMORYTYPE_HOST;
    // CHECK: cpy2_from_data_ct1.set_data_ptr(f_A);
    cpy2.srcHost = f_A;
    // CHECK: cpy2_from_data_ct1.set_pitch(5);
    cpy2.srcPitch = 5;
    // CHECK: cpy2_from_data_ct1.set_y(4);
    cpy2.srcHeight = 4;
    // CHECK: cpy2_from_pos_ct1[1] = 3;
    cpy2.srcY = 3;
    // CHECK: cpy2_from_pos_ct1[2] = 2;
    cpy2.srcZ = 2;
    // CHECK: cpy2_from_pos_ct1[0] = 1;
    cpy2.srcXInBytes = 1;
    //
    cpy2.srcLOD = 0;

    // CHECK: cpy2_size_ct1[0] = 3;
    cpy2.WidthInBytes = 3;
    // CHECK: cpy2_size_ct1[1] = 2;
    cpy2.Height = 2;
    // CHECK: cpy2_size_ct1[2] = 1;
    cpy2.Depth = 1;

    // CHECK: c2s::c2s_memcpy(cpy2_to_data_ct1, cpy2_to_pos_ct1, cpy2_from_data_ct1, cpy2_from_pos_ct1, cpy2_size_ct1);
    cuMemcpy3D(&cpy2);

    return 0;
}
