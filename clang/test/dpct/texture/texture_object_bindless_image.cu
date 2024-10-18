// RUN: dpct --format-range=none --use-experimental-features=bindless_images -out-root %T/texture/texture_object_bindless_image %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/texture/texture_object_bindless_image/texture_object_bindless_image.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/texture/texture_object_bindless_image/texture_object_bindless_image.dp.cpp -o %T/texture/texture_object_bindless_image/texture_object_bindless_image.dp.o %}

#include "cuda.h"

// CHECK: template <typename T> void kernel(sycl::ext::oneapi::experimental::sampled_image_handle tex) {
template <typename T> __global__ void kernel(cudaTextureObject_t tex) {
  int i;
  float j, k, l, m;
  // CHECK: sycl::ext::oneapi::experimental::sample_image<T>(tex, float(i));
  tex1Dfetch<T>(tex, i);
  // CHECK: sycl::ext::oneapi::experimental::sample_image<sycl::short2>(tex, float(i));
  tex1D<short2>(tex, i);
  // CHECK: i = sycl::ext::oneapi::experimental::sample_image<int>(tex, float(i));
  tex1D(&i, tex, i);
  // CHECK: sycl::ext::oneapi::experimental::sample_image<T>(tex, sycl::float2(j, k));
  tex2D<T>(tex, j, k);
  // CHECK: i = sycl::ext::oneapi::experimental::sample_image<int>(tex, sycl::float2(j, k));
  tex2D(&i, tex, j, k);
  // CHECK: sycl::ext::oneapi::experimental::sample_image<sycl::short2>(tex, sycl::float3(j, k, l));
  tex3D<short2>(tex, j, k, l);
  // CHECK: i = sycl::ext::oneapi::experimental::sample_image<int>(tex, sycl::float3(j, k, l));
  tex3D(&i, tex, j, k, l);
  // CHECK: sycl::ext::oneapi::experimental::sample_mipmap<sycl::short2>(tex, float(j), l);
  tex1DLod<short2>(tex, j, l);
  // CHECK: i = sycl::ext::oneapi::experimental::sample_mipmap<int>(tex, float(j), l);
  tex1DLod(&i, tex, j, l);
  // CHECK: sycl::ext::oneapi::experimental::sample_mipmap<sycl::short2>(tex, sycl::float2(j, k), l);
  tex2DLod<short2>(tex, j, k, l);
  // CHECK: i = sycl::ext::oneapi::experimental::sample_mipmap<int>(tex, sycl::float2(j, k), l);
  tex2DLod(&i, tex, j, k, l);
  // CHECK: sycl::ext::oneapi::experimental::sample_mipmap<T>(tex, sycl::float3(j, k, m), l);
  tex3DLod<T>(tex, j, k, m, l);
  // CHECK: i = sycl::ext::oneapi::experimental::sample_mipmap<int>(tex, sycl::float3(j, k, m), l);
  tex3DLod(&i, tex, j, k, m, l);
  // CHECK: sycl::ext::oneapi::experimental::sample_image_array<sycl::short2>(tex, float(j), i);
  tex1DLayered<short2>(tex, j, i);
  // CHECK: k = sycl::ext::oneapi::experimental::sample_image_array<float>(tex, float(j), i);
  tex1DLayered(&k, tex, j, i);
  // CHECK: sycl::ext::oneapi::experimental::sample_image_array<T>(tex, sycl::float2(j, k), i);
  tex2DLayered<T>(tex, j, k, i);
  // CHECK: l = sycl::ext::oneapi::experimental::sample_image_array<float>(tex, sycl::float2(j, k), i);
  tex2DLayered(&l, tex, j, k, i);
#ifndef NO_BUILD_TEST
  T t;
  // CHECK: t = sycl::ext::oneapi::experimental::sample_image<dpct_placeholder/*Fix the type mannually*/>(tex, float(i));
  tex1Dfetch(&t, tex, i);
  // CHECK: t = sycl::ext::oneapi::experimental::sample_image<dpct_placeholder/*Fix the type mannually*/>(tex, float(i));
  tex1D(&t, tex, i);
  // CHECK: t = sycl::ext::oneapi::experimental::sample_image<dpct_placeholder/*Fix the type mannually*/>(tex, sycl::float2(i, j));
  tex2D(&t, tex, i, j);
  // CHECK: t = sycl::ext::oneapi::experimental::sample_image<dpct_placeholder/*Fix the type mannually*/>(tex, sycl::float3(i, j, k));
  tex3D(&t, tex, i, j, k);
  // CHECK: t = sycl::ext::oneapi::experimental::sample_mipmap<dpct_placeholder/*Fix the type mannually*/>(tex, float(i), l);
  tex1DLod(&t, tex, i, l);
  // CHECK: t = sycl::ext::oneapi::experimental::sample_mipmap<dpct_placeholder/*Fix the type mannually*/>(tex, sycl::float2(i, j), l);
  tex2DLod(&t, tex, i, j, l);
  // CHECK: t = sycl::ext::oneapi::experimental::sample_mipmap<dpct_placeholder/*Fix the type mannually*/>(tex, sycl::float3(i, j, k), l);
  tex3DLod(&t, tex, i, j, k, l);
#endif
}

void driverMemoryManagement() {
  size_t s, s1, s2;
  unsigned u;
  void *pV;
  // CHECK: sycl::image_channel_type f;
  CUarray_format f;
  // CHECK: dpct::experimental::image_mem_wrapper_ptr *pArr;
  CUarray *pArr;
  // CHECK: dpct::device_ptr pD;
  CUdeviceptr pD;
  // CHECK: dpct::queue_ptr st;
  CUstream st;
  // CHECK: sycl::ext::oneapi::experimental::image_descriptor p3DDesc;
  CUDA_ARRAY3D_DESCRIPTOR p3DDesc;
  // CHECK: p3DDesc.width = s;
  p3DDesc.Width = s;
  // CHECK: p3DDesc.height = s;
  p3DDesc.Height = s;
  // CHECK: p3DDesc.depth = s;
  p3DDesc.Depth = s;
  // CHECK: p3DDesc.channel_type = f;
  p3DDesc.Format = f;
  // CHECK: p3DDesc.num_channels = u;
  p3DDesc.NumChannels = u;
  p3DDesc.Flags = 1;
  // CHECK: sycl::ext::oneapi::experimental::image_descriptor pDesc;
  CUDA_ARRAY_DESCRIPTOR pDesc;
  // CHECK: pDesc.channel_type = f;
  pDesc.Format = f;
  // CHECK: pDesc.height = s;
  pDesc.Height = s;
  // CHECK: pDesc.num_channels = u;
  pDesc.NumChannels = u;
  // CHECK: pDesc.width = s;
  pDesc.Width = s;
  // CHECK: dpct::memcpy_parameter p2d;
  CUDA_MEMCPY2D p2d;
  // CHECK: p2d.from.pos_x_in_bytes = s;
  p2d.srcXInBytes = s;
  // CHECK: p2d.from.pos[1] = s;
  p2d.srcY = s;
  // CHECK: ;
  p2d.srcMemoryType;
  // CHECK: p2d.from.pitched.set_data_ptr(pV);
  p2d.srcHost = pV;
  // CHECK: p2d.from.pitched.set_data_ptr(pD);
  p2d.srcDevice = pD;
  // CHECK: p2d.from.image_bindless = *pArr;
  p2d.srcArray = *pArr;
  // CHECK: p2d.from.pitched.set_pitch(s);
  p2d.srcPitch = s;
  // CHECK: p2d.to.pos_x_in_bytes = s;
  p2d.dstXInBytes = s;
  // CHECK: p2d.to.pos[1] = s;
  p2d.dstY = s;
  // CHECK: ;
  p2d.dstMemoryType;
  // CHECK: p2d.to.pitched.set_data_ptr(pV);
  p2d.dstHost = pV;
  // CHECK: p2d.to.pitched.set_data_ptr(pD);
  p2d.dstDevice = pD;
  // CHECK: p2d.to.image_bindless = *pArr;
  p2d.dstArray = *pArr;
  // CHECK: p2d.to.pitched.set_pitch(s);
  p2d.dstPitch = s;
  // CHECK: p2d.size_x_in_bytes = s;
  p2d.WidthInBytes = s;
  // CHECK: p2d.size[1] = s;
  p2d.Height = s;
  // CHECK: dpct::memcpy_parameter p3d;
  CUDA_MEMCPY3D p3d;
  // CHECK: p3d.from.pos_x_in_bytes = s;
  p3d.srcXInBytes = s;
  // CHECK: p3d.from.pos[1] = s;
  p3d.srcY = s;
  // CHECK: p3d.from.pos[2] = s;
  p3d.srcZ = s;
  // CHECK: ;
  p3d.srcLOD;
  // CHECK: ;
  p3d.srcMemoryType;
  // CHECK: p3d.from.pitched.set_data_ptr(pV);
  p3d.srcHost = pV;
  // CHECK: p3d.from.pitched.set_data_ptr(pD);
  p3d.srcDevice = pD;
  // CHECK: p3d.from.image_bindless = *pArr;
  p3d.srcArray = *pArr;
  // CHECK: p3d.from.pitched.set_pitch(s);
  p3d.srcPitch = s;
  // CHECK: p3d.from.pitched.set_y(s);
  p3d.srcHeight = s;
  // CHECK: p3d.to.pos_x_in_bytes = s;
  p3d.dstXInBytes = s;
  // CHECK: p3d.to.pos[1] = s;
  p3d.dstY = s;
  // CHECK: p3d.to.pos[2] = s;
  p3d.dstZ = s;
  // CHECK: ;
  p3d.dstLOD;
  // CHECK: ;
  p3d.dstMemoryType;
  // CHECK: p3d.to.pitched.set_data_ptr(pV);
  p3d.dstHost = pV;
  // CHECK: p3d.to.pitched.set_data_ptr(pD);
  p3d.dstDevice = pD;
  // CHECK: p3d.to.image_bindless = *pArr;
  p3d.dstArray = *pArr;
  // CHECK: p3d.to.pitched.set_pitch(s);
  p3d.dstPitch = s;
  // CHECK: p3d.to.pitched.set_y(s);
  p3d.dstHeight = s;
  // CHECK: p3d.size_x_in_bytes = s;
  p3d.WidthInBytes = s;
  // CHECK: p3d.size[1] = s;
  p3d.Height = s;
  // CHECK: p3d.size[2] = s;
  p3d.Depth = s;
  // CHECK: *pArr = new dpct::experimental::image_mem_wrapper(&p3DDesc);
  cuArray3DCreate(pArr, &p3DDesc);
  // CHECK: *pArr = new dpct::experimental::image_mem_wrapper(&pDesc);
  cuArrayCreate(pArr, &pDesc);
  // CHECK: delete (*pArr);
  cuArrayDestroy(*pArr);
  // CHECK: dpct::dpct_memcpy(p2d);
  cuMemcpy2D(&p2d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1124:{{[0-9]+}}: cuMemcpy2DAsync_v2 is migrated to asynchronous memcpy API. While the origin API might be synchronous, it depends on the type of operand memory, so you may need to call wait() on event return by memcpy API to ensure synchronization behavior.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::async_dpct_memcpy(p2d, *st);
  cuMemcpy2DAsync(&p2d, st);
  // CHECK: dpct::dpct_memcpy(p3d);
  cuMemcpy3D(&p3d);
  // CHECK: /*
  // CHECK-NEXT: DPCT1124:{{[0-9]+}}: cuMemcpy3DAsync_v2 is migrated to asynchronous memcpy API. While the origin API might be synchronous, it depends on the type of operand memory, so you may need to call wait() on event return by memcpy API to ensure synchronization behavior.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::async_dpct_memcpy(p3d, *st);
  cuMemcpy3DAsync(&p3d, st);
  // CHECK: dpct::experimental::dpct_memcpy(*pArr, s, 0, *pArr, s1, 0, s2);
  cuMemcpyAtoA(*pArr, s, *pArr, s1, s2);
  // CHECK: dpct::experimental::dpct_memcpy(pD, *pArr, s, 0, s1);
  cuMemcpyAtoD(pD, *pArr, s, s1);
  // CHECK: dpct::experimental::dpct_memcpy(pV, *pArr, s, 0, s1);
  cuMemcpyAtoH(pV, *pArr, s, s1);
  // CHECK: dpct::experimental::async_dpct_memcpy(pV, *pArr, s, 0, s1, *st);
  cuMemcpyAtoHAsync(pV, *pArr, s, s1, st);
  // CHECK: dpct::experimental::dpct_memcpy(*pArr, s, 0, pD, s1);
  cuMemcpyDtoA(*pArr, s, pD, s1);
  // CHECK: q_ct1.memcpy(pD, pD, s).wait();
  cuMemcpyDtoD(pD, pD, s);
  // CHECK: st->memcpy(pD, pD, s);
  cuMemcpyDtoDAsync(pD, pD, s, st);
  // CHECK: q_ct1.memcpy(pV, pD, s).wait();
  cuMemcpyDtoH(pV, pD, s);
  // CHECK: /*
  // CHECK-NEXT: DPCT1124:{{[0-9]+}}: cuMemcpyDtoHAsync_v2 is migrated to asynchronous memcpy API. While the origin API might be synchronous, it depends on the type of operand memory, so you may need to call wait() on event return by memcpy API to ensure synchronization behavior.
  // CHECK-NEXT: */
  // CHECK-NEXT: st->memcpy(pV, pD, s);
  cuMemcpyDtoHAsync(pV, pD, s, st);
  // CHECK: dpct::experimental::dpct_memcpy(*pArr, s, 0, pV, s1);
  cuMemcpyHtoA(*pArr, s, pV, s1);
  // CHECK: dpct::experimental::async_dpct_memcpy(*pArr, s, 0, pV, s1, *st);
  cuMemcpyHtoAAsync(*pArr, s, pV, s1, st);
  // CHECK: q_ct1.memcpy(pD, pV, s).wait();
  cuMemcpyHtoD(pD, pV, s);
  // CHECK: st->memcpy(pD, pV, s);
  cuMemcpyHtoDAsync(pD, pV, s, st);
}

void driver() {
  // CHECK: dpct::queue_ptr s;
  CUstream s;
  const int data[1] = {0};
  const void *cv;
  void *v;
  // CHECK: dpct::device_ptr dp;
  CUdeviceptr dp;
  // CHECK: dpct::experimental::image_mem_wrapper_ptr a;
  CUarray a;
  // CHECK: int c;
  CUcontext c;
  // CHECK: dpct::memcpy_parameter p3dp;
  CUDA_MEMCPY3D_PEER p3dp;
  // CHECK: p3dp.from.pos_x_in_bytes = 1;
  p3dp.srcXInBytes = 1;
  // CHECK: p3dp.from.pos[1] = 2;
  p3dp.srcY = 2;
  // CHECK: p3dp.from.pos[2] = 3;
  p3dp.srcZ = 3;
  p3dp.srcLOD = 4;
  p3dp.srcMemoryType = CU_MEMORYTYPE_HOST;
  // CHECK: p3dp.from.pitched.set_data_ptr(data);
  p3dp.srcHost = data;
  p3dp.srcHost = cv;
  // CHECK: p3dp.from.pitched.set_data_ptr(dp);
  p3dp.srcDevice = dp;
  // CHECK: p3dp.from.image_bindless = a;
  p3dp.srcArray = a;
  // CHECK: p3dp.from.dev_id = c;
  p3dp.srcContext = c;
  // CHECK: p3dp.from.pitched.set_pitch(5);
  p3dp.srcPitch = 5;
  // CHECK: p3dp.from.pitched.set_y(6);
  p3dp.srcHeight = 6;

  // CHECK: p3dp.to.pos_x_in_bytes = 1;
  p3dp.dstXInBytes = 1;
  // CHECK: p3dp.to.pos[1] = 2;
  p3dp.dstY = 2;
  // CHECK: p3dp.to.pos[2] = 3;
  p3dp.dstZ = 3;
  p3dp.dstLOD = 4;
  p3dp.dstMemoryType = CU_MEMORYTYPE_HOST;
  // CHECK: p3dp.to.pitched.set_data_ptr(v);
  p3dp.dstHost = v;
  // CHECK: p3dp.to.pitched.set_data_ptr(dp);
  p3dp.dstDevice = dp;
  // CHECK: p3dp.to.image_bindless = a;
  p3dp.dstArray = a;
  // CHECK: p3dp.to.dev_id = c;
  p3dp.dstContext = c;
  // CHECK: p3dp.to.pitched.set_pitch(5);
  p3dp.dstPitch = 5;
  // CHECK: p3dp.to.pitched.set_y(6);
  p3dp.dstHeight = 6;

  // CHECK: p3dp.size_x_in_bytes = 3;
  p3dp.WidthInBytes = 3;
  // CHECK: p3dp.size[1] = 2;
  p3dp.Height = 2;
  // CHECK: p3dp.size[2] = 1;
  p3dp.Depth = 1;
  // CHECK: dpct::dpct_memcpy(p3dp);
  cuMemcpy3DPeer(&p3dp);
  // CHECK: /*
  // CHECK-NEXT: DPCT1124:{{[0-9]+}}: cuMemcpy3DPeerAsync is migrated to asynchronous memcpy API. While the origin API might be synchronous, it depends on the type of operand memory, so you may need to call wait() on event return by memcpy API to ensure synchronization behavior.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::async_dpct_memcpy(p3dp, *s);
  cuMemcpy3DPeerAsync(&p3dp, s);

  // CHECK: sycl::ext::oneapi::experimental::sampled_image_handle o;
  CUtexObject o;
  // CHECK: dpct::image_data R;
  CUDA_RESOURCE_DESC R;
  // CHECK: dpct::sampling_info T;
  CUDA_TEXTURE_DESC T;
  // CHECK: o = dpct::experimental::create_bindless_image(R, T);
  cuTexObjectCreate(&o, &R, &T, NULL);
  // CHECK: dpct::experimental::destroy_bindless_image(o, dpct::get_in_order_queue());
  cuTexObjectDestroy(o);
  // CHECK: R = dpct::experimental::get_data(o);
  cuTexObjectGetResourceDesc(&R, o);
  // CHECK: T = dpct::experimental::get_sampling_info(o);
  cuTexObjectGetTextureDesc(&T, o);
}

int main() {
  const void *input;
  void *output;
  size_t w, h, sizeInBytes, w_offest_src, h_offest_src, w_offest_dest, h_offest_dest;
  unsigned int flag, l;
  cudaExtent e;
  cudaStream_t s;
  // CHECK: dpct::experimental::image_mem_wrapper_ptr pArr;
  cudaArray_t pArr;
  // CHECK: const dpct::experimental::image_mem_wrapper *pArr_src;
  const cudaArray *pArr_src;
  // CHECK: dpct::experimental::image_mem_wrapper_ptr pMipMapArr;
  cudaMipmappedArray_t pMipMapArr;
  // CHECK: dpct::image_channel desc;
  cudaChannelFormatDesc desc;
  // CHECK: pArr = new dpct::experimental::image_mem_wrapper(desc, e, sycl::ext::oneapi::experimental::image_type::standard);
  cudaMalloc3DArray(&pArr, &desc, e);
  // CHECK: pArr = new dpct::experimental::image_mem_wrapper(desc, e, sycl::ext::oneapi::experimental::image_type::array);
  cudaMalloc3DArray(&pArr, &desc, e, cudaArrayLayered);
  // CHECK: pArr = new dpct::experimental::image_mem_wrapper(desc, e, sycl::ext::oneapi::experimental::image_type::array);
  cudaMalloc3DArray(&pArr, &desc, e, 1);
  // CHECK: pArr = new dpct::experimental::image_mem_wrapper(desc, w, h);
  cudaMallocArray(&pArr, &desc, w, h);
  // CHECK: pArr = new dpct::experimental::image_mem_wrapper(desc, 1, 0.1);
  cudaMallocArray(&pArr, &desc, 1, 0.1);
  // CHECK: pArr = new dpct::experimental::image_mem_wrapper(desc, l);
  cudaMallocArray(&pArr, &desc, l);
  // CHECK: pMipMapArr = new dpct::experimental::image_mem_wrapper(desc, e, sycl::ext::oneapi::experimental::image_type::mipmap, l);
  cudaMallocMipmappedArray(&pMipMapArr, &desc, e, l, flag);
  // CHECK: pMipMapArr = new dpct::experimental::image_mem_wrapper(desc, sycl::range<3>(), sycl::ext::oneapi::experimental::image_type::mipmap, 2);
  cudaMallocMipmappedArray(&pMipMapArr, &desc, cudaExtent(), 2);
  // CHECK: pArr = pMipMapArr->get_mip_level(0);
  cudaGetMipmappedArrayLevel(&pArr, pMipMapArr, 0);
  // CHECK: desc = pArr->get_channel();
  // CHECK-NEXT: e = pArr->get_range();
  // CHECK-NEXT: flag = 0;
  cudaArrayGetInfo(&desc, &e, &flag, pArr);
  // CHECK: dpct::experimental::dpct_memcpy(pArr, w_offest_dest, h_offest_dest, pArr_src,
  // CHECK-NEXT:                          w_offest_src, h_offest_src, w, h);
  cudaMemcpy2DArrayToArray(pArr, w_offest_dest, h_offest_dest, pArr_src,
                           w_offest_src, h_offest_src, w, h,
                           cudaMemcpyHostToDevice);
  // CHECK: dpct::experimental::dpct_memcpy(pArr, w_offest_dest, h_offest_dest, pArr_src,
  // CHECK-NEXT:                          w_offest_src, h_offest_src, w, h);
  cudaMemcpy2DArrayToArray(pArr, w_offest_dest, h_offest_dest, pArr_src,
                           w_offest_src, h_offest_src, w, h);
  // CHECK: dpct::experimental::dpct_memcpy(output, w, pArr_src, w_offest_src, h_offest_src, w, h);
  cudaMemcpy2DFromArray(output, w, pArr_src, w_offest_src, h_offest_src, w, h,
                        cudaMemcpyHostToDevice);
  // CHECK: dpct::experimental::async_dpct_memcpy(output, w, pArr_src, w_offest_src, h_offest_src, w,
  // CHECK-NEXT:                            h);
  cudaMemcpy2DFromArrayAsync(output, w, pArr_src, w_offest_src, h_offest_src, w,
                             h, cudaMemcpyHostToDevice);
  // CHECK: dpct::experimental::async_dpct_memcpy(output, w, pArr_src, w_offest_src, h_offest_src, w,
  // CHECK-NEXT:                            h, *s);
  cudaMemcpy2DFromArrayAsync(output, w, pArr_src, w_offest_src, h_offest_src, w,
                             h, cudaMemcpyHostToDevice, s);
  // CHECK: dpct::experimental::dpct_memcpy(pArr, w_offest_dest, h_offest_dest, input, w, w, h);
  cudaMemcpy2DToArray(pArr, w_offest_dest, h_offest_dest, input, w, w, h,
                      cudaMemcpyHostToDevice);
  // CHECK:  dpct::experimental::async_dpct_memcpy(pArr, w_offest_dest, h_offest_dest, input, w, w, h);
  cudaMemcpy2DToArrayAsync(pArr, w_offest_dest, h_offest_dest, input, w, w, h,
                           cudaMemcpyHostToDevice);
  // CHECK:  dpct::experimental::async_dpct_memcpy(pArr, w_offest_dest, h_offest_dest, input, w, w, h, *s);
  cudaMemcpy2DToArrayAsync(pArr, w_offest_dest, h_offest_dest, input, w, w, h,
                           cudaMemcpyHostToDevice, s);
  // CHECK: dpct::experimental::dpct_memcpy(pArr, w_offest_dest, h_offest_dest, pArr_src,
  //                        w_offest_src, h_offest_src, w * h);
  cudaMemcpyArrayToArray(pArr, w_offest_dest, h_offest_dest, pArr_src,
                         w_offest_src, h_offest_src, w * h,
                         cudaMemcpyHostToDevice);
  // CHECK: dpct::experimental::dpct_memcpy(pArr, w_offest_dest, h_offest_dest, pArr_src,
  //                        w_offest_src, h_offest_src, w * h);
  cudaMemcpyArrayToArray(pArr, w_offest_dest, h_offest_dest, pArr_src,
                         w_offest_src, h_offest_src, w * h);
  // CHECK: dpct::experimental::dpct_memcpy(output, pArr_src, w_offest_src, h_offest_src, w * h);
  cudaMemcpyFromArray(output, pArr_src, w_offest_src, h_offest_src, w * h,
                      cudaMemcpyHostToDevice);
  // CHECK: dpct::experimental::async_dpct_memcpy(output, pArr_src, w_offest_src, h_offest_src, w * h);
  cudaMemcpyFromArrayAsync(output, pArr_src, w_offest_src, h_offest_src, w * h,
                           cudaMemcpyHostToDevice);
  // CHECK: dpct::experimental::async_dpct_memcpy(output, pArr_src, w_offest_src, h_offest_src, w * h, *s);
  cudaMemcpyFromArrayAsync(output, pArr_src, w_offest_src, h_offest_src, w * h,
                           cudaMemcpyHostToDevice, s);
  // CHECK: dpct::experimental::dpct_memcpy(pArr, w_offest_dest, h_offest_dest, input, w * h);
  cudaMemcpyToArray(pArr, w_offest_dest, h_offest_dest, input, w * h,
                    cudaMemcpyHostToDevice);
  // CHECK: dpct::experimental::async_dpct_memcpy(pArr, w_offest_dest, h_offest_dest, input, w * h);
  cudaMemcpyToArrayAsync(pArr, w_offest_dest, h_offest_dest, input, w * h,
                         cudaMemcpyHostToDevice);
  // CHECK: dpct::experimental::async_dpct_memcpy(pArr, w_offest_dest, h_offest_dest, input, w * h, *s);
  cudaMemcpyToArrayAsync(pArr, w_offest_dest, h_offest_dest, input, w * h,
                         cudaMemcpyHostToDevice, s);

  // CHECK: dpct::memcpy_parameter p3d;
  cudaMemcpy3DParms p3d;
  // CHECK: sycl::id<3> pos{0, 0, 0};
  cudaPos pos;
  // CHECK: dpct::pitched_data pp;
  cudaPitchedPtr pp;
  // CHECK: dpct::memcpy_direction k;
  cudaMemcpyKind k;
  // CHECK: p3d.from.image_bindless = pArr;
  p3d.srcArray = pArr;
  // CHECK: pArr_src = p3d.from.image_bindless;
  pArr_src = p3d.srcArray;
  // CHECK: p3d.from.pos = pos;
  p3d.srcPos = pos;
  // CHECK: p3d.from.pos[2] = 1;
  p3d.srcPos.x = 1;
  // CHECK: p3d.from.pos[1] = 2;
  p3d.srcPos.y = 2;
  // CHECK: p3d.from.pos[0] = 3;
  p3d.srcPos.z = 3;
  // CHECK: pos = p3d.from.pos;
  pos = p3d.srcPos;
  // CHECK: p3d.from.pitched = pp;
  p3d.srcPtr = pp;
  // CHECK: p3d.from.pitched.set_data_ptr(output);
  p3d.srcPtr.ptr = output;
  // CHECK: p3d.from.pitched.set_pitch(1);
  p3d.srcPtr.pitch = 1;
  // CHECK: p3d.from.pitched.set_x(2);
  p3d.srcPtr.xsize = 2;
  // CHECK: p3d.from.pitched.set_y(3);
  p3d.srcPtr.ysize = 3;
  // CHECK: pp = p3d.from.pitched;
  pp = p3d.srcPtr;
  // CHECK: p3d.to.image_bindless = pArr;
  p3d.dstArray = pArr;
  // CHECK: pArr = p3d.to.image_bindless;
  pArr = p3d.dstArray;
  // CHECK: p3d.to.pos = pos;
  p3d.dstPos = pos;
  // CHECK: p3d.to.pos[2] = 1;
  p3d.dstPos.x = 1;
  // CHECK: p3d.to.pos[1] = 2;
  p3d.dstPos.y = 2;
  // CHECK: p3d.to.pos[0] = 3;
  p3d.dstPos.z = 3;
  // CHECK: pos = p3d.to.pos;
  pos = p3d.dstPos;
  // CHECK: p3d.to.pitched = pp;
  p3d.dstPtr = pp;
  // CHECK: p3d.to.pitched.set_data_ptr(output);
  p3d.dstPtr.ptr = output;
  // CHECK: p3d.to.pitched.set_pitch(1);
  p3d.dstPtr.pitch = 1;
  // CHECK: p3d.to.pitched.set_x(2);
  p3d.dstPtr.xsize = 2;
  // CHECK: p3d.to.pitched.set_y(3);
  p3d.dstPtr.ysize = 3;
  // CHECK: pp = p3d.to.pitched;
  pp = p3d.dstPtr;
  // CHECK: p3d.size = e;
  p3d.extent = e;
  // CHECK: p3d.size[0] = 1;
  p3d.extent.width = 1;
  // CHECK: p3d.size[1] = 2;
  p3d.extent.height = 2;
  // CHECK: p3d.size[2] = 3;
  p3d.extent.depth = 3;
  // CHECK: e = p3d.size;
  e = p3d.extent;
  // CHECK: p3d.direction = k;
  p3d.kind = k;
  // CHECK: k = p3d.direction;
  k = p3d.kind;
  // CHECK: dpct::dpct_memcpy(p3d);
  cudaMemcpy3D(&p3d);
  // CHECK: dpct::async_dpct_memcpy(p3d);
  cudaMemcpy3DAsync(&p3d);
  // CHECK: dpct::async_dpct_memcpy(p3d, *s);
  cudaMemcpy3DAsync(&p3d, s);

  // CHECK: dpct::memcpy_parameter p3dp = {};
  cudaMemcpy3DPeerParms p3dp = {0};
  int d;
  // CHECK: p3dp.from.image_bindless = pArr;
  p3dp.srcArray = pArr;
  // CHECK: pArr_src = p3dp.from.image_bindless;
  pArr_src = p3dp.srcArray;
  // CHECK: p3dp.from.pos = pos;
  p3dp.srcPos = pos;
  // CHECK: pos = p3dp.from.pos;
  pos = p3dp.srcPos;
  // CHECK: p3dp.from.pitched = pp;
  p3dp.srcPtr = pp;
  // CHECK: pp = p3dp.from.pitched;
  pp = p3dp.srcPtr;
  // CHECK: p3dp.from.dev_id = d;
  p3dp.srcDevice = d;
  // CHECK: d = p3dp.from.dev_id;
  d = p3dp.srcDevice;
  // CHECK: p3dp.to.image_bindless = pArr;
  p3dp.dstArray = pArr;
  // CHECK: pArr = p3dp.to.image_bindless;
  pArr = p3dp.dstArray;
  // CHECK: p3dp.to.pos = pos;
  p3dp.dstPos = pos;
  // CHECK: pos = p3dp.to.pos;
  pos = p3dp.dstPos;
  // CHECK: p3dp.to.pitched = pp;
  p3dp.dstPtr = pp;
  // CHECK: pp = p3dp.to.pitched;
  pp = p3dp.dstPtr;
  // CHECK: p3dp.to.dev_id = d;
  p3dp.dstDevice = d;
  // CHECK: d = p3dp.to.dev_id;
  d = p3dp.dstDevice;
  // CHECK: p3dp.size = e;
  p3dp.extent = e;
  // CHECK: e = p3dp.size;
  e = p3dp.extent;
  // CHECK: dpct::dpct_memcpy(p3dp);
  cudaMemcpy3DPeer(&p3dp);
  // CHECK: dpct::async_dpct_memcpy(p3dp);
  cudaMemcpy3DPeerAsync(&p3dp);

  // CHECK: dpct::image_data resDesc0, resDesc1, resDesc2, resDesc3, resDesc4;
  cudaResourceDesc resDesc0, resDesc1, resDesc2, resDesc3, resDesc4;
  // CHECK: resDesc0.set_data_type(dpct::image_data_type::matrix);
  resDesc0.resType = cudaResourceTypeArray;
  // CHECK: resDesc1.set_data_ptr(pArr);
  resDesc1.res.array.array = pArr;
  // CHECK: resDesc0.set_data_type(dpct::image_data_type::matrix);
  resDesc0.resType = cudaResourceTypeMipmappedArray;
  // CHECK: resDesc2.set_data_ptr(pMipMapArr);
  resDesc2.res.mipmap.mipmap = pMipMapArr;
  // CHECK: resDesc0.set_data_type(dpct::image_data_type::linear);
  resDesc0.resType = cudaResourceTypeLinear;
  // CHECK: resDesc3.set_data_ptr(output);
  resDesc3.res.linear.devPtr = output;
  // CHECK: resDesc3.set_channel(desc);
  resDesc3.res.linear.desc = desc;
  // CHECK: resDesc3.set_x(sizeInBytes);
  resDesc3.res.linear.sizeInBytes = sizeInBytes;
  // CHECK: resDesc0.set_data_type(dpct::image_data_type::pitch);
  resDesc0.resType = cudaResourceTypePitch2D;
  // CHECK: resDesc4.set_data_ptr(output);
  resDesc4.res.pitch2D.devPtr = output;
  // CHECK: resDesc4.set_channel(desc);
  resDesc4.res.pitch2D.desc = desc;
  // CHECK: resDesc4.set_x(w);
  resDesc4.res.pitch2D.width = w;
  // CHECK: resDesc4.set_y(h);
  resDesc4.res.pitch2D.height = h;
  // CHECK: resDesc4.set_pitch(sizeInBytes);
  resDesc4.res.pitch2D.pitchInBytes = sizeInBytes;
  {
    // CHECK: dpct::image_data resDesc;
    cudaResourceDesc resDesc;
    // CHECK: resDesc.set_data(pArr);
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = pArr;
  }
  {
    // CHECK: dpct::image_data resDesc;
    cudaResourceDesc resDesc;
    // CHECK: resDesc.set_data_type(dpct::image_data_type::matrix);
    resDesc.resType = cudaResourceTypeMipmappedArray;
    // CHECK: resDesc.set_data_ptr(pMipMapArr);
    resDesc.res.mipmap.mipmap = pMipMapArr;
  }
  {
    // CHECK: dpct::image_data resDesc;
    cudaResourceDesc resDesc;
    // CHECK: resDesc.set_data(output, sizeInBytes, desc);
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = output;
    resDesc.res.linear.desc = desc;
    resDesc.res.linear.sizeInBytes = sizeInBytes;
  }
  {
    // CHECK: dpct::image_data resDesc;
    cudaResourceDesc resDesc;
    // CHECK: resDesc.set_data(output, w, h, sizeInBytes, desc);
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = output;
    resDesc.res.pitch2D.desc = desc;
    resDesc.res.pitch2D.width = w;
    resDesc.res.pitch2D.height = h;
    resDesc.res.pitch2D.pitchInBytes = sizeInBytes;
  }

  // CHECK: dpct::sampling_info texDesc1, texDesc2, texDesc3, texDesc4;
  cudaTextureDesc texDesc1, texDesc2, texDesc3, texDesc4;
  // CHECK: texDesc1.set(sycl::addressing_mode::repeat);
  texDesc1.addressMode[0] = cudaAddressModeWrap;
  // CHECK: texDesc2.set(sycl::addressing_mode::clamp_to_edge);
  texDesc2.addressMode[0] = cudaAddressModeClamp;
  // CHECK: texDesc3.set(sycl::addressing_mode::mirrored_repeat);
  texDesc3.addressMode[0] = cudaAddressModeMirror;
  // CHECK: texDesc4.set(sycl::addressing_mode::clamp);
  texDesc4.addressMode[0] = cudaAddressModeBorder;
  // CHECK: texDesc1.set(sycl::filtering_mode::nearest);
  texDesc1.filterMode = cudaFilterModePoint;
  // CHECK: texDesc2.set(sycl::filtering_mode::linear);
  texDesc2.filterMode = cudaFilterModeLinear;
  // CHECK: texDesc3.set(sycl::coordinate_normalization_mode::unnormalized);
  texDesc3.normalizedCoords = 0;
  // CHECK: texDesc4.set(sycl::coordinate_normalization_mode::normalized);
  texDesc4.normalizedCoords = 1;
  // CHECK: texDesc1.set_max_anisotropy(1);
  texDesc1.maxAnisotropy = 1;
  // CHECK: texDesc1.set_mipmap_filtering(sycl::filtering_mode::nearest);
  texDesc1.mipmapFilterMode = cudaFilterModePoint;
  // CHECK: texDesc2.set_mipmap_filtering(sycl::filtering_mode::linear);
  texDesc2.mipmapFilterMode = cudaFilterModeLinear;
  // CHECK: texDesc1.set_min_mipmap_level_clamp(1);
  texDesc1.minMipmapLevelClamp = 1;
  // CHECK:  texDesc1.set_max_mipmap_level_clamp(1);
  texDesc1.maxMipmapLevelClamp = 1;

  // CHECK: sycl::ext::oneapi::experimental::sampled_image_handle tex;
  cudaTextureObject_t tex;
  // CHECK: tex = dpct::experimental::create_bindless_image(resDesc1, texDesc1);
  cudaCreateTextureObject(&tex, &resDesc1, &texDesc1, NULL);
  // CHECK: desc = pArr->get_channel();
  cudaGetChannelDesc(&desc, pArr);
  // CHECK: resDesc1 = dpct::experimental::get_data(tex);
  cudaGetTextureObjectResourceDesc(&resDesc1, tex);
  // CHECK: texDesc1 = dpct::experimental::get_sampling_info(tex);
  cudaGetTextureObjectTextureDesc(&texDesc1, tex);
  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT: sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT: [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:   kernel<sycl::float4>(tex);
  // CHECK-NEXT: });
  kernel<float4><<<1, 1>>>(tex);
  // CHECK: dpct::experimental::destroy_bindless_image(tex, q_ct1);
  cudaDestroyTextureObject(tex);
  // CHECK: delete pArr;
  cudaFreeArray(pArr);
  // CHECK: delete pMipMapArr;
  cudaFreeMipmappedArray(pMipMapArr);
  return 0;
}
