// RUN: dpct --format-range=none --use-experimental-features=bindless_images -out-root %T/texture/texture_object_bindless_image %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/texture/texture_object_bindless_image/texture_object_bindless_image.dp.cpp --match-full-lines %s

__global__ void kernel(cudaTextureObject_t tex) {
  int i;
  float j, k;
  // CHECK: sycl::ext::oneapi::experimental::read_image<sycl::short2>(tex, (float)i);
  tex1Dfetch<short2>(tex, i);
  // CHECK: sycl::ext::oneapi::experimental::read_image<sycl::short2>(tex, sycl::float2(j, k));
  tex2D<short2>(tex, j, k);
}

int main() {
  void *input;
  size_t w, h, sizeInBytes;
  // CHECK: sycl::ext::oneapi::experimental::image_mem* pArr;
  cudaArray_t pArr;
  // TODO: need support.
  // cudaMipmappedArray_t pMipMapArr;
  // CHECK: dpct::image_channel desc;
  cudaChannelFormatDesc desc;
  // CHECK: pArr = new sycl::ext::oneapi::experimental::image_mem(sycl::ext::oneapi::experimental::image_descriptor({w, h}, desc.get_channel_order(), desc.get_channel_type()), q_ct1);
  cudaMallocArray(&pArr, &desc, w, h);
  // CHECK: q_ct1.ext_oneapi_copy(input, sycl::range<3>(0, 0, 0), sycl::range<3>(0, 0, 0), pArr->get_handle(), sycl::range<3>(0, 0, 0), pArr->get_descriptor(), sycl::range<3>(4 * w / dpct::getEleSize(pArr->get_descriptor()), h, 0));
  cudaMemcpy2DToArray(pArr, 0, 0, input, 4 * w, 4 * w, h,
                      cudaMemcpyHostToDevice);

  // CHECK: dpct::image_data resDesc;
  cudaResourceDesc resDesc;
  // CHECK: resDesc.set_data_type(dpct::image_data_type::matrix);
  resDesc.resType = cudaResourceTypeArray;
  // TODO: need support.
  // resDesc.resType = cudaResourceTypeMipmappedArray;
  // CHECK: resDesc.set_data_type(dpct::image_data_type::linear);
  resDesc.resType = cudaResourceTypeLinear;
  // CHECK: resDesc.set_data_type(dpct::image_data_type::pitch);
  resDesc.resType = cudaResourceTypePitch2D;
  // CHECK: resDesc.set_data_ptr(pArr);
  resDesc.res.array.array = pArr;
  // TODO: need support.
  // resDesc.res.mipmap.mipmap = pMipMapArr;
  // CHECK: resDesc.set_data_ptr(input);
  resDesc.res.linear.devPtr = input;
  // CHECK: resDesc.set_channel(desc);
  resDesc.res.linear.desc = desc;
  // CHECK: resDesc.set_x(sizeInBytes);
  resDesc.res.linear.sizeInBytes = sizeInBytes;
  // CHECK: resDesc.set_data_ptr(input);
  resDesc.res.pitch2D.devPtr = input;
  // CHECK: resDesc.set_channel(desc);
  resDesc.res.pitch2D.desc = desc;
  // CHECK: resDesc.set_x(sizeInBytes);
  resDesc.res.pitch2D.width = sizeInBytes;
  // CHECK: resDesc.set_y(sizeInBytes);
  resDesc.res.pitch2D.height = sizeInBytes;
  // CHECK: resDesc.set_pitch(sizeInBytes);
  resDesc.res.pitch2D.pitchInBytes = sizeInBytes;

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

  // CHECK: sycl::ext::oneapi::experimental::sampled_image_handle tex;
  cudaTextureObject_t tex;
  // CHECK: tex = dpct::create_bindless_image(resDesc, texDesc1);
  cudaCreateTextureObject(&tex, &resDesc, &texDesc1, NULL);
  // CHECK: q_ct1.parallel_for(
  // CHECK-NEXT: sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT: [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:   kernel(tex);
  // CHECK-NEXT: });
  kernel<<<1, 1>>>(tex);
  // CHECK: sycl::ext::oneapi::experimental::destroy_image_handle(tex, q_ct1);
  cudaDestroyTextureObject(tex);

  return 0;
}
