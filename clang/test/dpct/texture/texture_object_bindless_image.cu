// RUN: dpct --format-range=none --use-experimental-features=bindless_images -out-root %T/texture/texture_object_bindless_image %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/texture/texture_object_bindless_image/texture_object_bindless_image.dp.cpp --match-full-lines %s

__global__ void kernel(cudaTextureObject_t tex) {
  int i;
  // CHECK: sycl::ext::oneapi::experimental::read_image<sycl::short2>(tex, (float)i);
  tex1Dfetch<short2>(tex, i);
}

int main() {
  void *input;
  size_t sizeInBytes;
  // CHECK: dpct::image_matrix_p pArr;
  cudaArray_t pArr;
  // TODO: need support.
  // cudaMipmappedArray_t pMipMapArr;
  // CHECK: dpct::image_channel desc;
  cudaChannelFormatDesc desc;
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
  // CHECK: dpct::sampling_info texDesc;
  cudaTextureDesc texDesc;
  // CHECK: sycl::ext::oneapi::experimental::sampled_image_handle tex;
  cudaTextureObject_t tex;
  // CHECK: tex = dpct::create_bindless_image(resDesc, texDesc);
  cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
  // CHECK: sycl::ext::oneapi::experimental::destroy_image_handle(tex, dpct::get_in_order_queue());
  cudaDestroyTextureObject(tex);
  return 0;
}
