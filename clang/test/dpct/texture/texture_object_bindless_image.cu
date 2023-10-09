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
  // CHECK: dpct::image_channel desc;
  cudaChannelFormatDesc desc;
  // CHECK: dpct::image_mem_wrapper resDesc;
  cudaResourceDesc resDesc;
  // CHECK: resDesc.set_num_levels(dpct::image_data_type::linear);
  resDesc.resType = cudaResourceTypeLinear;
  // CHECK: resDesc.set_data_ptr(input);
  resDesc.res.linear.devPtr = input;
  // CHECK: resDesc.set_channel(desc);
  resDesc.res.linear.desc = desc;
  // CHECK: resDesc.set_x(sizeInBytes);
  resDesc.res.linear.sizeInBytes = sizeInBytes;
  // CHECK: dpct::sampling_info texDesc;
  cudaTextureDesc texDesc;
  // CHECK: sycl::ext::oneapi::experimental::sampled_image_handle tex;
  cudaTextureObject_t tex;
  // CHECK: tex = resDesc.create_image(texDesc, q_ct1);
  cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
  // CHECK: sycl::ext::oneapi::experimental::destroy_image_handle(tex, q_ct1);
  cudaDestroyTextureObject(tex);
  return 0;
}
