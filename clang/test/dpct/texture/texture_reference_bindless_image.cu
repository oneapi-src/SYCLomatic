// UNSUPPORTED: cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3
// UNSUPPORTED: v12.0, v12.1, v12.2, v12.3
// RUN: dpct --format-range=none --use-experimental-features=bindless_images -out-root %T/texture/texture_reference_bindless_image %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/texture/texture_reference_bindless_image/texture_reference_bindless_image.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/texture/texture_reference_bindless_image/texture_reference_bindless_image.dp.cpp -o %T/texture/texture_reference_bindless_image/texture_reference_bindless_image.dp.o %}

// CHECK: dpct::experimental::bindless_image_wrapper<sycl::short2, 1> tex0;
static texture<short2, 1> tex0;
// CHECK: dpct::experimental::bindless_image_wrapper<sycl::float4, 2> tex;
static texture<float4, 2> tex;

// CHECK: void kernel(sycl::ext::oneapi::experimental::sampled_image_handle tex0,
// CHECK-NEXT:             sycl::ext::oneapi::experimental::sampled_image_handle tex) {
__global__ void kernel() {
  // CHECK: sycl::ext::oneapi::experimental::read_image<sycl::short2>(tex0, (float)1);
  tex1D(tex0, 1);
  // CHECK: sycl::float4 f42 = sycl::ext::oneapi::experimental::read_image<sycl::float4>(tex, sycl::float2(1.0f, 1.0f));
  float4 f42 = tex2D(tex, 1.0f, 1.0f);
}

int main() {
  int i;
  // CHECK: tex.set(sycl::addressing_mode::repeat);
  tex.addressMode[0] = cudaAddressModeWrap;
  // CHECK: auto addressMode = tex.get_addressing_mode();
  auto addressMode = tex.addressMode[0];
  // CHECK: tex.set_channel_size(1, i);
  tex.channelDesc.x = i;
  // CHECK: i = tex.get_channel_size();
  i = tex.channelDesc.x;
  // CHECK: tex.set_channel_size(2, i);
  tex.channelDesc.y = i;
  // CHECK: i = tex.get_channel_size();
  i = tex.channelDesc.y;
  // CHECK: tex.set_channel_size(3, i);
  tex.channelDesc.z = i;
  // CHECK: i = tex.get_channel_size();
  i = tex.channelDesc.z;
  // CHECK: tex.set_channel_size(4, i);
  tex.channelDesc.w = i;
  // CHECK: i = tex.get_channel_size();
  i = tex.channelDesc.w;
  // CHECK: tex.set_channel_data_type(dpct::image_channel_data_type::fp);
  tex.channelDesc.f = cudaChannelFormatKindFloat;
  // CHECK: auto f = tex.get_channel_data_type();
  auto f = tex.channelDesc.f;
  // CHECK: tex.set_channel(dpct::image_channel::create<sycl::float4>());
  tex.channelDesc = cudaCreateChannelDesc<float4>();
  // CHECK:  auto channelDesc = tex.get_channel();
  auto channelDesc = tex.channelDesc;
  // CHECK: tex.set(sycl::filtering_mode::nearest);
  tex.filterMode = cudaFilterModePoint;
  // CHECK: auto filterMode = tex.get_filtering_mode();
  auto filterMode = tex.filterMode;
  // CHECK: tex.set(sycl::coordinate_normalization_mode::unnormalized);
  tex.normalized = 0;
  // CHECK: i = tex.is_coordinate_normalized();
  i = tex.normalized;

  void *dataPtr;
  const size_t w = 4;
  const size_t h = 2;
  size_t pitch = sizeof(float4) * 4;
  float4 expect[h * w] = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
  };
  cudaMalloc(&dataPtr, sizeof(expect));
  cudaMemcpy(dataPtr, &expect, sizeof(expect), cudaMemcpyHostToDevice);
  // CHECK: tex.attach(dataPtr, pitch * h);
  cudaBindTexture(0, tex, dataPtr, pitch * h);
  // CHECK: tex.attach(dataPtr, w, h, pitch);
  cudaBindTexture2D(0, tex, dataPtr, w, h, pitch);
  // CHECK: dpct::experimental::image_mem_wrapper_ptr pArr;
  cudaArray_t pArr;
  // CHECK: tex.attach(pArr);
  cudaBindTextureToArray(tex, pArr);
  // CHECK: q_ct1.submit(
  // CHECK-NEXT: [&](sycl::handler &cgh) {
  // CHECK-NEXT:   auto tex0_handle = tex0.get_handle();
  // CHECK-NEXT:   auto tex_handle = tex.get_handle();
  // CHECK-EMPTY:
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:     sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:     [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:       kernel(tex0_handle, tex_handle);
  // CHECK-NEXT:     });
  // CHECK-NEXT: });
  kernel<<<1, 1>>>();
  // CHECK: tex.detach();
  cudaUnbindTexture(tex);

  return 0;
}
