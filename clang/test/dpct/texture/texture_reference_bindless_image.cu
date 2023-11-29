// UNSUPPORTED: cuda-12.0, cuda-12.1, cuda-12.2
// UNSUPPORTED: v12.0, v12.1, v12.2
// RUN: dpct --format-range=none --use-experimental-features=bindless_images -out-root %T/texture/texture_reference_bindless_image %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/texture/texture_reference_bindless_image/texture_reference_bindless_image.dp.cpp --match-full-lines %s

// CHECK: dpct::bindless_image_wrapper<sycl::float4, 2> tex;
static texture<float4, 2> tex;

// CHECK: void kernel(sycl::ext::oneapi::experimental::sampled_image_handle tex) {
__global__ void kernel() {
  // CHECK: sycl::float4 f42 = sycl::ext::oneapi::experimental::read_image<sycl::float4>(tex, sycl::float2(1.0f, 1.0f));
  float4 f42 = tex2D(tex, 1.0f, 1.0f);
}

int main() {
  // CHECK: tex.set(sycl::addressing_mode::repeat);
  tex.addressMode[0] = cudaAddressModeWrap;
  // CHECK: tex.set_channel(dpct::image_channel::create<sycl::float4>());
  tex.channelDesc = cudaCreateChannelDesc<float4>();
  // CHECK: tex.set(sycl::filtering_mode::nearest);
  tex.filterMode = cudaFilterModePoint;
  // CHECK: tex.set(sycl::coordinate_normalization_mode::unnormalized);
  tex.normalized = 0;

  void *dataPtr;
  size_t w, h, pitch;
  // CHECK: tex.attach(dataPtr, w, h, pitch);
  cudaBindTexture2D(0, tex, dataPtr, w, h, pitch);
  // CHECK: dpct::get_in_order_queue().submit(
  // CHECK-NEXT: [&](sycl::handler &cgh) {
  // CHECK-NEXT:   auto tex_handle = tex.img;
  // CHECK-EMPTY:
  // CHECK-NEXT:   cgh.parallel_for(
  // CHECK-NEXT:     sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:     [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:       kernel(tex_handle);
  // CHECK-NEXT:     });
  // CHECK-NEXT: });
  kernel<<<1, 1>>>();
  // CHECK: tex.detach();
  cudaUnbindTexture(tex);

  return 0;
}
