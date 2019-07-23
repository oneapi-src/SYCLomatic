// RUN: compute++ texture.cpp -o texture -sycl-driver -I
// /usr/local/ComputeCpp/include/ -L /usr/local/ComputeCpp/lib/ -lOpenCL
// -lComputeCpp

#include "../include/syclct.hpp"

syclct::syclct_texture<cl::sycl::float4, 2> tex42;
syclct::syclct_texture<cl::sycl::float2, 1> tex21;
syclct::syclct_texture<float, 3> tex13;

void test_texture(syclct::syclct_texture_accessor<cl::sycl::float4, 2> acc42,
                  syclct::syclct_texture_accessor<cl::sycl::float2, 1> acc21,
                  syclct::syclct_texture_accessor<float, 3> acc13) {
  cl::sycl::float4 data42 = syclct::syclct_read_texture(acc42, 1.0f, 1.0f);
  float data13 = syclct::syclct_read_texture(acc13, 1.0f, 1.0f, 1.0f);
  cl::sycl::float2 data32 = syclct::syclct_read_texture(acc21, 1.0f);
}

int main() {

  cl::sycl::float4 *host_buffer = new cl::sycl::float4[640 * 480 * 24];
  cl::sycl::float4 *device_buffer;
  syclct::sycl_malloc(&device_buffer,
                      640 * 480 * 24 * sizeof(cl::sycl::float4));

  syclct::syclct_channel_desc chn1 =
      syclct::create_channel_desc(32, 0, 0, 0, syclct::channel_float);
  syclct::syclct_channel_desc chn2 =
      syclct::create_channel_desc(32, 32, 0, 0, syclct::channel_float);
  syclct::syclct_channel_desc chn4 =
      syclct::create_channel_desc(32, 32, 32, 32, syclct::channel_float);

  syclct::syclct_array array1;
  syclct::syclct_array array2;
  syclct::syclct_array array3;

  syclct::syclct_malloc_array(&array1, &chn1, 640);
  syclct::syclct_malloc_array(&array2, &chn4, 640, 480);
  syclct::syclct_malloc_array(&array3, &chn2, 640, 480, 24);

  syclct::syclct_memcpy_to_array(array1, host_buffer);
  syclct::syclct_memcpy_to_array(array2, host_buffer);
  syclct::syclct_memcpy_to_array(array3, device_buffer);

  syclct::syclct_bind_texture(tex42, array2);
  syclct::syclct_bind_texture(tex21, array1);
  syclct::syclct_bind_texture(tex13, array3);

  tex42.set_addr_mode(cl::sycl::addressing_mode::clamp);
  tex21.set_addr_mode(cl::sycl::addressing_mode::clamp);
  tex13.set_addr_mode(cl::sycl::addressing_mode::clamp);

  tex42.set_coord_norm_mode(1);
  tex21.set_coord_norm_mode(1);
  tex13.set_coord_norm_mode(1);

  tex42.set_filter_mode(cl::sycl::filtering_mode::linear);
  tex21.set_filter_mode(cl::sycl::filtering_mode::linear);
  tex13.set_filter_mode(cl::sycl::filtering_mode::linear);

  {
    syclct::get_default_queue().submit([&](cl::sycl::handler &cgh) {
      auto acc42 = tex42.get_access(cgh);
      auto acc13 = tex13.get_access(cgh);
      auto acc21 = tex21.get_access(cgh);
      cgh.single_task<syclct_kernel_name<class syclct_single_kernel>>(
          [=] { test_texture(acc42, acc21, acc13); });
    });
  }

  syclct::syclct_unbind_texture(tex42);
  syclct::syclct_unbind_texture(tex21);
  syclct::syclct_unbind_texture(tex13);
}