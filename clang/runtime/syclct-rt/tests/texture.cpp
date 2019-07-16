// RUN: compute++ texture.cpp -o texture -sycl-driver -I /usr/local/ComputeCpp/include/ -L /usr/local/ComputeCpp/lib/ -lOpenCL -lComputeCpp

#include "../include/syclct.hpp"

syclct::syclct_texture<float, 2, 4> tex24;
syclct::syclct_texture<float, 1, 3> tex13;
syclct::syclct_texture<float, 3, 2> text32;

void test_texture(syclct::syclct_texture_accessor<float, 2, 4> acc24,
                  syclct::syclct_texture_accessor<float, 1, 3> acc13,
                  syclct::syclct_texture_accessor<float, 3, 2> acc32) {
  cl::sycl::float4 data24 = acc24.read(cl::sycl::float2(1.0f, 1.0f));
  cl::sycl::float4 data13 = acc13.read(1.0f);
  cl::sycl::float2 data32 = acc32.read(cl::sycl::float4(1.0f, 1.0f, 1.0f, 0));
}

int main() {

  cl::sycl::float4 *host_buffer = new cl::sycl::float4[640 * 480 * 24];
  cl::sycl::float4 *device_buffer;
  syclct::sycl_malloc(&device_buffer,
                      640 * 480 * 24 * sizeof(cl::sycl::float4));

  syclct::syclct_channel_desc chn2 = syclct::cread_channel_desc(32, 32, 0, 0, 2);
  syclct::syclct_channel_desc chn3 =
      syclct::cread_channel_desc(32, 32, 32, 0, 2);
  syclct::syclct_channel_desc chn4 =
      syclct::cread_channel_desc(32, 32, 32, 32, 2);

  syclct::syclct_array<1> array1;
  syclct::syclct_array<2> array2;
  syclct::syclct_array<3> array3;

  array1.init(chn3, cl::sycl::range<1>(640));
  array2.init(chn4, cl::sycl::range<2>(640, 480));
  array3.init(chn2, cl::sycl::range<3>(640, 480, 24));

  array1.copy_from(host_buffer);
  array2.copy_from(host_buffer);
  array3.copy_from(device_buffer);

  tex24.bind(array2);
  tex13.bind(array1);
  text32.bind(array3);

  tex24.set_addr_mode(0);
  tex13.set_addr_mode(0);
  text32.set_addr_mode(0);

  tex24.set_coord_norm_mode(1);
  tex13.set_coord_norm_mode(1);
  text32.set_coord_norm_mode(1);

  tex24.set_filter_mode(1);
  tex13.set_filter_mode(1);
  text32.set_filter_mode(1);

  {
    syclct::get_default_queue().submit([&](cl::sycl::handler &cgh) {
      auto acc24 = tex24.get_access(cgh);
      auto acc13 = tex13.get_access(cgh);
      auto acc32 = text32.get_access(cgh);
      cgh.single_task<syclct_kernel_name<class syclct_single_kernel>>(
          [=] { test_texture(acc24, acc13, acc32); });
    });
  }

  array1.free();
  array2.free();
  array3.free();
}