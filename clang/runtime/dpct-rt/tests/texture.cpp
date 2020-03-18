// RUN: dpcpp texture.cpp -o texture

#define DPCT_NAMED_LAMBDA
#include "../include/dpct.hpp"

dpct::image<cl::sycl::float4, 2> tex42;
dpct::image<cl::sycl::float2, 1> tex21;
dpct::image<unsigned short, 3> tex13;

void test_texture(dpct::image_accessor<cl::sycl::float4, 2> acc42,
                  dpct::image_accessor<cl::sycl::float2, 1> acc21,
                  dpct::image_accessor<unsigned short, 3> acc13) {
  cl::sycl::float4 data42 = dpct::read_image(acc42, 1.0f, 1.0f);
  unsigned short data13 = dpct::read_image(acc13, 1.0f, 1.0f, 1.0f);
  cl::sycl::float2 data32 = dpct::read_image(acc21, 1.0f);

}

int main() {

  cl::sycl::float4 *host_buffer = new cl::sycl::float4[640 * 480 * 24];
  cl::sycl::float4 *device_buffer;
  dpct::dpct_malloc(&device_buffer,
                      640 * 480 * 24 * sizeof(cl::sycl::float4));

  dpct::image_channel chn1 =
      dpct::create_image_channel(16, 0, 0, 0, dpct::channel_unsigned);
  dpct::image_channel chn2 =
      dpct::create_image_channel(32, 32, 0, 0, dpct::channel_float);
  dpct::image_channel chn4 =
      dpct::create_image_channel(32, 32, 32, 32, dpct::channel_float);

  dpct::image_matrix_p array1;
  dpct::image_matrix_p array2;
  dpct::image_matrix_p array3;

  array1 = new dpct::image_matrix(chn2, sycl::range<2>(640, 1));
  array2 = new dpct::image_matrix(chn4, sycl::range<2>(640, 480));
  array3 = new dpct::image_matrix(chn1, sycl::range<3>(640, 480, 24));

  dpct::dpct_memcpy(array1->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(host_buffer, 640 * sizeof(cl::sycl::float2), 640 * sizeof(cl::sycl::float2), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(640 * sizeof(cl::sycl::float2), 1, 1));
  dpct::dpct_memcpy(array2->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(host_buffer, 640 * 480 * sizeof(cl::sycl::float4), 640 * 480 * sizeof(cl::sycl::float4), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(640 * 480 * sizeof(cl::sycl::float4), 1, 1));
  dpct::dpct_memcpy(array3->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(device_buffer, 640 * 480 * 24 * sizeof(unsigned short), 640 * 480 * 24 * sizeof(unsigned short), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(640 * 480 * 24 * sizeof(unsigned short), 1, 1));

  dpct::attach_image(tex42, array2);
  dpct::attach_image(tex21, array1);
  dpct::attach_image(tex13, array3);

  tex42.addr_mode()=cl::sycl::addressing_mode::clamp;
  tex21.addr_mode()=cl::sycl::addressing_mode::clamp;
  tex13.addr_mode()=cl::sycl::addressing_mode::clamp;

  tex42.coord_normalized()=1;
  tex21.coord_normalized()=1;
  tex13.coord_normalized()=1;

  tex42.filter_mode()=cl::sycl::filtering_mode::linear;
  tex21.filter_mode()=cl::sycl::filtering_mode::linear;
  tex13.filter_mode()=cl::sycl::filtering_mode::linear;

  {
    dpct::get_default_queue().submit([&](cl::sycl::handler &cgh) {
      auto acc42 = tex42.get_access(cgh);
      auto acc13 = tex13.get_access(cgh);
      auto acc21 = tex21.get_access(cgh);
      cgh.single_task<dpct_kernel_name<class dpct_single_kernel>>(
          [=] { test_texture(acc42, acc21, acc13); });
    });
  }

  dpct::detach_image(tex42);
  dpct::detach_image(tex21);
  dpct::detach_image(tex13);
}
