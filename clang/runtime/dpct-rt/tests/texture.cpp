// RUN: compute++ texture.cpp -o texture -sycl-driver -I
// /usr/local/ComputeCpp/include/ -L /usr/local/ComputeCpp/lib/ -lOpenCL
// -lComputeCpp

#include "../include/dpct.hpp"

dpct::dpct_image<cl::sycl::float4, 2> tex42;
dpct::dpct_image<cl::sycl::float2, 1> tex21;
dpct::dpct_image<unsigned short, 3> tex13;

void test_texture(dpct::dpct_image_accessor<cl::sycl::float4, 2> acc42,
                  dpct::dpct_image_accessor<cl::sycl::float2, 1> acc21,
                  dpct::dpct_image_accessor<unsigned short, 3> acc13) {
  cl::sycl::float4 data42 = dpct::dpct_read_image(acc42, 1.0f, 1.0f);
  unsigned short data13 = dpct::dpct_read_image(acc13, 1.0f, 1.0f, 1.0f);
  cl::sycl::float2 data32 = dpct::dpct_read_image(acc21, 1.0f);
}

int main() {

  cl::sycl::float4 *host_buffer = new cl::sycl::float4[640 * 480 * 24];
  cl::sycl::float4 *device_buffer;
  dpct::dpct_malloc(&device_buffer,
                      640 * 480 * 24 * sizeof(cl::sycl::float4));

  dpct::dpct_image_channel chn1 =
      dpct::create_image_channel(16, 0, 0, 0, dpct::channel_unsigned);
  dpct::dpct_image_channel chn2 =
      dpct::create_image_channel(32, 32, 0, 0, dpct::channel_float);
  dpct::dpct_image_channel chn4 =
      dpct::create_image_channel(32, 32, 32, 32, dpct::channel_float);

  dpct::dpct_image_matrix_p array1;
  dpct::dpct_image_matrix_p array2;
  dpct::dpct_image_matrix_p array3;

  dpct::dpct_malloc_matrix(&array1, &chn2, 640);
  dpct::dpct_malloc_matrix(&array2, &chn4, 640, 480);
  dpct::dpct_malloc_matrix(&array3, &chn1, 640, 480, 24);

  dpct::dpct_memcpy_to_matrix(array1, 0, 0, host_buffer, 640 * sizeof(cl::sycl::float2));
  dpct::dpct_memcpy_to_matrix(array2, 0, 0, host_buffer, 640 * 480 * sizeof(cl::sycl::float4));
  dpct::dpct_memcpy_to_matrix(array3, 0, 0, device_buffer, 640 * 480 * 24 * sizeof(unsigned short));

  dpct::dpct_attach_image(tex42, array2);
  dpct::dpct_attach_image(tex21, array1);
  dpct::dpct_attach_image(tex13, array3);

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

  dpct::dpct_detach_image(tex42);
  dpct::dpct_detach_image(tex21);
  dpct::dpct_detach_image(tex13);
}
