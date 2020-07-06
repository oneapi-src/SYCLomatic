// RUN: dpcpp image_array.cpp -o image_array

#define DPCT_NAMED_LAMBDA
#include <dpct/dpct.hpp>

dpct::image<cl::sycl::float4, 3, true> tex43;
dpct::image<cl::sycl::float2, 2, true> tex22;

void test_image(sycl::float4* out, dpct::image_accessor<cl::sycl::float4, 2,true> acc42,
                  dpct::image_accessor<cl::sycl::float2, 1,true> acc21) {
  out[0] = acc42.read(16, 0.5f, 0.5f);
  cl::sycl::float2 data32 = acc21.read(16, 0.5f);
  out[1].x() = data32.x();
  out[1].y() = data32.y();
}

int main() {

  cl::sycl::float4 *host_buffer = new cl::sycl::float4[640 * 480 * 24];

  for(int i = 0; i < 640 * 480 * 24; ++i) {
	  host_buffer[i] = sycl::float4{10.0f, 10.0f, 10.0f, 10.0f};
  }
  cl::sycl::float4 *device_buffer;
  dpct::dpct_malloc(&device_buffer,
                      640 * 480 * 24 * sizeof(cl::sycl::float4));
  dpct::dpct_memcpy(device_buffer, host_buffer, 640 * 480 * 24 * sizeof(sycl::float4));

  dpct::image_channel chn2 =
      dpct::create_image_channel(32, 32, 0, 0, dpct::channel_float);
  dpct::image_channel chn4 =
      dpct::create_image_channel(32, 32, 32, 32, dpct::channel_float);
  chn4.set_channel_size(4, 32);

  dpct::image_matrix_p array1;
  dpct::image_matrix_p array2;
  dpct::image_matrix_p array3;
  
  array2 = new dpct::image_matrix(chn2, sycl::range<2>(640, 480));
  array3 = new dpct::image_matrix(chn4, sycl::range<3>(640, 480, 24));

  dpct::dpct_memcpy(array2->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(host_buffer, 640 * 480 * sizeof(cl::sycl::float2), 640 * 480 * sizeof(cl::sycl::float2), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(640 * 480 * sizeof(cl::sycl::float2), 1, 1));
  dpct::dpct_memcpy(array3->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(device_buffer, 640 * 480 * 24 * sizeof(cl::sycl::float4), 640 * 480 * 24 * sizeof(cl::sycl::float4), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(640 * 480 * 24 * sizeof(cl::sycl::float4), 1, 1));

  tex43.attach(array3);
  tex22.attach(array2);

  tex43.addr_mode()=cl::sycl::addressing_mode::clamp;
  tex22.addr_mode()=cl::sycl::addressing_mode::clamp;

  tex43.coord_normalized()=1;
  tex22.coord_normalized()=1;

  tex43.filter_mode()=cl::sycl::filtering_mode::linear;
  tex22.filter_mode()=cl::sycl::filtering_mode::linear;

  sycl::float4 d[32];
  for(int i = 0; i < 32; ++i) {
	  d[i] = sycl::float4{1.0f, 1.0f, 1.0f, 1.0f};
  }
  {
    sycl::buffer<sycl::float4, 1> buf(d, sycl::range<1>(32));
    dpct::get_default_queue().submit([&](cl::sycl::handler &cgh) {
      auto acc42 = tex43.get_access(cgh);
      auto acc21 = tex22.get_access(cgh);

      auto smpl42 = tex43.get_sampler();
      auto smpl21 = tex22.get_sampler();

      auto acc_out = buf.get_access<sycl::access::mode::read_write, sycl::access::target::global_buffer>(cgh);

      cgh.single_task<dpct_kernel_name<class dpct_single_kernel>>([=] {
        test_image(acc_out.get_pointer(),dpct::image_accessor<cl::sycl::float4, 2, true>(smpl42, acc42),
                   dpct::image_accessor<cl::sycl::float2, 1, true>(smpl21, acc21));
      });
    });
  }

  printf("d[0]: x[%f] y[%f] z[%f] w[%f]\n", d[0].x(), d[0].y(), d[0].z(), d[0].w());
  printf("d[1]: x[%f] y[%f] z[%f] w[%f]\n", d[1].x(), d[1].y(), d[1].z(), d[1].w());

  tex43.detach();
  tex22.detach();
}
