// RUN: dpcpp image_array.cpp -o image_array

#define DPCT_NAMED_LAMBDA
#include <dpct/dpct.hpp>

dpct::image_wrapper<cl::sycl::float4, 3, true> image43;
dpct::image_wrapper<cl::sycl::float2, 2, true> image22;

void test_image(sycl::float4* out, dpct::image_accessor_ext<cl::sycl::float4, 2,true> acc42,
                  dpct::image_accessor_ext<cl::sycl::float2, 1,true> acc21) {
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
  cl::sycl::float4 *device_buffer = (cl::sycl::float4 *)dpct::dpct_malloc(
      640 * 480 * 24 * sizeof(cl::sycl::float4));
  dpct::dpct_memcpy(device_buffer, host_buffer, 640 * 480 * 24 * sizeof(sycl::float4));

  dpct::image_channel chn2 =
      dpct::image_channel(32, 32, 0, 0, dpct::image_channel_data_type::fp);
  dpct::image_channel chn4 =
      dpct::image_channel(32, 32, 32, 32, dpct::image_channel_data_type::fp);
  chn4.set_channel_size(4, 32);

  dpct::image_matrix_p array1;
  dpct::image_matrix_p array2;
  dpct::image_matrix_p array3;
  
  array2 = new dpct::image_matrix(chn2, sycl::range<2>(640, 480));
  array3 = new dpct::image_matrix(chn4, sycl::range<3>(640, 480, 24));

  dpct::dpct_memcpy(array2->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(host_buffer, 640 * 480 * sizeof(cl::sycl::float2), 640 * 480 * sizeof(cl::sycl::float2), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(640 * 480 * sizeof(cl::sycl::float2), 1, 1));
  dpct::dpct_memcpy(array3->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(device_buffer, 640 * 480 * 24 * sizeof(cl::sycl::float4), 640 * 480 * 24 * sizeof(cl::sycl::float4), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(640 * 480 * 24 * sizeof(cl::sycl::float4), 1, 1));

  dpct::image_wrapper_base *image22;
  dpct::image_data res22;
  dpct::sampling_info smpl22;
  res22.set_data_type(dpct::image_data_type::matrix);
  res22.set_data_ptr(array2);

  image43.attach(array3);

  image43.set(cl::sycl::addressing_mode::clamp);
  smpl22.set(cl::sycl::addressing_mode::clamp);

  image43.set(cl::sycl::coordinate_normalization_mode::normalized);
  smpl22.set(cl::sycl::coordinate_normalization_mode::unnormalized);
  smpl22.set_coordinate_normalization_mode(1);

  image43.set(cl::sycl::filtering_mode::linear);
  smpl22.set(cl::sycl::filtering_mode::linear);

  image22 = dpct::create_image_wrapper(res22, smpl22);

  sycl::float4 d[32];
  for(int i = 0; i < 32; ++i) {
	  d[i] = sycl::float4{1.0f, 1.0f, 1.0f, 1.0f};
  }
  {
    sycl::buffer<sycl::float4, 1> buf(d, sycl::range<1>(32));
    dpct::get_default_queue().submit([&](cl::sycl::handler &cgh) {
      auto acc42 = image43.get_access(cgh);
      auto acc21 = static_cast<dpct::image_wrapper<cl::sycl::float2, 2, true> *>(image22)->get_access(cgh);

      auto smpl42 = image43.get_sampler();
      auto smpl21 = image22->get_sampler();

      auto acc_out = buf.get_access<sycl::access::mode::read_write, sycl::access::target::global_buffer>(cgh);

      cgh.single_task<dpct_kernel_name<class dpct_single_kernel>>([=] {
        test_image(acc_out.get_pointer(),dpct::image_accessor_ext<cl::sycl::float4, 2, true>(smpl42, acc42),
                   dpct::image_accessor_ext<cl::sycl::float2, 1, true>(smpl21, acc21));
      });
    });
  }

  printf("d[0]: x[%f] y[%f] z[%f] w[%f]\n", d[0].x(), d[0].y(), d[0].z(), d[0].w());
  printf("d[1]: x[%f] y[%f] z[%f] w[%f]\n", d[1].x(), d[1].y(), d[1].z(), d[1].w());

  image43.detach();

  delete image22;
}
