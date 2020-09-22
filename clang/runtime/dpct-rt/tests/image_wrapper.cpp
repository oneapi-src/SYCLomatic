// RUN: dpcpp image_wrapper.cpp -o image_wrapper

#define DPCT_NAMED_LAMBDA
#include <dpct/dpct.hpp>

dpct::image_wrapper<cl::sycl::float4, 2> image42;
dpct::image_wrapper<cl::sycl::float2, 1> image21;
dpct::image_wrapper<unsigned short, 3> image13;

void test_image(sycl::float4* out, dpct::image_accessor_ext<cl::sycl::float4, 2> acc42,
                  dpct::image_accessor_ext<cl::sycl::float2, 1> acc21,
                  dpct::image_accessor_ext<unsigned short, 3> acc13) {
  out[0] = acc42.read(0.5f, 0.5f);
  unsigned short data13 = acc13.read(0.5f, 0.5f, 0.5f);
  cl::sycl::float2 data21 = acc21.read(0.5f);
  out[1].x() = data21.x();
  out[1].y() = data13;
}

int main() {

  cl::sycl::float4 *host_buffer = new cl::sycl::float4[640 * 480 * 24];

  for(int i = 0; i < 640 * 480 * 24; ++i) {
	  host_buffer[i] = sycl::float4{10.0f, 10.0f, 10.0f, 10.0f};
  }
  cl::sycl::float4 *device_buffer = (cl::sycl::float4 *)dpct::dpct_malloc(
      640 * 480 * 24 * sizeof(cl::sycl::float4));
  dpct::dpct_memcpy(device_buffer, host_buffer, 640 * 480 * 24 * sizeof(sycl::float4));

  dpct::image_channel chn1 =
      dpct::image_channel(16, 0, 0, 0, dpct::image_channel_data_type::unsigned_int);
  dpct::image_channel chn2 =
      dpct::image_channel(32, 32, 0, 0, dpct::image_channel_data_type::fp);
  dpct::image_channel chn4 =
      dpct::image_channel(32, 32, 32, 32, dpct::image_channel_data_type::fp);
  chn4.set_channel_data_type(dpct::image_channel_data_type::fp);
  chn4.set_channel_size(4, 32);
  dpct::image_channel_data_type chn4_type = chn4.get_channel_data_type();
  unsigned chn4_size = chn4.get_channel_size();

  sycl::float4 *image_data2 = (sycl::float4 *)std::malloc(650 * 480 * sizeof(sycl::float4));

  dpct::image_matrix_p array1;
  dpct::image_matrix_p array3;

  array1 = new dpct::image_matrix(chn2, sycl::range<1>(640));
  array3 = new dpct::image_matrix(chn1, sycl::range<3>(640, 480, 24));

  dpct::dpct_memcpy(array1->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(host_buffer, 640 * sizeof(cl::sycl::float2), 640 * sizeof(cl::sycl::float2), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(640 * sizeof(cl::sycl::float2), 1, 1));
  dpct::dpct_memcpy(dpct::pitched_data(image_data2, 650 * sizeof(cl::sycl::float4), 640 * sizeof(cl::sycl::float4*), 480), sycl::id<3>(0, 0, 0), dpct::pitched_data(host_buffer, 640 * 480 * sizeof(cl::sycl::float4), 640 * 480 * sizeof(cl::sycl::float4), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(640 * 480 * sizeof(cl::sycl::float4), 1, 1));
  dpct::dpct_memcpy(array3->to_pitched_data(), sycl::id<3>(0, 0, 0), dpct::pitched_data(device_buffer, 640 * 480 * 24 * sizeof(unsigned short), 640 * 480 * 24 * sizeof(unsigned short), 1), sycl::id<3>(0, 0, 0), sycl::range<3>(640 * 480 * 24 * sizeof(unsigned short), 1, 1));

  image42.attach(image_data2, 640 * sizeof(cl::sycl::float4), 480, 650 * sizeof(cl::sycl::float4));
  image13.attach(array3);

  dpct::image_wrapper_base *image21;
  dpct::image_data res21(array1);
  dpct::sampling_info smpl21;
  res21.set_data(image_data2, 640, chn2);
  res21.set_data(image_data2, 640, 480, 670, chn2);
  res21.set_data(array1);
  res21.set_data_ptr(array1);

  image42.set(cl::sycl::addressing_mode::clamp);
  smpl21.set(cl::sycl::addressing_mode::clamp);
  image13.set(cl::sycl::addressing_mode::clamp);

  image42.set(cl::sycl::coordinate_normalization_mode::normalized);
  smpl21.set(cl::sycl::coordinate_normalization_mode::normalized);
  image13.set(cl::sycl::coordinate_normalization_mode::normalized);

  image42.set(cl::sycl::filtering_mode::linear);
  smpl21.set(cl::sycl::filtering_mode::linear);
  image13.set(cl::sycl::filtering_mode::linear);

  image21 = dpct::create_image_wrapper(res21, smpl21);

  sycl::float4 d[32];
  for(int i = 0; i < 32; ++i) {
	  d[i] = sycl::float4{1.0f, 1.0f, 1.0f, 1.0f};
  }
  {
    sycl::buffer<sycl::float4, 1> buf(d, sycl::range<1>(32));
    dpct::get_default_queue().submit([&](cl::sycl::handler &cgh) {
      auto acc42 = image42.get_access(cgh);
      auto acc13 = image13.get_access(cgh);
      auto acc21 = static_cast<dpct::image_wrapper<cl::sycl::float2, 1> *>(image21)->get_access(cgh);

      auto smpl42 = image42.get_sampler();
      auto smpl13 = image13.get_sampler();
      auto smpl21 = image21->get_sampler();

      auto acc_out = buf.get_access<sycl::access::mode::read_write, sycl::access::target::global_buffer>(cgh);

      cgh.single_task<dpct_kernel_name<class dpct_single_kernel>>([=] {
        test_image(acc_out.get_pointer(),dpct::image_accessor_ext<cl::sycl::float4, 2>(smpl42, acc42),
                   dpct::image_accessor_ext<cl::sycl::float2, 1>(smpl21, acc21),
                   dpct::image_accessor_ext<unsigned short, 3>(smpl13, acc13));
      });
    });
  }

  printf("d[0]: x[%f] y[%f] z[%f] w[%f]\n", d[0].x(), d[0].y(), d[0].z(), d[0].w());
  printf("d[1]: x[%f] y[%f] z[%f] w[%f]\n", d[1].x(), d[1].y(), d[1].z(), d[1].w());

  image42.detach();
  image13.detach();

  delete image21;
  sycl::free(device_buffer, dpct::get_default_queue());
  std::free(host_buffer);
  std::free(image_data2);
}
