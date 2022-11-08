// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2
// RUN: dpct --format-range=none -in-root %S -out-root %T/devicelevel/devicesegmentedreduce %S/devicesegmentedreduce.cu --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/devicelevel/devicesegmentedreduce/devicesegmentedreduce.dp.cpp --match-full-lines %s

#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define DATA_NUM 100

template<typename T = int>
void init_data(T* data, int num) {
  for(int i = 0; i < num; i++)
    data[i] = i;
}

template<typename T = int>
bool verify_data(T* data, T* expect, int num, int step = 1) {
  for(int i = 0; i < num; i = i + step) {
    if(data[i] != expect[i]) {
      return false;
    }
  }
  return true;
}

template<typename T = int>
void print_data(T* data, int num) {
  for (int i = 0; i < num; i++) {
    std::cout << data[i] << ", ";
    if((i+1)%32 == 0)
        std::cout << std::endl;
  }
  std::cout << std::endl;
}

//CHECK:  struct UserMin
//CHECK:  {
//CHECK:    template <typename T>
//CHECK:    __dpct_inline__
//CHECK:    T operator()(const T &a, const T &b) const {
//CHECK:        return (b < a) ? b : a;
//CHECK:    }
//CHECK:  };

struct UserMin
{
  template <typename T>
  __device__ __host__ __forceinline__
  T operator()(const T &a, const T &b) const {
      return (b < a) ? b : a;
  }
};

//CHECK:  bool test_reduce_1(){
//CHECK:  dpct::device_ext &dev_ct1 = dpct::get_current_device();
//CHECK:  sycl::queue &q_ct1 = dev_ct1.default_queue();
//CHECK:    int          num_segments = 10;
//CHECK:    int          *device_offsets;
//CHECK:    int          *device_in;
//CHECK:    int          *device_out;
//CHECK:    UserMin      min_op;
//CHECK:    int          initial_value = INT_MAX;

//CHECK:    int expect[DATA_NUM] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};

//CHECK:    device_offsets = sycl::malloc_shared<int>((num_segments + 1), q_ct1);
//CHECK:    device_in = sycl::malloc_shared<int>(DATA_NUM, q_ct1);
//CHECK:    device_out = sycl::malloc_shared<int>(num_segments, q_ct1);
//CHECK:    init_data(device_in, DATA_NUM);
//CHECK:    for(int i = 0; i < num_segments + 1; i++) {
//CHECK:      device_offsets[i] = i * 10;
//CHECK:    }


//CHECK:    /*
//CHECK:    DPCT1091:{{[0-9]+}}: The function dpct::segmented_reduce only supports DPC++ native binary operation. Replace "dpct_placeholder" with a DPC++ native binary operation.
//CHECK:    */
//CHECK:    /*
//CHECK:    DPCT1092:{{[0-9]+}}: Consider replacing work-group size 128 with differnt value for specific hardware for better performance.
//CHECK:    */
//CHECK:    dpct::device::segmented_reduce<128>(q_ct1, device_in, device_out, num_segments, (unsigned int *)(device_offsets), (unsigned int *)(device_offsets + 1), dpct_placeholder, initial_value);


//CHECK:    dev_ct1.queues_wait_and_throw();

//CHECK:    if(!verify_data(device_out, expect, num_segments)) {
//CHECK:      std::cout << "Reduce" << " verify failed" << std::endl;
//CHECK:      std::cout << "expect:" << std::endl;
//CHECK:      print_data<int>(expect, num_segments);
//CHECK:      std::cout << "current result:" << std::endl;
//CHECK:      print_data<int>(device_out, num_segments);
//CHECK:      return false;
//CHECK:    }
//CHECK:    return true;
//CHECK:  }
bool test_reduce_1(){
  int          num_segments = 10;
  int          *device_offsets;
  int          *device_in;
  int          *device_out;
  UserMin      min_op;
  int          initial_value = INT_MAX;
  void     *temp_storage = NULL;
  size_t   temp_storage_size = 0;
  int expect[DATA_NUM] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};

  cudaMallocManaged(&device_offsets, (num_segments + 1) * sizeof(int));
  cudaMallocManaged(&device_in, DATA_NUM * sizeof(int));
  cudaMallocManaged(&device_out, num_segments * sizeof(int));
  init_data(device_in, DATA_NUM);
  for(int i = 0; i < num_segments + 1; i++) {
    device_offsets[i] = i * 10;
  }
  cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1, min_op, initial_value);

  cudaMalloc(&temp_storage, temp_storage_size);

  cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1, min_op, initial_value);

  cudaDeviceSynchronize();

  if(!verify_data(device_out, expect, num_segments)) {
    std::cout << "Reduce" << " verify failed" << std::endl;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect, num_segments);
    std::cout << "current result:" << std::endl;
    print_data<int>(device_out, num_segments);
    return false;
  }
  return true;
}

//CHECK:  bool test_reduce_2(){
//CHECK:  dpct::device_ext &dev_ct1 = dpct::get_current_device();
//CHECK:  sycl::queue &q_ct1 = dev_ct1.default_queue();
//CHECK:    int          num_segments = 10;
//CHECK:    int          *device_offsets;
//CHECK:    int          *device_in;
//CHECK:    int          *device_out;
//CHECK:    UserMin      min_op;
//CHECK:    int          initial_value = INT_MAX;
//CHECK:    int expect[DATA_NUM] = {0, 2147483647, 20, 30, 40, 50, 60, 70, 80, 90};

//CHECK:    device_offsets = sycl::malloc_shared<int>((num_segments + 1), q_ct1);
//CHECK:    device_in = sycl::malloc_shared<int>(DATA_NUM, q_ct1);
//CHECK:    device_out = sycl::malloc_shared<int>(num_segments, q_ct1);
//CHECK:    init_data(device_in, DATA_NUM);
//CHECK:    for(int i = 0; i < num_segments + 1; i++) {
//CHECK:      device_offsets[i] = i * 10;
//CHECK:    }

//CHECK:    /*
//CHECK:    DPCT1092:{{[0-9]+}}: Consider replacing work-group size 128 with differnt value for specific hardware for better performance.
//CHECK:    */
//CHECK:    dpct::device::segmented_reduce<128>(q_ct1, device_in, device_out, num_segments, (unsigned int *)(device_offsets), (unsigned int *)(device_offsets + 1), sycl::minimum<>(), initial_value);

//CHECK:    dev_ct1.queues_wait_and_throw();

//CHECK:    if(!verify_data(device_out, expect, num_segments)) {
//CHECK:      std::cout << "Reduce" << " verify failed" << std::endl;
//CHECK:      std::cout << "expect:" << std::endl;
//CHECK:      print_data<int>(expect, num_segments);
//CHECK:      std::cout << "current result:" << std::endl;
//CHECK:      print_data<int>(device_out, num_segments);
//CHECK:      return false;
//CHECK:    }
//CHECK:    return true;
//CHECK:  }
bool test_reduce_2(){
  int          num_segments = 10;
  int          *device_offsets;
  int          *device_in;
  int          *device_out;
  UserMin      min_op;
  int          initial_value = INT_MAX;
  void     *temp_storage = NULL;
  size_t   temp_storage_size = 0;
  int expect[DATA_NUM] = {0, 2147483647, 20, 30, 40, 50, 60, 70, 80, 90};

  cudaMallocManaged(&device_offsets, (num_segments + 1) * sizeof(int));
  cudaMallocManaged(&device_in, DATA_NUM * sizeof(int));
  cudaMallocManaged(&device_out, num_segments * sizeof(int));
  init_data(device_in, DATA_NUM);
  for(int i = 0; i < num_segments + 1; i++) {
    device_offsets[i] = i * 10;
  }

  cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1, cub::Min(), initial_value);

  cudaMalloc(&temp_storage, temp_storage_size);

  cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1, cub::Min(), initial_value);

  cudaDeviceSynchronize();

  if(!verify_data(device_out, expect, num_segments)) {
    std::cout << "Reduce" << " verify failed" << std::endl;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect, num_segments);
    std::cout << "current result:" << std::endl;
    print_data<int>(device_out, num_segments);
    return false;
  }
  return true;
}

bool test_reduce_3(){
  int          num_segments = 10;
  int          *device_offsets;
  int          *device_in;
  int          *device_out;
  UserMin      min_op;
  int          initial_value = INT_MAX;
  // CHECK: void     *temp_storage = NULL;
  // CHECK: size_t   temp_storage_size = 0;
  void     *temp_storage = NULL;
  size_t   temp_storage_size = 0;
  int expect[DATA_NUM] = {0, 2147483647, 20, 30, 40, 50, 60, 70, 80, 90};

  cudaMallocManaged(&device_offsets, (num_segments + 1) * sizeof(int));
  cudaMallocManaged(&device_in, DATA_NUM * sizeof(int));
  cudaMallocManaged(&device_out, num_segments * sizeof(int));
  init_data(device_in, DATA_NUM);
  for(int i = 0; i < num_segments + 1; i++) {
    device_offsets[i] = i * 10;
  }
  // case 1:
  // CHECK: for(int i = 0; i < 10; i++) {
  // CHECK:   /*
  // CHECK:   DPCT1092:{{[0-9]+}}: Consider replacing work-group size 128 with differnt value for specific hardware for better performance.
  // CHECK:   */
  // CHECK:   dpct::device::segmented_reduce<128>(q_ct1, device_in, device_out, num_segments, (unsigned int *)(device_offsets), (unsigned int *)(device_offsets + 1), sycl::minimum<>(), initial_value);
  // CHECK:   temp_storage = (void *)sycl::malloc_device(temp_storage_size, q_ct1);
  // CHECK: }
  for(int i = 0; i < 10; i++) {
    cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1, cub::Min(), initial_value);

    cudaMalloc(&temp_storage, temp_storage_size);
  }

  // case 2:
  // CHECK:  for(int i = 0; i < 10; i++) {
  // CHECK:    temp_storage = nullptr;
  // CHECK:    temp_storage = (void *)sycl::malloc_device(temp_storage_size, q_ct1);
  // CHECK:    /*
  // CHECK:    DPCT1092:{{[0-9]+}}: Consider replacing work-group size 128 with differnt value for specific hardware for better performance.
  // CHECK:    */
  // CHECK:    dpct::device::segmented_reduce<128>(q_ct1, device_in, device_out, num_segments, (unsigned int *)(device_offsets), (unsigned int *)(device_offsets + 1), sycl::minimum<>(), initial_value);
  // CHECK:  }
  for(int i = 0; i < 10; i++) {
    temp_storage = nullptr;

    cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1, cub::Min(), initial_value);
    
    cudaMalloc(&temp_storage, temp_storage_size);

    cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1, cub::Min(), initial_value);

  }

  // case 3:
  // CHECK:  for(int i = 0; i < 10; i++) {
  // CHECK:    temp_storage = nullptr;
  // CHECK:    temp_storage = (void *)sycl::malloc_device(temp_storage_size, q_ct1);
  // CHECK:    /*
  // CHECK:    DPCT1092:{{[0-9]+}}: Consider replacing work-group size 128 with differnt value for specific hardware for better performance.
  // CHECK:    */
  // CHECK:    dpct::device::segmented_reduce<128>(q_ct1, device_in, device_out, num_segments, (unsigned int *)(device_offsets), (unsigned int *)(device_offsets + 1), sycl::minimum<>(), initial_value);
  // CHECK:    temp_storage = (void *)sycl::malloc_device(temp_storage_size, q_ct1);
  // CHECK:    /*
  // CHECK:    DPCT1092:{{[0-9]+}}: Consider replacing work-group size 128 with differnt value for specific hardware for better performance.
  // CHECK:    */
  // CHECK:    dpct::device::segmented_reduce<128>(q_ct1, device_in, device_out, num_segments, (unsigned int *)(device_offsets), (unsigned int *)(device_offsets + 1), sycl::minimum<>(), initial_value);
  // CHECK:  }
  for(int i = 0; i < 10; i++) {
    temp_storage = nullptr;

    cudaMalloc(&temp_storage, temp_storage_size);

    cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1, cub::Min(), initial_value);
    
    cudaMalloc(&temp_storage, temp_storage_size);

    cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1, cub::Min(), initial_value);

  }

  // CHECK: temp_storage = (void *)sycl::malloc_device(temp_storage_size, q_ct1);
  // CHECK: /*
  // CHECK: DPCT1092:{{[0-9]+}}: Consider replacing work-group size 128 with differnt value for specific hardware for better performance.
  // CHECK: */
  // CHECK: dpct::device::segmented_reduce<128>(q_ct1, device_in, device_out, num_segments, (unsigned int *)(device_offsets), (unsigned int *)(device_offsets + 1), sycl::minimum<>(), initial_value);

  cudaMalloc(&temp_storage, temp_storage_size);

  cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1, cub::Min(), initial_value);

  cudaDeviceSynchronize();

  if(!verify_data(device_out, expect, num_segments)) {
    std::cout << "Reduce" << " verify failed" << std::endl;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect, num_segments);
    std::cout << "current result:" << std::endl;
    print_data<int>(device_out, num_segments);
    return false;
  }
  return true;
}

//CHECK:  bool test_sum_1(){
//CHECK:    dpct::device_ext &dev_ct1 = dpct::get_current_device();
//CHECK:    sycl::queue &q_ct1 = dev_ct1.default_queue();
//CHECK:      int          num_segments = 10;
//CHECK:      int          *device_offsets;
//CHECK:      int          *device_in;
//CHECK:      int          *device_out;


//CHECK:      int expect[DATA_NUM] = {45, 145, 245, 345, 445, 545, 645, 745, 845, 945};

//CHECK:      device_offsets = sycl::malloc_shared<int>((num_segments + 1), q_ct1);
//CHECK:      device_in = sycl::malloc_shared<int>(DATA_NUM, q_ct1);
//CHECK:      device_out = sycl::malloc_shared<int>(num_segments, q_ct1);
//CHECK:      init_data(device_in, DATA_NUM);
//CHECK:      for(int i = 0; i < num_segments + 1; i++) {
//CHECK:        device_offsets[i] = i * 10;
//CHECK:      }

//CHECK:      /*
//CHECK:      DPCT1092:{{[0-9]+}}: Consider replacing work-group size 128 with differnt value for specific hardware for better performance.
//CHECK:      */
//CHECK:      dpct::device::segmented_reduce<128>(q_ct1, device_in, device_out, num_segments, (unsigned int *)(device_offsets), (unsigned int *)(device_offsets + 1), sycl::plus<>(), 0);

//CHECK:      dev_ct1.queues_wait_and_throw();

//CHECK:      if(!verify_data(device_out, expect, num_segments)) {
//CHECK:        std::cout << "Sum" << " verify failed" << std::endl;
//CHECK:        std::cout << "expect:" << std::endl;
//CHECK:        print_data<int>(expect, num_segments);
//CHECK:        std::cout << "current result:" << std::endl;
//CHECK:        print_data<int>(device_out, num_segments);
//CHECK:        return false;
//CHECK:      }
//CHECK:      return true;
//CHECK:    }

bool test_sum_1(){
  int          num_segments = 10;
  int          *device_offsets;
  int          *device_in;
  int          *device_out;
  void     *temp_storage = NULL;
  size_t   temp_storage_size = 0;
  int expect[DATA_NUM] = {45, 145, 245, 345, 445, 545, 645, 745, 845, 945};

  cudaMallocManaged(&device_offsets, (num_segments + 1) * sizeof(int));
  cudaMallocManaged(&device_in, DATA_NUM * sizeof(int));
  cudaMallocManaged(&device_out, num_segments * sizeof(int));
  init_data(device_in, DATA_NUM);
  for(int i = 0; i < num_segments + 1; i++) {
    device_offsets[i] = i * 10;
  }

  cub::DeviceSegmentedReduce::Sum(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1);

  cudaMalloc(&temp_storage, temp_storage_size);

  cub::DeviceSegmentedReduce::Sum(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1);

  cudaDeviceSynchronize();

  if(!verify_data(device_out, expect, num_segments)) {
    std::cout << "Sum" << " verify failed" << std::endl;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect, num_segments);
    std::cout << "current result:" << std::endl;
    print_data<int>(device_out, num_segments);
    return false;
  }
  return true;
}

//CHECK:  bool test_sum_2(){
//CHECK:  dpct::device_ext &dev_ct1 = dpct::get_current_device();
//CHECK:  sycl::queue &q_ct1 = dev_ct1.default_queue();
//CHECK:    int          num_segments = 10;
//CHECK:    int          *device_offsets;
//CHECK:    int          *device_in;
//CHECK:    int          *device_out;

//CHECK:    int expect[DATA_NUM] = {190, 0, 245, 345, 445, 545, 645, 745, 845, 945};

//CHECK:    device_offsets = sycl::malloc_shared<int>((num_segments + 1), q_ct1);
//CHECK:    device_in = sycl::malloc_shared<int>(DATA_NUM, q_ct1);
//CHECK:    device_out = sycl::malloc_shared<int>(num_segments, q_ct1);
//CHECK:    init_data(device_in, DATA_NUM);
//CHECK:    for(int i = 0; i < num_segments + 1; i++) {
//CHECK:      device_offsets[i] = i * 10;
//CHECK:    }
//CHECK:    device_offsets[1] = 20;

//CHECK:    /*
//CHECK:    DPCT1092:{{[0-9]+}}: Consider replacing work-group size 128 with differnt value for specific hardware for better performance.
//CHECK:    */
//CHECK:    dpct::device::segmented_reduce<128>(q_ct1, device_in, device_out, num_segments, (unsigned int *)(device_offsets), (unsigned int *)(device_offsets + 1), sycl::plus<>(), 0);

//CHECK:    dev_ct1.queues_wait_and_throw();

//CHECK:    if(!verify_data(device_out, expect, num_segments)) {
//CHECK:      std::cout << "Sum" << " verify failed" << std::endl;
//CHECK:      std::cout << "expect:" << std::endl;
//CHECK:      print_data<int>(expect, num_segments);
//CHECK:      std::cout << "current result:" << std::endl;
//CHECK:      print_data<int>(device_out, num_segments);
//CHECK:      return false;
//CHECK:    }
//CHECK:    return true;
//CHECK:  }

bool test_sum_2(){
  int          num_segments = 10;
  int          *device_offsets;
  int          *device_in;
  int          *device_out;
  void     *temp_storage = NULL;
  size_t   temp_storage_size = 0;
  int expect[DATA_NUM] = {190, 0, 245, 345, 445, 545, 645, 745, 845, 945};

  cudaMallocManaged(&device_offsets, (num_segments + 1) * sizeof(int));
  cudaMallocManaged(&device_in, DATA_NUM * sizeof(int));
  cudaMallocManaged(&device_out, num_segments * sizeof(int));
  init_data(device_in, DATA_NUM);
  for(int i = 0; i < num_segments + 1; i++) {
    device_offsets[i] = i * 10;
  }
  device_offsets[1] = 20;
  cub::DeviceSegmentedReduce::Sum(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1);

  cudaMalloc(&temp_storage, temp_storage_size);

  cub::DeviceSegmentedReduce::Sum(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1);

  cudaDeviceSynchronize();

  if(!verify_data(device_out, expect, num_segments)) {
    std::cout << "Sum" << " verify failed" << std::endl;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect, num_segments);
    std::cout << "current result:" << std::endl;
    print_data<int>(device_out, num_segments);
    return false;
  }
  return true;
}

//CHECK:  bool test_min(){
//CHECK:  dpct::device_ext &dev_ct1 = dpct::get_current_device();
//CHECK:  sycl::queue &q_ct1 = dev_ct1.default_queue();
//CHECK:    int          num_segments = 10;
//CHECK:    int          *device_offsets;
//CHECK:    int          *device_in;
//CHECK:    int          *device_out;

//CHECK:    int expect[DATA_NUM] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};

//CHECK:    device_offsets = sycl::malloc_shared<int>((num_segments + 1), q_ct1);
//CHECK:    device_in = sycl::malloc_shared<int>(DATA_NUM, q_ct1);
//CHECK:    device_out = sycl::malloc_shared<int>(num_segments, q_ct1);
//CHECK:    init_data(device_in, DATA_NUM);
//CHECK:    for(int i = 0; i < num_segments + 1; i++) {
//CHECK:      device_offsets[i] = i * 10;
//CHECK:    }

//CHECK:    /*
//CHECK:    DPCT1092:{{[0-9]+}}: Consider replacing work-group size 128 with differnt value for specific hardware for better performance.
//CHECK:    */
//CHECK:    dpct::device::segmented_reduce<128>(q_ct1, device_in, device_out, num_segments, (unsigned int *)(device_offsets), (unsigned int *)(device_offsets + 1), sycl::minimum<>(), std::numeric_limits<int>::max());

//CHECK:    dev_ct1.queues_wait_and_throw();

//CHECK:    if(!verify_data(device_out, expect, num_segments)) {
//CHECK:      std::cout << "Min" << " verify failed" << std::endl;
//CHECK:      std::cout << "expect:" << std::endl;
//CHECK:      print_data<int>(expect, num_segments);
//CHECK:      std::cout << "current result:" << std::endl;
//CHECK:      print_data<int>(device_out, num_segments);
//CHECK:      return false;
//CHECK:    }
//CHECK:    return true;
//CHECK:  }

bool test_min(){
  int          num_segments = 10;
  int          *device_offsets;
  int          *device_in;
  int          *device_out;
  void     *temp_storage = NULL;
  size_t   temp_storage_size = 0;
  int expect[DATA_NUM] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};

  cudaMallocManaged(&device_offsets, (num_segments + 1) * sizeof(int));
  cudaMallocManaged(&device_in, DATA_NUM * sizeof(int));
  cudaMallocManaged(&device_out, num_segments * sizeof(int));
  init_data(device_in, DATA_NUM);
  for(int i = 0; i < num_segments + 1; i++) {
    device_offsets[i] = i * 10;
  }

  cub::DeviceSegmentedReduce::Min(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1);

  cudaMalloc(&temp_storage, temp_storage_size);

  cub::DeviceSegmentedReduce::Min(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1);

  cudaDeviceSynchronize();

  if(!verify_data(device_out, expect, num_segments)) {
    std::cout << "Min" << " verify failed" << std::endl;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect, num_segments);
    std::cout << "current result:" << std::endl;
    print_data<int>(device_out, num_segments);
    return false;
  }
  return true;
}

//CHECK:  bool test_max(){
//CHECK:  dpct::device_ext &dev_ct1 = dpct::get_current_device();
//CHECK:  sycl::queue &q_ct1 = dev_ct1.default_queue();
//CHECK:    int          num_segments = 10;
//CHECK:    int          *device_offsets;
//CHECK:    int          *device_in;
//CHECK:    int          *device_out;

//CHECK:    int expect[DATA_NUM] = {9, 19, 29, 39, 49, 59, 69, 79, 89, 99};

//CHECK:    device_offsets = sycl::malloc_shared<int>((num_segments + 1), q_ct1);
//CHECK:    device_in = sycl::malloc_shared<int>(DATA_NUM, q_ct1);
//CHECK:    device_out = sycl::malloc_shared<int>(num_segments, q_ct1);
//CHECK:    init_data(device_in, DATA_NUM);
//CHECK:    for(int i = 0; i < num_segments + 1; i++) {
//CHECK:      device_offsets[i] = i * 10;
//CHECK:    }

//CHECK:    /*
//CHECK:    DPCT1092:{{[0-9]+}}: Consider replacing work-group size 128 with differnt value for specific hardware for better performance.
//CHECK:    */
//CHECK:    dpct::device::segmented_reduce<128>(q_ct1, device_in, device_out, num_segments, (unsigned int *)(device_offsets), (unsigned int *)(device_offsets + 1), sycl::maximum<>(), std::numeric_limits<int>::lowest());

//CHECK:    dev_ct1.queues_wait_and_throw();

//CHECK:    if(!verify_data(device_out, expect, num_segments)) {
//CHECK:      std::cout << "Max" << " verify failed" << std::endl;
//CHECK:      std::cout << "expect:" << std::endl;
//CHECK:      print_data<int>(expect, num_segments);
//CHECK:      std::cout << "current result:" << std::endl;
//CHECK:      print_data<int>(device_out, num_segments);
//CHECK:      return false;
//CHECK:    }
//CHECK:    return true;
//CHECK:  }

bool test_max(){
  int          num_segments = 10;
  int          *device_offsets;
  int          *device_in;
  int          *device_out;
  void     *temp_storage = NULL;
  size_t   temp_storage_size = 0;
  int expect[DATA_NUM] = {9, 19, 29, 39, 49, 59, 69, 79, 89, 99};

  cudaMallocManaged(&device_offsets, (num_segments + 1) * sizeof(int));
  cudaMallocManaged(&device_in, DATA_NUM * sizeof(int));
  cudaMallocManaged(&device_out, num_segments * sizeof(int));
  init_data(device_in, DATA_NUM);
  for(int i = 0; i < num_segments + 1; i++) {
    device_offsets[i] = i * 10;
  }

  cub::DeviceSegmentedReduce::Max(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1);

  cudaMalloc(&temp_storage, temp_storage_size);

  cub::DeviceSegmentedReduce::Max(temp_storage, temp_storage_size, device_in, device_out, num_segments, device_offsets, device_offsets + 1);

  cudaDeviceSynchronize();

  if(!verify_data(device_out, expect, num_segments)) {
    std::cout << "Max" << " verify failed" << std::endl;
    std::cout << "expect:" << std::endl;
    print_data<int>(expect, num_segments);
    std::cout << "current result:" << std::endl;
    print_data<int>(device_out, num_segments);
    return false;
  }
  return true;
}

int main() {
  bool Result = true;
  Result = test_reduce_1() && Result;
  Result = test_reduce_2() && Result;
  Result = test_reduce_3() && Result;
  Result = test_sum_1() && Result;
  Result = test_sum_2() && Result;
  Result = test_min() && Result;
  Result = test_max() && Result;
  if(Result) {
    std::cout << "Pass" << std::endl;
  }
  return 0;
}
