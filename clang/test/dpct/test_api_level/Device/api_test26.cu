// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test26_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test26_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test26_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test26_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test26_out

// CHECK: 15
// TEST_FEATURE: Device_device_info_set_global_mem_size
// TEST_FEATURE: Device_device_info_set_integrated
// TEST_FEATURE: Device_device_info_set_local_mem_size
// TEST_FEATURE: Device_device_info_set_major_version
// TEST_FEATURE: Device_device_info_set_max_clock_frequency
// TEST_FEATURE: Device_device_info_set_max_compute_units
// TEST_FEATURE: Device_device_info_set_max_sub_group_size
// TEST_FEATURE: Device_device_info_set_max_work_group_size
// TEST_FEATURE: Device_device_info_set_max_work_items_per_compute_unit
// TEST_FEATURE: Device_device_info_set_minor_version
// TEST_FEATURE: Device_device_info_set_memory_clock_rate
// TEST_FEATURE: Device_device_info_set_memory_bus_width

// array type is not assignable
// WORK_AROUND_TEST_FEATURE: Device_device_info_set_name
// WORK_AROUND_TEST_FEATURE: Device_device_info_set_max_nd_range_size
// WORK_AROUND_TEST_FEATURE: Device_device_info_set_max_work_item_sizes

int main() {
  cudaDeviceProp deviceProp;
  deviceProp.totalGlobalMem = 12345;
  deviceProp.integrated = 1;
  deviceProp.sharedMemPerBlock = 1;
  deviceProp.major = 1;
  deviceProp.clockRate = 1;
  deviceProp.multiProcessorCount = 1;
  deviceProp.warpSize = 1;
  deviceProp.maxThreadsPerBlock = 1;
  deviceProp.maxThreadsPerMultiProcessor = 1;
  deviceProp.minor = 1;
  deviceProp.memoryClockRate = 1;
  deviceProp.memoryBusWidth = 1;
  return 0;
}
