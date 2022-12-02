// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test25_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test25_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test25_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test25_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test25_out

// CHECK: 17
// TEST_FEATURE: Device_device_info_get_global_mem_size
// TEST_FEATURE: Device_device_info_get_integrated
// TEST_FEATURE: Device_device_info_get_local_mem_size
// TEST_FEATURE: Device_device_info_get_major_version
// TEST_FEATURE: Device_device_info_get_max_clock_frequency
// TEST_FEATURE: Device_device_info_get_max_compute_units
// TEST_FEATURE: Device_device_info_get_max_nd_range_size
// TEST_FEATURE: Device_device_info_get_max_sub_group_size
// TEST_FEATURE: Device_device_info_get_max_work_group_size
// TEST_FEATURE: Device_device_info_get_max_work_items_per_compute_unit
// TEST_FEATURE: Device_device_info_get_name
// TEST_FEATURE: Device_device_info_get_max_work_item_sizes
// TEST_FEATURE: Device_device_info_get_memory_clock_rate
// TEST_FEATURE: Device_device_info_get_memory_bus_width

int main() {
  cudaDeviceProp deviceProp;
  int a;
  a = deviceProp.totalGlobalMem;
  a = deviceProp.integrated;
  a = deviceProp.sharedMemPerBlock;
  a = deviceProp.major;
  a = deviceProp.clockRate;
  a = deviceProp.multiProcessorCount;
  size_t *a_ptr = (size_t*)deviceProp.maxGridSize;
  a = deviceProp.warpSize;
  a = deviceProp.maxThreadsPerBlock;
  a = deviceProp.maxThreadsPerMultiProcessor;
  char *c = deviceProp.name;
  a = deviceProp.maxThreadsDim[2];
  a = deviceProp.memoryClockRate;
  a = deviceProp.memoryBusWidth;
  return 0;
}
