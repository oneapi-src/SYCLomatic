// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Device/api_test26_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Device/api_test26_out/MainSourceFiles.yaml | wc -l > %T/Device/api_test26_out/count.txt
// RUN: FileCheck --input-file %T/Device/api_test26_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Device/api_test26_out

// CHECK: 17
// TEST_FEATURE: Device_device_info_get_uuid
// TEST_FEATURE: Device_device_info_set_uuid
// TEST_FEATURE: Device_device_info_set_max_work_items_per_compute_unit
// TEST_FEATURE: Device_device_info_get_max_work_items_per_compute_unit 
// TEST_FEATURE: Device_device_info_set_memory_bus_width
// TEST_FEATURE: Device_device_info_get_memory_bus_width 
// TEST_FEATURE: Device_device_info_set_memory_clock_rate
// TEST_FEATURE: Device_device_info_get_memory_clock_rate 
// TEST_FEATURE: Device_device_info_set_minor_version
// TEST_FEATURE: Device_device_info_get_minor_version 


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

  int tmp = deviceProp.memoryBusWidth;
  tmp = deviceProp.memoryClockRate;
  return 0;
}