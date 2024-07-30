void test(int *pi, CUdevice d) {
  // Start
  /* 1 */ cuDeviceGetAttribute(
      pi /*int **/, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, d /*CUdevice*/);
  /* 2 */ cuDeviceGetAttribute(
      pi /*int **/, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, d /*CUdevice*/);
  /* 3 */ cuDeviceGetAttribute(
      pi /*int **/, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, d /*CUdevice*/);
  /* 4 */ cuDeviceGetAttribute(
      pi /*int **/, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, d /*CUdevice*/);
  /* 5 */ cuDeviceGetAttribute(pi /*int **/,
                               CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                               d /*CUdevice*/);
  /* 6 */ cuDeviceGetAttribute(
      pi /*int **/, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, d /*CUdevice*/);
  /* 7 */ cuDeviceGetAttribute(pi /*int **/, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                               d /*CUdevice*/);
  /* 8 */ cuDeviceGetAttribute(pi /*int **/,
                               CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
                               d /*CUdevice*/);
  /* 9 */ cuDeviceGetAttribute(pi /*int **/, CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                               d /*CUdevice*/);
  /* 10 */ cuDeviceGetAttribute(
      pi /*int **/, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, d /*CUdevice*/);
  /* 11 */ cuDeviceGetAttribute(
      pi /*int **/, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, d /*CUdevice*/);
  /* 12 */ cuDeviceGetAttribute(pi /*int **/, CU_DEVICE_ATTRIBUTE_INTEGRATED,
                                d /*CUdevice*/);
  /* 13 */ cuDeviceGetAttribute(
      pi /*int **/, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, d /*CUdevice*/);
  /* 14 */ cuDeviceGetAttribute(pi /*int **/,
                                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                d /*CUdevice*/);
  /* 15 */ cuDeviceGetAttribute(pi /*int **/,
                                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                d /*CUdevice*/);
  /* 16 */ cuDeviceGetAttribute(
      pi /*int **/, CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED,
      d /*CUdevice*/);
  // End
}
