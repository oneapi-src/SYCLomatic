void test(int value, CUdevice srcDevice, CUdevice dstDevice) {
  // Start
  CUdevice_P2PAttribute attrib;
  /* 1 */ cuDeviceGetP2PAttribute(
      &value /*int **/, CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED,
      srcDevice /*CUdevice*/, dstDevice /*CUdevice*/);
  /* 2 */ cuDeviceGetP2PAttribute(
      &value /*int **/, CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED,
      srcDevice /*CUdevice*/, dstDevice /*CUdevice*/);
  // End
}
