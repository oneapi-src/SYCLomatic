void test(int *pi, int i) {
  // Start
  // Only support migration of some cudaDeviceAttr type.
  /* 1 */ cudaDeviceGetAttribute(pi /*int **/,
                                 cudaDevAttrMaxThreadsPerBlock
                                 /*cudaDeviceAttr*/,
                                 i /*int*/);
  /* 2 */ cudaDeviceGetAttribute(pi /*int **/,
                                 cudaDevAttrClockRate
                                 /*cudaDeviceAttr*/,
                                 i /*int*/);
  /* 3 */ cudaDeviceGetAttribute(pi /*int **/,
                                 cudaDevAttrTextureAlignment
                                 /*cudaDeviceAttr*/,
                                 i /*int*/);
  /* 4 */ cudaDeviceGetAttribute(pi /*int **/,
                                 cudaDevAttrMultiProcessorCount
                                 /*cudaDeviceAttr*/,
                                 i /*int*/);
  /* 5 */ cudaDeviceGetAttribute(pi /*int **/,
                                 cudaDevAttrIntegrated
                                 /*cudaDeviceAttr*/,
                                 i /*int*/);
  /* 6 */ cudaDeviceGetAttribute(pi /*int **/,
                                 cudaDevAttrComputeMode
                                 /*cudaDeviceAttr*/,
                                 i /*int*/);
  /* 7 */ cudaDeviceGetAttribute(pi /*int **/,
                                 cudaDevAttrComputeCapabilityMajor
                                 /*cudaDeviceAttr*/,
                                 i /*int*/);
  /* 8 */ cudaDeviceGetAttribute(pi /*int **/,
                                 cudaDevAttrComputeCapabilityMinor
                                 /*cudaDeviceAttr*/,
                                 i /*int*/);
  /* 9 */ cudaDeviceGetAttribute(pi /*int **/,
                                 cudaDevAttrHostNativeAtomicSupported
                                 /*cudaDeviceAttr*/,
                                 i /*int*/);
  /* 10 */ cudaDeviceGetAttribute(pi /*int **/,
                                  cudaDevAttrConcurrentManagedAccess
                                  /*cudaDeviceAttr*/,
                                  i /*int*/);
  // End
}
