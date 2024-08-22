void test(CUuuid *pu, CUdevice d) {
  // Start
  cuDeviceGetUuid_v2(pu /*CUuuid **/, d /*CUdevice*/);
  // End
}
