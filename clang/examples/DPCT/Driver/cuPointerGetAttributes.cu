void test(unsigned int numAttr, CUpointer_attribute *attr,
          void **data, CUdeviceptr ptr) {
  // Start
  cuPointerGetAttributes(numAttr /*unsigned int*/,
                        attr /*CUpointer_attribute **/, data /*void ***/,
                        ptr /*CUdeviceptr*/);
  // End
}
