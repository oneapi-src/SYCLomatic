void test(unsigned int numAttributes, CUpointer_attribute *attributes,
          void **data, CUdeviceptr ptr) {
  // Start
  cuPointerGetAttribute(numAttributes /*unsigned int*/,
                        attributes /*CUpointer_attribute **/, data /*void ***/,
                        ptr /*CUdeviceptr*/);
  // End
}
