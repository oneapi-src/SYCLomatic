void test(const CUDA_ARRAY_DESCRIPTOR *pa, CUdeviceptr d, size_t s) {
  // Start
  CUtexref t;
  cuTexRefSetAddress2D(t /*CUtexref*/, pa /*const CUDA_ARRAY_DESCRIPTOR **/,
                       d /*CUdeviceptr*/, s /*size_t*/);
  // End
}
