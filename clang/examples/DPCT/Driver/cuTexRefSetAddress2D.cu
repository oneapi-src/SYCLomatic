void test(const CUDA_ARRAY_DESCRIPTOR *pa, CUdeviceptr d, size_t s) {
  // Start
  CUtexref t;
  cuTexRefSetAddress2D(t /*CUtexref*/, pa /*size_t **/, d /*CUdeviceptr*/,
                       s /*size_t*/);
  // End
}
