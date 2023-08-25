void test(size_t *ps, CUdeviceptr d, size_t s) {
  // Start
  CUtexref t;
  cuTexRefSetAddress(ps /*size_t **/, t /*CUtexref*/, d /*CUdeviceptr*/,
                     s /*size_t*/);
  // End
}
