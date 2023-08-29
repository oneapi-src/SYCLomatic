void test(CUarray a, unsigned int u) {
  // Start
  CUtexref t;
  cuTexRefSetArray(t /*CUtexref*/, a /*CUarray*/, u /*unsigned int*/);
  // End
}
