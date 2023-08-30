void test(CUarray_format a, int i) {
  // Start
  CUtexref t;
  cuTexRefSetFormat(t /*CUtexref*/, a /*CUarray_format*/, i /*int*/);
  // End
}
