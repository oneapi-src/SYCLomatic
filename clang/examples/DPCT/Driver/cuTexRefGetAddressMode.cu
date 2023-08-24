void test(CUaddress_mode *pa, int i) {
  // Start
  CUtexref t;
  cuTexRefGetAddressMode(pa /*CUaddress_mode **/, t /*CUtexref*/, i /*int*/);
  // End
}
