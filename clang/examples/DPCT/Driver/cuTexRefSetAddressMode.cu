void test(int i, CUaddress_mode a) {
  // Start
  CUtexref t;
  cuTexRefSetAddressMode(t /*CUtexref*/, i /*int **/, a /*CUaddress_mode*/);
  // End
}
