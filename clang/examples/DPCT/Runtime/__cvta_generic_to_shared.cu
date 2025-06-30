__device__ void test(const void *ptr) {
  // Start
  size_t addr = __cvta_generic_to_shared(ptr /*const void **/);
  // End
  uint32_t val;
  asm volatile("{ld.shared.b32 %0, [%1];}" : : "r"(val), "r"(addr) : "memory");
}
