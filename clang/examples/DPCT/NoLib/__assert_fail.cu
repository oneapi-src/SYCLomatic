__global__ void test(const char *msg, const char *file, unsigned line,
                     const char *func) {
  // Start
  __assert_fail(msg /*const char **/, file /*const char **/, line /*unsigned*/,
                func /*const char **/);
  // End
}
