__global__ void test(const char *msg, const char *file, unsigned line,
                     const char *func, size_t charSize) {
  // Start
  __assertfail(msg /*const char **/, file /*const char **/, line /*unsigned*/,
               func /*const char **/, charSize /*size_t*/);
  // End
}
