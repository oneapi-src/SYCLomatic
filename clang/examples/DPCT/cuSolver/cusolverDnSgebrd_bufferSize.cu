#include "cusolverDn.h"

void test(cusolverDnHandle_t handle, int m, int n) {
  // Start
  int buffer_size;
  cusolverDnSgebrd_bufferSize(handle /*cusolverDnHandle_t*/, m /*int*/,
                              n /*int*/, &buffer_size /*int **/);
  // End
}
