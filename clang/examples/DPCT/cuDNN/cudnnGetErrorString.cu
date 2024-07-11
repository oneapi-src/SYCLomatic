#include <cudnn.h>

void test(const char *r, cudnnStatus_t s) {
  // Start
  r = cudnnGetErrorString(s /*cudnnStatus_t*/);
  // End
}