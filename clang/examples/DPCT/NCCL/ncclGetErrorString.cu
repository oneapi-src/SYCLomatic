#include <nccl.h>

void test(ncclResult_t r) {
  // Start
  ncclGetErrorString(r /*ncclResult_t*/);
  // End
}
