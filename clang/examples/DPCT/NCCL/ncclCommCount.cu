#include <nccl.h>

void test(const ncclComm_t comm, int *count) {
  // Start
  ncclCommCount(comm /*ncclComm_t*/, count /*int **/);
  // End
}