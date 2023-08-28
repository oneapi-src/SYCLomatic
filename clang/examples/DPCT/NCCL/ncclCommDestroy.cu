#include <nccl.h>

void test(ncclComm_t comm) {
  // Start
  ncclCommDestroy(comm /*ncclComm_t*/);
  // End
}