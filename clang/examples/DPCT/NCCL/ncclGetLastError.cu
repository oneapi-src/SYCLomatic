#include <nccl.h>

void test(ncclComm_t comm) {
  // Start
  ncclGetLastError(comm /*ncclComm_t*/);
  // End
}
