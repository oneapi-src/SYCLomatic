#include <nccl.h>

void test(ncclComm_t comm, ncclResult_t *r) {
  // Start
  ncclCommGetAsyncError(comm /*ncclComm_t*/, r /*ncclResult_t **/);
  // End
}
