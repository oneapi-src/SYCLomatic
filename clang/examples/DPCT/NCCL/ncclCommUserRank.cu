#include <nccl.h>

void test(const ncclComm_t comm, int *rank) {
  // Start
  ncclCommUserRank(comm /*ncclComm_t*/, rank /*int **/);
  // End
}