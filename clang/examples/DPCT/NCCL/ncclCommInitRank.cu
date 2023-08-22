#include <nccl.h>

void test(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank) {
  // Start
  ncclCommInitRank(comm /*ncclComm_t **/, nranks /*int*/,
                  commId /*ncclUniqueId*/, rank /*int*/);
  // End
}