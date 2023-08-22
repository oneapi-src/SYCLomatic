#include <nccl.h>

void test(const ncclComm_t comm, int *device) {
  // Start
  ncclCommCuDevice(comm /*ncclComm_t*/, device /*int **/);
  // End
}