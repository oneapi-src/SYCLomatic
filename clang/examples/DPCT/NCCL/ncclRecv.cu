#include <nccl.h>

void test(void *sendbuff, size_t count, ncclDataType_t datatype, int peer,
          ncclComm_t comm, cudaStream_t stream) {
  // Start
  ncclRecv(sendbuff /*void **/, count /*size_t*/, datatype /*ncclDataType_t*/,
           peer /*int*/, comm /*ncclComm_t*/, stream /*cudaStream_t*/);
  // End
}
