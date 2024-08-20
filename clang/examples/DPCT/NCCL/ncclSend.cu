#include <nccl.h>

void test(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer,
          ncclComm_t comm, cudaStream_t stream) {
  // Start
  ncclSend(sendbuff /*const void **/, count /*size_t*/,
           datatype /*ncclDataType_t*/, peer /*int*/, comm /*ncclComm_t*/,
           stream /*cudaStream_t*/);
  // End
}
