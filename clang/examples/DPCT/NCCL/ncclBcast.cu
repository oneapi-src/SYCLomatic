#include <nccl.h>

void test(void *buff, size_t count, ncclDataType_t datatype, int root,
          ncclComm_t comm, cudaStream_t stream) {
  // Start
  ncclBcast(buff /*void **/, count /*size_t*/, datatype /*ncclDataType_t*/,
                root /*int*/, comm /*ncclComm_t*/, stream /*cudaStream_t*/);
  // End
}