#include <nccl.h>

void test(ncclUniqueId *uniqueId) {
  // Start
  ncclGetUniqueId(uniqueId /*ncclUniqueId **/);
  // End
}