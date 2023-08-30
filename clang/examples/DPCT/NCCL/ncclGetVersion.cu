#include <nccl.h>

void test(int *version) {
  // Start
  ncclGetVersion(version /*int **/);
  // End
}