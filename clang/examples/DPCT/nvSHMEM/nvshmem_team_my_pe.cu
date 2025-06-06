#include <nvshmem.h>

__host__ void test(nvshmem_team_t team) {
  // Start
  nvshmem_team_my_pe(team /*nvshmem_team_t*/);
  // End
}
