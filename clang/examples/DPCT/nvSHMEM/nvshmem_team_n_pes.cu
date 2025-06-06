#include <nvshmem.h>

__host__ void test(nvshmem_team_t team) {
  // Start
  nvshmem_team_n_pes(team /*nvshmem_team_t*/);
  // End
}
