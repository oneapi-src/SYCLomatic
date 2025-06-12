#include <nvshmem.h>

__host__ void test() {
  // Start
  nvshmem_team_t team;
  nvshmem_team_destroy(team /*nvshmem_team_t*/);
  // End
}
