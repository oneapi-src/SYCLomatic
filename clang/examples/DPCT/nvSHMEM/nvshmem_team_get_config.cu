#include <nvshmem.h>

__host__ void test(nvshmem_team_t team) {
  // Start
  nvshmem_team_config_t *config;
  nvshmem_team_get_config(team /*nvshmem_team_t*/,
                          config /*nvshmem_team_config_t **/);
  // End
}
