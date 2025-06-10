#include <nvshmem.h>

__host__ void test(nvshmem_team_t parent_team, int start, int stride, int size,
                   const nvshmem_team_config_t *config, long config_mask,
                   nvshmem_team_t *new_team) {
  // Start
  nvshmem_team_split_strided(
      parent_team /*nvshmem_team_t*/, start /*int*/, stride /*int*/,
      size /*int*/, config /*const nvshmem_team_config_t **/,
      config_mask /*long*/, new_team /*nvshmem_team_t **/);
  // End
}
