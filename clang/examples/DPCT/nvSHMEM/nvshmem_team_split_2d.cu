#include <nvshmem.h>

__host__ void test(nvshmem_team_t parent_team, int xrange,
                   const nvshmem_team_config_t *xaxis_config, long xaxis_mask,
                   nvshmem_team_t *xaxis_team,
                   const nvshmem_team_config_t *yaxis_config, long yaxis_mask,
                   nvshmem_team_t *yaxis_team) {
  // Start
  nvshmem_team_split_2d(parent_team /*nvshmem_team_t*/, xrange /*int*/,
                        xaxis_config /*const nvshmem_team_config_t **/,
                        xaxis_mask /*long*/, xaxis_team /*nvshmem_team_t **/,
                        yaxis_config /*const nvshmem_team_config_t **/,
                        yaxis_mask /*long*/, yaxis_team /*nvshmem_team_t **/);
  // End
}
