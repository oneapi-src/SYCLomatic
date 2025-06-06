#include <nvshmem.h>

__host__ void test(nvshmem_team_t src_team, int src_pe,
                   nvshmem_team_t dest_team) {
  // Start
  nvshmem_team_translate_pe(src_team /*nvshmem_team_t*/, src_pe /*int*/,
                            dest_team /*nvshmem_team_t*/);
  // End
}
