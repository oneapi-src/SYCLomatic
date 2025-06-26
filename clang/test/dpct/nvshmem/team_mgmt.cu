// REQUIRES: system-linux
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7
// RUN: dpct --format-range=none -out-root %T/nvshmem %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/nvshmem/team_mgmt.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/nvshmem/team_mgmt.dp.cpp -o %T/nvshmem/team_mgmt.dp.o %}
#include <nvshmem.h>
#include <nvshmemx.h>

int main() {
  // CHECK: ishmem_team_t team;
  nvshmem_team_t team;

  // CHECK: ishmem_team_config_t config;
  nvshmem_team_config_t config;

  int n_pes = 2;
  int start = 0;
  int stride = 2;

  // CHECK: ishmem_team_split_strided(ISHMEM_TEAM_WORLD, start, stride, n_pes/2, NULL, 0, &team);
  nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, start, stride, n_pes/2, NULL, 0, &team);

  // CHECK: if (team != ISHMEM_TEAM_INVALID) {
  if (team != NVSHMEM_TEAM_INVALID) {
    // CHECK: int team_my_pe = ishmem_team_my_pe(team);
    // CHECK-NEXT: int team_n_pes = ishmem_team_n_pes(team);
    int team_my_pe = nvshmem_team_my_pe(team);
    int team_n_pes = nvshmem_team_n_pes(team);

    // CHECK: ishmem_team_get_config(team, 0, &config);
    nvshmem_team_get_config(team, &config);

    // CHECK: int world_pe = ishmem_team_translate_pe(team, team_my_pe, ISHMEM_TEAM_WORLD);
    int world_pe = nvshmem_team_translate_pe(team, team_my_pe, NVSHMEM_TEAM_WORLD);
  }

  int xrange = 2;
  long mask = 0;

  // CHECK: int status = ishmem_team_split_2d(ISHMEM_TEAM_WORLD, xrange, &config, mask, &team, &config, mask, &team);
  int status = nvshmem_team_split_2d(NVSHMEM_TEAM_WORLD, xrange, &config, mask, &team, &config, mask, &team);

  // CHECK: if (team != ISHMEM_TEAM_INVALID) {
  if (team != NVSHMEM_TEAM_INVALID) {
    // CHECK: ishmem_team_destroy(team);
    nvshmem_team_destroy(team);
  }
}
