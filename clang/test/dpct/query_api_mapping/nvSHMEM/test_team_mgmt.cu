// REQUIRES: system-linux
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_team_destroy | FileCheck %s -check-prefix=NVSHMEM_TEAM_DESTROY
// NVSHMEM_TEAM_DESTROY: CUDA API:
// NVSHMEM_TEAM_DESTROY-NEXT:   nvshmem_team_t team;
// NVSHMEM_TEAM_DESTROY-NEXT:   nvshmem_team_destroy(team /*nvshmem_team_t*/);
// NVSHMEM_TEAM_DESTROY-NEXT: Is migrated to:
// NVSHMEM_TEAM_DESTROY-NEXT:   ishmem_team_t team;
// NVSHMEM_TEAM_DESTROY-NEXT:   ishmem_team_destroy(team);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_team_get_config | FileCheck %s -check-prefix=NVSHMEM_TEAM_GET_CONFIG
// NVSHMEM_TEAM_GET_CONFIG: CUDA API:
// NVSHMEM_TEAM_GET_CONFIG-NEXT:   nvshmem_team_config_t *config;
// NVSHMEM_TEAM_GET_CONFIG-NEXT:   nvshmem_team_get_config(team /*nvshmem_team_t*/,
// NVSHMEM_TEAM_GET_CONFIG-NEXT:                           config /*nvshmem_team_config_t **/);
// NVSHMEM_TEAM_GET_CONFIG-NEXT: Is migrated to:
// NVSHMEM_TEAM_GET_CONFIG-NEXT:   ishmem_team_config_t *config;
// NVSHMEM_TEAM_GET_CONFIG-NEXT:   ishmem_team_get_config(team, 0, config);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_team_my_pe | FileCheck %s -check-prefix=NVSHMEM_TEAM_MY_PE
// NVSHMEM_TEAM_MY_PE: CUDA API:
// NVSHMEM_TEAM_MY_PE-NEXT:   nvshmem_team_my_pe(team /*nvshmem_team_t*/);
// NVSHMEM_TEAM_MY_PE-NEXT: Is migrated to:
// NVSHMEM_TEAM_MY_PE-NEXT:   ishmem_team_my_pe(team);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_team_n_pes | FileCheck %s -check-prefix=NVSHMEM_TEAM_N_PES
// NVSHMEM_TEAM_N_PES: CUDA API:
// NVSHMEM_TEAM_N_PES-NEXT:   nvshmem_team_n_pes(team /*nvshmem_team_t*/);
// NVSHMEM_TEAM_N_PES-NEXT: Is migrated to:
// NVSHMEM_TEAM_N_PES-NEXT:   ishmem_team_n_pes(team);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_team_split_2d | FileCheck %s -check-prefix=NVSHMEM_TEAM_SPLIT_2D
// NVSHMEM_TEAM_SPLIT_2D: CUDA API:
// NVSHMEM_TEAM_SPLIT_2D-NEXT:   nvshmem_team_split_2d(parent_team /*nvshmem_team_t*/, xrange /*int*/,
// NVSHMEM_TEAM_SPLIT_2D-NEXT:                         xaxis_config /*const nvshmem_team_config_t **/,
// NVSHMEM_TEAM_SPLIT_2D-NEXT:                         xaxis_mask /*long*/, xaxis_team /*nvshmem_team_t **/,
// NVSHMEM_TEAM_SPLIT_2D-NEXT:                         yaxis_config /*const nvshmem_team_config_t **/,
// NVSHMEM_TEAM_SPLIT_2D-NEXT:                         yaxis_mask /*long*/, yaxis_team /*nvshmem_team_t **/);
// NVSHMEM_TEAM_SPLIT_2D-NEXT: Is migrated to:
// NVSHMEM_TEAM_SPLIT_2D-NEXT:   ishmem_team_split_2d(parent_team, xrange, xaxis_config, xaxis_mask, xaxis_team, yaxis_config, yaxis_mask, yaxis_team);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_team_split_strided | FileCheck %s -check-prefix=NVSHMEM_TEAM_SPLIT_STRIDED
// NVSHMEM_TEAM_SPLIT_STRIDED: CUDA API:
// NVSHMEM_TEAM_SPLIT_STRIDED-NEXT:   nvshmem_team_split_strided(
// NVSHMEM_TEAM_SPLIT_STRIDED-NEXT:       parent_team /*nvshmem_team_t*/, start /*int*/, stride /*int*/,
// NVSHMEM_TEAM_SPLIT_STRIDED-NEXT:       size /*int*/, config /*const nvshmem_team_config_t **/,
// NVSHMEM_TEAM_SPLIT_STRIDED-NEXT:       config_mask /*long*/, new_team /*nvshmem_team_t **/);
// NVSHMEM_TEAM_SPLIT_STRIDED-NEXT: Is migrated to:
// NVSHMEM_TEAM_SPLIT_STRIDED-NEXT:   ishmem_team_split_strided(parent_team, start, stride, size, config, config_mask, new_team);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=nvshmem_team_translate_pe | FileCheck %s -check-prefix=NVSHMEM_TEAM_TRANSLATE_PE
// NVSHMEM_TEAM_TRANSLATE_PE: CUDA API:
// NVSHMEM_TEAM_TRANSLATE_PE-NEXT:   nvshmem_team_translate_pe(src_team /*nvshmem_team_t*/, src_pe /*int*/,
// NVSHMEM_TEAM_TRANSLATE_PE-NEXT:                             dest_team /*nvshmem_team_t*/);
// NVSHMEM_TEAM_TRANSLATE_PE-NEXT: Is migrated to:
// NVSHMEM_TEAM_TRANSLATE_PE-NEXT:   ishmem_team_translate_pe(src_team, src_pe, dest_team);
