// REQUIRES: system-linux
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1
// RUN: dpct --format-range=none -out-root %T/nvshmem %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/nvshmem/nvshmem.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/nvshmem/nvshmem.dp.cpp -o %T/nvshmem/nvshmem.dp.o %}
#include <nvshmem.h>
#include <nvshmemx.h>

int main() {
    // CHECK: ishmem_team_t team;
    nvshmem_team_t team;

    // CHECK: ishmem_team_config_t config;
    nvshmem_team_config_t config;

    // CHECK: ishmemx_attr_t attr;
    nvshmemx_init_attr_t attr;
#ifndef NO_BUILD_TEST
    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of nvshmemx_init_args_t is not supported.
    // CHECK-NEXT: */
    nvshmemx_init_args_t args;
    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of nvshmemx_uniqueid_t is not supported.
    // CHECK-NEXT: */
    nvshmemx_uniqueid_t uid;
    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of nvshmemx_uniqueid_args_t is not supported.
    // CHECK-NEXT: */
    nvshmemx_uniqueid_args_t uid_args;

    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of nvshmemx_uniqueid_t::internal is not supported.
    // CHECK-NEXT: */
    char *internal = uid.internal;

    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of nvshmemx_uniqueid_args_t::id is not supported.
    // CHECK-NEXT: */
    uid_args.id = &uid;
    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of nvshmemx_uniqueid_args_t::myrank is not supported.
    // CHECK-NEXT: */
    uid_args.myrank = 0;
    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of nvshmemx_uniqueid_args_t::nranks is not supported.
    // CHECK-NEXT: */
    uid_args.nranks = 0;

    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of nvshmemx_init_args_t::uid_args is not supported.
    // CHECK-NEXT: */
    args.uid_args = uid_args;
    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of nvshmemx_init_args_t::content is not supported.
    // CHECK-NEXT: */
    char *content = args.content;

    attr.mpi_comm = NULL;
    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of nvshmemx_init_attr_t::args is not supported.
    // CHECK-NEXT: */
    attr.args = args;
#endif // NO_BUILD_TEST

    // CHECK: (&attr)->runtime = ISHMEMX_RUNTIME_MPI;
    // CHECK-NEXT: ishmemx_init_attr(&attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

#ifndef NO_BUILD_TEST
    // CHECK: /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of nvshmemx_init_attr is not supported.
    // CHECK-NEXT: */
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM | NVSHMEMX_INIT_WITH_SHMEM, &attr);
#endif // NO_BUILD_TEST

    // CHECK: unsigned int rt = ISHMEMX_RUNTIME_MPI;
    // CHECK-NEXT: (&attr)->runtime = static_cast<ishmemx_runtime_type_t>(rt);
    // CHECK-NEXT: ishmemx_init_attr(&attr);
    unsigned int rt = NVSHMEMX_INIT_WITH_MPI_COMM;
    nvshmemx_init_attr(rt, &attr);

    // CHECK: rt = ISHMEMX_RUNTIME_OPENSHMEM;
    // CHECK-NEXT: (&attr)->runtime = static_cast<ishmemx_runtime_type_t>(rt);
    // CHECK-NEXT: ishmemx_init_attr(&attr);
    rt = NVSHMEMX_INIT_WITH_SHMEM;
    nvshmemx_init_attr(rt, &attr);

    // CHECK: ishmem_init();
    nvshmemx_init_attr(0, &attr);

    // CHECK: ishmem_init();
    nvshmem_init();

    // CHECK: int my_pe = ishmem_my_pe();
    // CHECK-NEXT: int n_pes = ishmem_n_pes();
    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();

    int major, minor;
    // CHECK: ishmem_info_get_version(&major, &minor);
    nvshmem_info_get_version(&major, &minor);

    char* name;
    // CHECK: ishmem_info_get_name(name);
    nvshmem_info_get_name(name);

    size_t size = 1024;
    // CHECK: int* array = (int*)ishmem_malloc(size * sizeof(int));
    // CHECK-NEXT: array = (int*)ishmem_align(64, size * sizeof(int));
    // CHECK-NEXT: array = (int*)ishmem_calloc(size, sizeof(int));
    // CHECK-NEXT: void* symmetric_ptr = ishmem_ptr(array, my_pe);
    int* array = (int*)nvshmem_malloc(size * sizeof(int));
    array = (int*)nvshmem_align(64, size * sizeof(int));
    array = (int*)nvshmem_calloc(size, sizeof(int));
    void* symmetric_ptr = nvshmem_ptr(array, my_pe);

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

    // CHECK: ishmem_free(array);
    nvshmem_free(array);

    // CHECK: ishmem_finalize();
    nvshmem_finalize();
}
