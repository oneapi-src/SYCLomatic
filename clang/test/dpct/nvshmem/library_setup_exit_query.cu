// REQUIRES: system-linux
// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2
// RUN: dpct --format-range=none -out-root %T/nvshmem %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/nvshmem/library_setup_exit_query.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl -DNO_BUILD_TEST %T/nvshmem/library_setup_exit_query.dp.cpp -o %T/nvshmem/library_setup_exit_query.dp.o %}
// CHECK: #include <ishmem.h>
// CHECK-NEXT: #include <ishmemx.h>
// CHECK-NEXT: #include <dpct/shmem_utils.hpp>
#include <nvshmem.h>
#include <nvshmemx.h>

int main() {
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

  // CHECK: dpct::shmemx::init_attr(dpct::shmemx::RUNTIME_MPI, &attr);
  // CHECK-NEXT: dpct::shmemx::init_attr(dpct::shmemx::RUNTIME_OPENSHMEM, &attr);
  // CHECK-NEXT: dpct::shmemx::init_attr(dpct::shmemx::RUNTIME_MPI | dpct::shmemx::RUNTIME_OPENSHMEM, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_SHMEM, &attr);
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM | NVSHMEMX_INIT_WITH_SHMEM, &attr);

  // CHECK: unsigned int rt = dpct::shmemx::RUNTIME_MPI;
  // CHECK-NEXT: rt = dpct::shmemx::RUNTIME_OPENSHMEM;
  // CHECK-NEXT: dpct::shmemx::init_attr(rt, &attr);
  unsigned int rt = NVSHMEMX_INIT_WITH_MPI_COMM;
  rt = NVSHMEMX_INIT_WITH_SHMEM;
  nvshmemx_init_attr(rt, &attr);

  // CHECK: dpct::shmemx::init_attr(0, &attr);
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

  // CHECK: ishmem_finalize();
  nvshmem_finalize();
}
