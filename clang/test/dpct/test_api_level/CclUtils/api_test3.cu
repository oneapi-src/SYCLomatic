// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/CclUtils/apt_test3_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/CclUtils/apt_test3_out/MainSourceFiles.yaml | wc -l > %T/CclUtils/apt_test3_out/count.txt
// RUN: FileCheck --input-file %T/CclUtils/apt_test3_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/CclUtils/apt_test3_out

// CHECK: 35
// TEST_FEATURE: CclUtils_create_kvs
// TEST_FEATURE: CclUtils_get_kvs_detail
// TEST_FEATURE: CclUtils_ccl_init_helper
// TEST_FEATURE: CclUtils_typedef_comm_ptr
// TEST_FEATURE: CclUtils_communicator_wrapper
// TEST_FEATURE: CclUtils_communicator_wrapper_allreduce
// TEST_FEATURE: Device_get_default_context
// TEST_FEATURE: CclUtils_communicator_wrapper_size
// TEST_FEATURE: CclUtils_communicator_wrapper_rank
// TEST_FEATURE: CclUtils_communicator_wrapper_reduce_scatter
// TEST_FEATURE: CclUtils_communicator_ext_allgather
// TEST_FEATURE: CclUtils_communicator_wrapper_reduce
// TEST_FEATURE: CclUtils_communicator_wrapper_broadcast

#include <nccl.h>

int main() {
  ncclUniqueId Id;
  ncclComm_t Comm;
  int Rank;
  int size;
  void *buff;
  void * recvbuff;
  size_t count;
  cudaStream_t stream;
  ncclCommInitRank(&Comm, Rank, Id, Rank);
  ncclAllReduce(buff, recvbuff, count, ncclChar, ncclSum, comm, stream);
  ncclCommCount(Comm, &size);
  ncclCommUserRank(Comm, &Rank);
  ncclReduceScatter(buff, recvbuff, count, ncclChar, ncclSum, comm, stream);
  ncclAllGather(buff, recvbuff, count, ncclChar, comm, stream);
  ncclBroadcast(buff, recvbuff, count, ncclChar, rank, comm, stream);
  ncclReduce(buff, recvbuff, count, ncclChar, ncclSum, rank, comm, stream);
  return 0;
}
