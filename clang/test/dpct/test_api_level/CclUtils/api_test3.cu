// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/CclUtils/apt_test3_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/CclUtils/apt_test3_out/MainSourceFiles.yaml | wc -l > %T/CclUtils/apt_test3_out/count.txt
// RUN: FileCheck --input-file %T/CclUtils/apt_test3_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/CclUtils/apt_test3_out

// CHECK: 26
// TEST_FEATURE: CclUtils_create_kvs
// TEST_FEATURE: CclUtils_get_kvs_detail
// TEST_FEATURE: CclUtils_communicator_ext
// TEST_FEATURE: CclUtils_create_communicator_ext 
// TEST_FEATURE: CclUtils_communicator_ext_get_ccl_communicator
// TEST_FEATURE: Device_get_default_context

#include <nccl.h>

int main() {
  ncclUniqueId Id;
  ncclComm_t Comm;
  int Rank;
  void *buff;
  void * recvbuff;
  size_t count;
  cudaStream_t stream;
  ncclCommInitRank(&Comm, Rank, Id, Rank);
  ncclAllReduce(buff, recvbuff, count, ncclChar, ncclSum, comm, stream);
  return 0;
}
