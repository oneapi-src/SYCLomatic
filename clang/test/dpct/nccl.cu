// RUN: dpct --format-range=none -out-root %T/nccl %s --cuda-include-path="%cuda-path/include" --extra-arg="-xc"
// RUN: FileCheck %s --match-full-lines --input-file %T/nccl/nccl.dp.cpp

// CHECK: #include <dpct/ccl_utils.hpp>
#include <nccl.h>

int check(ncclResult_t);

int main() {
    int version, nranks, rank, count;
    // CHECK: oneapi::ccl::kvs::address_type id;
    ncclUniqueId id;
    // CHECK: oneapi::ccl::communicator * comm;
    ncclComm_t comm;

    // CHECK: version = dpct::ccl::get_version();
    ncclGetVersion(&version);

    // CHECK: check((version = dpct::ccl::get_version(), 0));
    check(ncclGetVersion(&version));

    // CHECK: id = dpct::ccl::create_kvs_address();
    ncclGetUniqueId(&id);

    // CHECK: check((id = dpct::ccl::create_kvs_address(), 0));
    check(ncclGetUniqueId(&id));

    // CHECK: comm = new oneapi::ccl::communicator(oneapi::ccl::create_communicator(nranks, rank, dpct::ccl::create_kvs(id)));
    ncclCommInitRank(&comm, nranks, id, rank);

    // CHECK: check((comm = new oneapi::ccl::communicator(oneapi::ccl::create_communicator(nranks, rank, dpct::ccl::create_kvs(id))), 0));    
    check(ncclCommInitRank(&comm, nranks, id, rank));

    // CHECK: count = comm.size();
    ncclCommCount(comm, &count);

    // CHECK: check((count = comm.size(), 0));
    check(ncclCommCount(comm, &count));

    // CHECK: count = comm.get_device();
    ncclCommCuDevice(comm, &count);

    // CHECK: check((count = comm.get_device(), 0));
    check(ncclCommCuDevice(comm, &count));

    void *buff;
    size_t count;
    ncclDataType_t datatype;
    int peer;
    cudaStream_t stream;
    // CHECK:     /*
    // CHECK-NEXT: DPCT1007:0: Migration of ncclGroupStart is not supported.
    // CHECK-NEXT: */
    ncclGroupStart();

    // CHECK:     /*
    // CHECK-NEXT: DPCT1007:1: Migration of ncclSend is not supported.
    // CHECK-NEXT: */
    ncclSend(buff, count, datatype, peer, comm, stream);

    // CHECK:     /*
    // CHECK-NEXT: DPCT1007:2: Migration of ncclRecv is not supported.
    // CHECK-NEXT: */
    ncclRecv(buff, count, datatype, peer, comm, stream);

    // CHECK:     /*
    // CHECK-NEXT: DPCT1007:3: Migration of ncclGroupEnd is not supported.
    // CHECK-NEXT: */
    ncclGroupEnd();
}