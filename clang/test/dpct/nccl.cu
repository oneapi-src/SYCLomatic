// RUN: dpct --format-range=none -out-root %T/nccl %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/nccl/nccl.dp.cpp

// CHECK: #include <dpct/ccl_utils.hpp>
#include <nccl.h>

int check(ncclResult_t);

int main() {
    int version, nranks, rank, device, rank_rev;
    // CHECK: oneapi::ccl::kvs::address_type id;
    ncclUniqueId id;
    // CHECK: dpct::ccl::comm_ptr comm;
    ncclComm_t comm;

    // CHECK: version = dpct::ccl::get_version();
    ncclGetVersion(&version);

    // CHECK: check(CHECK_SYCL_ERROR(version = dpct::ccl::get_version()));
    check(ncclGetVersion(&version));

    // CHECK: id = dpct::ccl::create_kvs_address();
    ncclGetUniqueId(&id);

    // CHECK: check(CHECK_SYCL_ERROR(id = dpct::ccl::create_kvs_address()));
    check(ncclGetUniqueId(&id));

    // CHECK: comm = new dpct::ccl::communicator_wrapper(nranks, rank, id);
    ncclCommInitRank(&comm, nranks, id, rank);

    // CHECK: check(CHECK_SYCL_ERROR(comm = new dpct::ccl::communicator_wrapper(nranks, rank, id)));    
    check(ncclCommInitRank(&comm, nranks, id, rank));

    // CHECK: device = comm->size();
    ncclCommCount(comm, &device);

    // CHECK: check(CHECK_SYCL_ERROR(device = comm->size()));
    check(ncclCommCount(comm, &device));

    // CHECK: device = dpct::get_device_id(comm->get_device());
    ncclCommCuDevice(comm, &device);

    // CHECK: check(CHECK_SYCL_ERROR(device = dpct::get_device_id(comm->get_device())));
    check(ncclCommCuDevice(comm, &device));

    // CHECK: rank_rev = comm->rank();
    ncclCommUserRank(comm, &rank_rev);

    // CHECK: check(CHECK_SYCL_ERROR(rank_rev = comm->rank()));
    check(ncclCommUserRank(comm, &rank_rev));

    void *buff;
    void * recvbuff;
    size_t count;
    // CHECK: oneapi::ccl::reduction op = oneapi::ccl::reduction::sum;
    ncclRedOp_t op = ncclSum;
    // CHECK: op = oneapi::ccl::reduction::prod;
    op = ncclProd;
    // CHECK: op = oneapi::ccl::reduction::min;
    op = ncclMin;
    // CHECK: op = oneapi::ccl::reduction::max;
    op = ncclMax;
    // CHECK: oneapi::ccl::datatype datatype = oneapi::ccl::datatype::int8;
    ncclDataType_t datatype = ncclChar;
    // CHECK: datatype = oneapi::ccl::datatype::int8;
    datatype = ncclChar;
    // CHECK: datatype = oneapi::ccl::datatype::uint8;
    datatype = ncclUint8;
    // CHECK: datatype = oneapi::ccl::datatype::int32;
    datatype = ncclInt32;
    // CHECK: datatype = oneapi::ccl::datatype::int32;
    datatype = ncclInt;
    // CHECK: datatype = oneapi::ccl::datatype::uint32;
    datatype = ncclUint32;
    // CHECK: datatype = oneapi::ccl::datatype::int64;
    datatype = ncclInt64;
    // CHECK: datatype = oneapi::ccl::datatype::uint64;
    datatype = ncclUint64;
    // CHECK: datatype = oneapi::ccl::datatype::float16;
    datatype = ncclFloat16;
    // CHECK: datatype = oneapi::ccl::datatype::float16;
    datatype = ncclHalf;
    // CHECK: datatype = oneapi::ccl::datatype::float32;
    datatype = ncclFloat32;
    // CHECK: datatype = oneapi::ccl::datatype::float32;
    datatype = ncclFloat;
    // CHECK: datatype = oneapi::ccl::datatype::float64;
    datatype = ncclFloat64;
    // CHECK: datatype = oneapi::ccl::datatype::float64;
    datatype = ncclDouble;
    int peer;
    cudaStream_t stream;
    // CHECK:     /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of ncclGroupStart is not supported.
    // CHECK-NEXT: */
    ncclGroupStart();

    // CHECK:     /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of ncclSend is not supported.
    // CHECK-NEXT: */
    ncclSend(buff, count, datatype, peer, comm, stream);

    // CHECK:     /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of ncclRecv is not supported.
    // CHECK-NEXT: */
    ncclRecv(buff, count, datatype, peer, comm, stream);

    // CHECK:     /*
    // CHECK-NEXT: DPCT1007:{{[0-9]+}}: Migration of ncclGroupEnd is not supported.
    // CHECK-NEXT: */
    ncclGroupEnd();

    // CHECK: comm->allreduce(buff, recvbuff, count, oneapi::ccl::datatype::int8, oneapi::ccl::reduction::sum, stream);
    ncclAllReduce(buff, recvbuff, count, ncclChar, ncclSum, comm, stream);

    // CHECK:     /*
    // CHECK-NEXT: DPCT1067:{{[0-9]+}}: The 'ncclAvg' parameter could not be migrated. You may need to update the code manually.
    // CHECK-NEXT: */
    ncclAllReduce(buff, recvbuff, count, ncclChar, ncclAvg, comm, stream);

    // CHECK: comm->reduce(buff, recvbuff, count, oneapi::ccl::datatype::int8, oneapi::ccl::reduction::sum, rank, stream);
    ncclReduce(buff, recvbuff, count, ncclChar, ncclSum, rank, comm, stream);

    // CHECK:     /*
    // CHECK-NEXT: DPCT1067:{{[0-9]+}}: The 'ncclAvg' parameter could not be migrated. You may need to update the code manually.
    // CHECK-NEXT: */
    ncclReduce(buff, recvbuff, count, ncclChar, ncclAvg, rank, comm, stream);

    
    // CHECK: comm->broadcast(buff, recvbuff, count, oneapi::ccl::datatype::int8, rank, stream);
    ncclBroadcast(buff, recvbuff, count, ncclChar, rank, comm, stream);

    // CHECK: comm->reduce_scatter(buff, recvbuff, count, oneapi::ccl::datatype::int8, oneapi::ccl::reduction::sum, stream);
    ncclReduceScatter(buff, recvbuff, count, ncclChar, ncclSum, comm, stream);

    // CHECK: /*
    // CHECK-NEXT: DPCT1067:{{[0-9]+}}: The 'ncclAvg' parameter could not be migrated. You may need to update the code manually.
    // CHECK-NEXT: */
    ncclReduceScatter(buff, recvbuff, count, ncclChar, ncclAvg, comm, stream);

    // CHECK: delete comm;
    ncclCommDestroy(comm);
}