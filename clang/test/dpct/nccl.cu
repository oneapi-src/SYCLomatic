// RUN: dpct --format-range=none -out-root %T/nccl %s --cuda-include-path="%cuda-path/include" --extra-arg="-xc"
// RUN: FileCheck %s --match-full-lines --input-file %T/nccl/nccl.dp.cpp

// CHECK: #include <dpct/ccl_utils.hpp>
#include <nccl.h>

int check(ncclResult_t);

int main() {
    int version, nranks, rank, device;
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

    // CHECK: device = comm->size();
    ncclCommCount(comm, &device);

    // CHECK: check((device = comm->size(), 0));
    check(ncclCommCount(comm, &device));

    // CHECK: device = dpct::dev_mgr::instance().get_device_id(comm->get_device().get_native());
    ncclCommCuDevice(comm, &device);

    // CHECK: check((device = dpct::dev_mgr::instance().get_device_id(comm->get_device().get_native()), 0));
    check(ncclCommCuDevice(comm, &device));

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

    // CHECK: oneapi::ccl::allreduce(buff, recvbuff, count, oneapi::ccl::datatype::int8, oneapi::ccl::reduction::sum, *comm, oneapi::ccl::create_stream(*stream));
    ncclAllReduce(buff, recvbuff, count, ncclChar, ncclSum, comm, stream);

    // CHECK:     /*
    // CHECK-NEXT: DPCT1067:4: The 'ncclAvg' parameter could not be migrated. You may need to update the code manually.
    // CHECK-NEXT: */
    ncclAllReduce(buff, recvbuff, count, ncclChar, ncclAvg, comm, stream);
}