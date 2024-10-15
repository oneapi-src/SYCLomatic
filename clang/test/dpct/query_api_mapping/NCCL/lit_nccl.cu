/// Communicator Creation and Management Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ncclGetLastError | FileCheck %s -check-prefix=NCCLGETLASTERROR
// NCCLGETLASTERROR: CUDA API:
// NCCLGETLASTERROR-NEXT:   ncclGetLastError(comm /*ncclComm_t*/);
// NCCLGETLASTERROR-NEXT: The API is Removed.
// NCCLGETLASTERROR-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ncclGetErrorString | FileCheck %s -check-prefix=NCCLGETERRORSTRING
// NCCLGETERRORSTRING: CUDA API:
// NCCLGETERRORSTRING-NEXT:   ncclGetErrorString(r /*ncclResult_t*/);
// NCCLGETERRORSTRING-NEXT: Is migrated to:
// NCCLGETERRORSTRING-NEXT:   dpct::get_error_dummy(r);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ncclGetVersion | FileCheck %s -check-prefix=ncclGetVersion
// ncclGetVersion: CUDA API:
// ncclGetVersion-NEXT:   ncclGetVersion(version /*int **/);
// ncclGetVersion-NEXT: Is migrated to:
// ncclGetVersion-NEXT:   *version = dpct::ccl::get_version();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ncclGetUniqueId | FileCheck %s -check-prefix=ncclGetUniqueId
// ncclGetUniqueId: CUDA API:
// ncclGetUniqueId-NEXT:   ncclGetUniqueId(uniqueId /*ncclUniqueId **/);
// ncclGetUniqueId-NEXT: Is migrated to:
// ncclGetUniqueId-NEXT:   *uniqueId = dpct::ccl::create_kvs_address();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ncclCommInitRank | FileCheck %s -check-prefix=ncclCommInitRank
// ncclCommInitRank: CUDA API:
// ncclCommInitRank-NEXT:   ncclCommInitRank(comm /*ncclComm_t **/, nranks /*int*/,
// ncclCommInitRank-NEXT:                   commId /*ncclUniqueId*/, rank /*int*/);
// ncclCommInitRank-NEXT: Is migrated to:
// ncclCommInitRank-NEXT:   *comm = new dpct::ccl::communicator_wrapper(nranks, rank, commId);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ncclCommDestroy | FileCheck %s -check-prefix=ncclCommDestroy
// ncclCommDestroy: CUDA API:
// ncclCommDestroy-NEXT:   ncclCommDestroy(comm /*ncclComm_t*/);
// ncclCommDestroy-NEXT: Is migrated to:
// ncclCommDestroy-NEXT:   delete comm;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ncclCommGetAsyncError | FileCheck %s -check-prefix=NCCLCOMMGETASYNCERROR
// NCCLCOMMGETASYNCERROR: CUDA API:
// NCCLCOMMGETASYNCERROR-NEXT:   ncclCommGetAsyncError(comm /*ncclComm_t*/, r /*ncclResult_t **/);
// NCCLCOMMGETASYNCERROR-NEXT: The API is Removed.
// NCCLCOMMGETASYNCERROR-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ncclCommCount | FileCheck %s -check-prefix=ncclCommCount
// ncclCommCount: CUDA API:
// ncclCommCount-NEXT:    ncclCommCount(comm /*ncclComm_t*/, count /*int **/);
// ncclCommCount-NEXT: Is migrated to:
// ncclCommCount-NEXT:   *count = comm->size();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ncclCommCuDevice | FileCheck %s -check-prefix=ncclCommCuDevice
// ncclCommCuDevice: CUDA API:
// ncclCommCuDevice-NEXT:   ncclCommCuDevice(comm /*ncclComm_t*/, device /*int **/);
// ncclCommCuDevice-NEXT: Is migrated to:
// ncclCommCuDevice-NEXT:   *device = dpct::get_device_id(comm->get_device());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ncclCommUserRank | FileCheck %s -check-prefix=ncclCommUserRank
// ncclCommUserRank: CUDA API:
// ncclCommUserRank-NEXT:   ncclCommUserRank(comm /*ncclComm_t*/, rank /*int **/);
// ncclCommUserRank-NEXT: Is migrated to:
// ncclCommUserRank-NEXT:   *rank = comm->rank();

/// Collective Communication Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ncclAllReduce | FileCheck %s -check-prefix=ncclAllReduce
// ncclAllReduce: CUDA API:
// ncclAllReduce-NEXT:   ncclAllReduce(sendbuff /*void **/, recvbuff /*void **/, count /*size_t*/,
// ncclAllReduce-NEXT:                 datatype /*ncclDataType_t*/, op /*ncclRedOp_t*/,
// ncclAllReduce-NEXT:                 comm /*ncclComm_t*/, stream /*cudaStream_t*/);
// ncclAllReduce-NEXT: Is migrated to:
// ncclAllReduce-NEXT:   comm->allreduce(sendbuff, recvbuff, count, datatype, op, stream);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ncclBroadcast | FileCheck %s -check-prefix=ncclBroadcast
// ncclBroadcast: CUDA API:
// ncclBroadcast-NEXT:   ncclBroadcast(sendbuff /*void **/, recvbuff /*void **/, count /*size_t*/,
// ncclBroadcast-NEXT:             datatype /*ncclDataType_t*/, root /*int*/, comm /*ncclComm_t*/,
// ncclBroadcast-NEXT:             stream /*cudaStream_t*/);
// ncclBroadcast-NEXT: Is migrated to:
// ncclBroadcast-NEXT:   comm->broadcast(sendbuff, recvbuff, count, datatype, root, stream);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ncclBcast | FileCheck %s -check-prefix=ncclBcast
// ncclBcast: CUDA API:
// ncclBcast-NEXT:   ncclBcast(buff /*void **/, count /*size_t*/, datatype /*ncclDataType_t*/,
// ncclBcast-NEXT:                 root /*int*/, comm /*ncclComm_t*/, stream /*cudaStream_t*/);
// ncclBcast-NEXT: Is migrated to:
// ncclBcast-NEXT:   comm->broadcast(buff, buff, count, datatype, root, stream);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ncclReduce | FileCheck %s -check-prefix=ncclReduce
// ncclReduce: CUDA API:
// ncclReduce-NEXT:   ncclReduce(sendbuff /*void **/, recvbuff /*void **/, count /*size_t*/,
// ncclReduce-NEXT:             datatype /*ncclDataType_t*/, op /*ncclRedOp_t*/, root /*int*/,
// ncclReduce-NEXT:             comm /*ncclComm_t*/, stream /*cudaStream_t*/);
// ncclReduce-NEXT: Is migrated to:
// ncclReduce-NEXT:   comm->reduce(sendbuff, recvbuff, count, datatype, op, root, stream);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ncclReduceScatter | FileCheck %s -check-prefix=ncclReduceScatter
// ncclReduceScatter: CUDA API:
// ncclReduceScatter-NEXT:   ncclReduceScatter(sendbuff /*void **/, recvbuff /*void **/,
// ncclReduceScatter-NEXT:                     recvcount /*size_t*/, datatype /*ncclDataType_t*/,
// ncclReduceScatter-NEXT:                     op /*ncclRedOp_t*/, comm /*ncclComm_t*/,
// ncclReduceScatter-NEXT:                     stream /*cudaStream_t*/);
// ncclReduceScatter-NEXT: Is migrated to:
// ncclReduceScatter-NEXT:   comm->reduce_scatter(sendbuff, recvbuff, recvcount, datatype, op, stream);

/// Point To Point Communication Functions

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ncclSend | FileCheck %s -check-prefix=NCCLSEND
// NCCLSEND: CUDA API:
// NCCLSEND-NEXT:   ncclSend(sendbuff /*const void **/, count /*size_t*/,
// NCCLSEND-NEXT:            datatype /*ncclDataType_t*/, peer /*int*/, comm /*ncclComm_t*/,
// NCCLSEND-NEXT:            stream /*cudaStream_t*/);
// NCCLSEND-NEXT: Is migrated to:
// NCCLSEND-NEXT:   comm->send(sendbuff, count, datatype, peer, stream);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=ncclRecv | FileCheck %s -check-prefix=NCCLRECV
// NCCLRECV: CUDA API:
// NCCLRECV-NEXT:   ncclRecv(sendbuff /*void **/, count /*size_t*/, datatype /*ncclDataType_t*/,
// NCCLRECV-NEXT:            peer /*int*/, comm /*ncclComm_t*/, stream /*cudaStream_t*/);
// NCCLRECV-NEXT: Is migrated to:
// NCCLRECV-NEXT:   comm->recv(sendbuff, count, datatype, peer, stream);
