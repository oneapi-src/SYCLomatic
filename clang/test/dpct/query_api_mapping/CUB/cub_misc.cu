// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.2, v11.4, v11.5, v11.6, v11.7, v11.8
// UNSUPPORTED: system-windows
// clang-format off

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BFE | FileCheck %s -check-prefix=CHECK_BFE
// CHECK_BFE:  CUDA API:
// CHECK_BFE:    cub::BFE(input/*int*/, bit_start/*unsigned int*/, num_bits/*unsigned int*/);
// CHECK_BFE:  Is migrated to:
// CHECK_BFE:    dpct::bfe_safe(input, bit_start, num_bits);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BFI | FileCheck %s -check-prefix=CHECK_BFI
// CHECK_BFI:  CUDA API:
// CHECK_BFI:    cub::BFI(a/*unsigned int*/, b/*unsigned int*/, c/*unsigned int*/, bit_start/*unsigned int*/, num_bits/*unsigned int*/);
// CHECK_BFI:  Is migrated to:
// CHECK_BFI:    a = dpct::bfi_safe<unsigned>(c, b, bit_start, num_bits);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::CurrentDevice | FileCheck %s -check-prefix=CHECK_CURRENTDEVICE
// CHECK_CURRENTDEVICE:  CUDA API:
// CHECK_CURRENTDEVICE:    res = cub::CurrentDevice();
// CHECK_CURRENTDEVICE:  Is migrated to:
// CHECK_CURRENTDEVICE:    res = dpct::get_current_device_id();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::Debug | FileCheck %s -check-prefix=CHECK_DEBUG
// CHECK_DEBUG:  CUDA API:
// CHECK_DEBUG:    cub::Debug(e/*cudaError_t*/, filename/*const char**/, line/*int*/);
// CHECK_DEBUG:  Is migrated to:
// CHECK_DEBUG:    DPCT_CHECK_ERROR(e);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceCount | FileCheck %s -check-prefix=CHECK_DEVICECOUNT
// CHECK_DEVICECOUNT:  CUDA API:
// CHECK_DEVICECOUNT:    res = cub::DeviceCount();
// CHECK_DEVICECOUNT:  Is migrated to:
// CHECK_DEVICECOUNT:    res = dpct::device_count();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceCountCachedValue | FileCheck %s -check-prefix=CHECK_DEVICECOUNTCACHEDVALUE
// CHECK_DEVICECOUNTCACHEDVALUE:  CUDA API:
// CHECK_DEVICECOUNTCACHEDVALUE:    res = cub::DeviceCountCachedValue();
// CHECK_DEVICECOUNTCACHEDVALUE:  Is migrated to:
// CHECK_DEVICECOUNTCACHEDVALUE:    res = dpct::device_count();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceCountUncached | FileCheck %s -check-prefix=CHECK_DEVICECOUNTUNCACHED
// CHECK_DEVICECOUNTUNCACHED:  CUDA API:
// CHECK_DEVICECOUNTUNCACHED:    res = cub::DeviceCountUncached();
// CHECK_DEVICECOUNTUNCACHED:  Is migrated to:
// CHECK_DEVICECOUNTUNCACHED:    res = dpct::device_count();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::IADD3 | FileCheck %s -check-prefix=CHECK_IADD3
// CHECK_IADD3:  CUDA API:
// CHECK_IADD3:    result = cub::IADD3(a/*unsigned int*/, b/*unsigned int*/, c/*unsigned int*/);
// CHECK_IADD3:  Is migrated to:
// CHECK_IADD3:    result = (a + b + c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::LaneId | FileCheck %s -check-prefix=CHECK_LANEID
// CHECK_LANEID:  CUDA API:
// CHECK_LANEID:    result = cub::LaneId();
// CHECK_LANEID:  Is migrated to:
// CHECK_LANEID:    result = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_sub_group().get_local_linear_id();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::LoadDirectBlocked | FileCheck %s -check-prefix=CHECK_LOADDIRECTBLOCKED
// CHECK_LOADDIRECTBLOCKED:  CUDA API:
// CHECK_LOADDIRECTBLOCKED:    cub::LoadDirectBlocked(id/*int*/, data/*int **/, thread_data/*int (&)[4]*/);
// CHECK_LOADDIRECTBLOCKED:  Is migrated to:
// CHECK_LOADDIRECTBLOCKED:    dpct::group::load_direct_blocked(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), data, thread_data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::LoadDirectStriped | FileCheck %s -check-prefix=CHECK_LOADDIRECTSTRIPED
// CHECK_LOADDIRECTSTRIPED:  CUDA API:
// CHECK_LOADDIRECTSTRIPED:    cub::LoadDirectStriped<128>(id/*int*/, data/*int **/, thread_data/*int (&)[4]*/);
// CHECK_LOADDIRECTSTRIPED:  Is migrated to:
// CHECK_LOADDIRECTSTRIPED:    dpct::group::load_direct_striped(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), data, thread_data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::LoadDirectWarpStriped | FileCheck %s -check-prefix=CHECK_LOADDIRECTWARPSTRIPED
// CHECK_LOADDIRECTWARPSTRIPED:  CUDA API:
// CHECK_LOADDIRECTWARPSTRIPED:    cub::LoadDirectWarpStriped(id/*int*/, data/*int **/, thread_data/*int (&)[4]*/);
// CHECK_LOADDIRECTWARPSTRIPED:  Is migrated to:
// CHECK_LOADDIRECTWARPSTRIPED:    dpct::group::load_direct_sub_group_striped(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), data, thread_data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::PtxVersion | FileCheck %s -check-prefix=CHECK_PTXVERSION
// CHECK_PTXVERSION:  CUDA API:
// CHECK_PTXVERSION:    cub::PtxVersion(r/*int*/);
// CHECK_PTXVERSION:  Is migrated to:
// CHECK_PTXVERSION:    r = DPCT_COMPATIBILITY_TEMP;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::PtxVersionUncached | FileCheck %s -check-prefix=CHECK_PTXVERSIONUNCACHED
// CHECK_PTXVERSIONUNCACHED:  CUDA API:
// CHECK_PTXVERSIONUNCACHED:    cub::PtxVersionUncached(r/*int*/);
// CHECK_PTXVERSIONUNCACHED:  Is migrated to:
// CHECK_PTXVERSIONUNCACHED:    r = DPCT_COMPATIBILITY_TEMP;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::RowMajorTid | FileCheck %s -check-prefix=CHECK_ROWMAJORTID
// CHECK_ROWMAJORTID:  CUDA API:
// CHECK_ROWMAJORTID:    r = cub::RowMajorTid(dim_x/*int*/, dim_y/*int*/, dim_z/*int*/);
// CHECK_ROWMAJORTID:  Is migrated to:
// CHECK_ROWMAJORTID:    r = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_linear_id();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::SHL_ADD | FileCheck %s -check-prefix=CHECK_SHL_ADD
// CHECK_SHL_ADD:  CUDA API:
// CHECK_SHL_ADD:    res = cub::SHL_ADD(a/*int*/, b/*int*/, c/*int*/);
// CHECK_SHL_ADD:  Is migrated to:
// CHECK_SHL_ADD:    res = dpct::extend_shl_clamp<uint32_t>(a, b, c, sycl::plus<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::SHR_ADD | FileCheck %s -check-prefix=CHECK_SHR_ADD
// CHECK_SHR_ADD:  CUDA API:
// CHECK_SHR_ADD:    res = cub::SHR_ADD(a/*int*/, b/*int*/, c/*int*/);
// CHECK_SHR_ADD:  Is migrated to:
// CHECK_SHR_ADD:    res = dpct::extend_shr_clamp<uint32_t>(a, b, c, sycl::plus<>());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::SmVersion | FileCheck %s -check-prefix=CHECK_SMVERSION
// CHECK_SMVERSION:  CUDA API:
// CHECK_SMVERSION:    cub::SmVersion(res/*int*/);
// CHECK_SMVERSION:  Is migrated to:
// CHECK_SMVERSION:    res = dpct::get_major_version(dpct::get_current_device()) * 100 + dpct::get_minor_version(dpct::get_current_device()) * 10;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::SmVersionUncached | FileCheck %s -check-prefix=CHECK_SMVERSIONUNCACHED
// CHECK_SMVERSIONUNCACHED:  CUDA API:
// CHECK_SMVERSIONUNCACHED:    cub::SmVersionUncached(res/*int*/);
// CHECK_SMVERSIONUNCACHED:  Is migrated to:
// CHECK_SMVERSIONUNCACHED:    res = dpct::get_major_version(dpct::get_current_device()) * 100 + dpct::get_minor_version(dpct::get_current_device()) * 10;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::StoreDirectBlocked | FileCheck %s -check-prefix=CHECK_STOREDIRECTBLOCKED
// CHECK_STOREDIRECTBLOCKED:  CUDA API:
// CHECK_STOREDIRECTBLOCKED:    cub::StoreDirectBlocked(id/*int*/, data/*int **/, thread_data/*int (&)[4]*/);
// CHECK_STOREDIRECTBLOCKED: Is migrated to:
// CHECK_STOREDIRECTBLOCKED:    dpct::group::store_direct_blocked(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), data, thread_data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::StoreDirectStriped | FileCheck %s -check-prefix=CHECK_STOREDIRECTSTRIPED
// CHECK_STOREDIRECTSTRIPED:  CUDA API:
// CHECK_STOREDIRECTSTRIPED:    cub::StoreDirectStriped<128>(id/*int*/, data/*int **/, thread_data/*int (&)[4]*/);
// CHECK_STOREDIRECTSTRIPED:  Is migrated to:
// CHECK_STOREDIRECTSTRIPED:    dpct::group::store_direct_striped(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), data, thread_data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::StoreDirectWarpStriped | FileCheck %s -check-prefix=CHECK_STOREDIRECTWARPSTRIPED
// CHECK_STOREDIRECTWARPSTRIPED:  CUDA API:
// CHECK_STOREDIRECTWARPSTRIPED:    cub::StoreDirectWarpStriped(id/*int*/, data/*int **/, thread_data/*int (&)[4]*/);
// CHECK_STOREDIRECTWARPSTRIPED:  Is migrated to:
// CHECK_STOREDIRECTWARPSTRIPED:    dpct::group::store_direct_sub_group_striped(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), data, thread_data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::SyncStream | FileCheck %s -check-prefix=CHECK_SYNCSTREAM
// CHECK_SYNCSTREAM:  CUDA API:
// CHECK_SYNCSTREAM:    cub::SyncStream(s/*cudaStream_t*/);
// CHECK_SYNCSTREAM:  Is migrated to:
// CHECK_SYNCSTREAM:    s->wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::ThreadLoad | FileCheck %s -check-prefix=CHECK_THREADLOAD
// CHECK_THREADLOAD:  CUDA API:
// CHECK_THREADLOAD:    res = cub::ThreadLoad<cub::LOAD_CA>(data/*int **/);
// CHECK_THREADLOAD:  Is migrated to:
// CHECK_THREADLOAD:    res = *(data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::ThreadStore | FileCheck %s -check-prefix=CHECK_THREADSTORE
// CHECK_THREADSTORE:  CUDA API:
// CHECK_THREADSTORE:    cub::ThreadStore<cub::STORE_CG>(dst/*int **/, data/*int*/);
// CHECK_THREADSTORE:  Is migrated to:
// CHECK_THREADSTORE:    *(dst) = data;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::WarpId | FileCheck %s -check-prefix=CHECK_WARPID
// CHECK_WARPID:  CUDA API:
// CHECK_WARPID:    res = cub::WarpId();
// CHECK_WARPID:  Is migrated to:
// CHECK_WARPID:    res = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_sub_group().get_group_linear_id();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::ArgIndexInputIterator | FileCheck %s -check-prefix=CHECK_ARGINDEXINPUTITERATOR
// CHECK_ARGINDEXINPUTITERATOR:  CUDA API:
// CHECK_ARGINDEXINPUTITERATOR:    cub::ArgIndexInputIterator<int *> Iter(d_in);
// CHECK_ARGINDEXINPUTITERATOR:  Is migrated to:
// CHECK_ARGINDEXINPUTITERATOR:    dpct::arg_index_input_iterator<int *> Iter(d_in);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::ConstantInputIterator | FileCheck %s -check-prefix=CHECK_CONSTANTINPUTITERATOR
// CHECK_CONSTANTINPUTITERATOR:  CUDA API:
// CHECK_CONSTANTINPUTITERATOR:    cub::ConstantInputIterator<int> Iter(d_in);
// CHECK_CONSTANTINPUTITERATOR:  Is migrated to:
// CHECK_CONSTANTINPUTITERATOR:    dpct::constant_iterator<int> Iter(d_in);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::CountingInputIterator | FileCheck %s -check-prefix=CHECK_COUNTINGINPUTITERATOR
// CHECK_COUNTINGINPUTITERATOR:  CUDA API:
// CHECK_COUNTINGINPUTITERATOR:    cub::CountingInputIterator<int> Iter(d_in);
// CHECK_COUNTINGINPUTITERATOR:  Is migrated to:
// CHECK_COUNTINGINPUTITERATOR:    oneapi::dpl::counting_iterator<int> Iter(d_in);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DiscardOutputIterator | FileCheck %s -check-prefix=CHECK_DISCARDOUTPUTITERATOR
// CHECK_DISCARDOUTPUTITERATOR:  CUDA API:
// CHECK_DISCARDOUTPUTITERATOR:    cub::DiscardOutputIterator<int> Iter;
// CHECK_DISCARDOUTPUTITERATOR:  Is migrated to:
// CHECK_DISCARDOUTPUTITERATOR:    oneapi::dpl::discard_iterator Iter;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::TransformInputIterator | FileCheck %s -check-prefix=CHECK_TRANSFORMINPUTITERATOR
// CHECK_TRANSFORMINPUTITERATOR:  CUDA API:
// CHECK_TRANSFORMINPUTITERATOR:    cub::TransformInputIterator<double, UserDefMul, double *> iter(d_in /*double **/, op /*Op*/);
// CHECK_TRANSFORMINPUTITERATOR:  Is migrated to:
// CHECK_TRANSFORMINPUTITERATOR:    oneapi::dpl::transform_iterator<double *, UserDefMul> iter(d_in /*double **/, op /*Op*/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::ShuffleUp | FileCheck %s -check-prefix=CHECK_SHUFFLEUP
// CHECK_SHUFFLEUP:  CUDA API:
// CHECK_SHUFFLEUP:    output /*int*/ = cub::ShuffleUp<32>(input /*int*/, src_offset /*int*/, first_thread /*int*/, member_mask /*unsigned int*/);
// CHECK_SHUFFLEUP:  Is migrated to (with the option --use-experimental-features=non-uniform-groups):
// CHECK_SHUFFLEUP:    output /*int*/ = dpct::experimental::shift_sub_group_right<32>(sycl::ext::oneapi::this_work_item::get_sub_group(), input, src_offset, first_thread, member_mask);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::ShuffleDown | FileCheck %s -check-prefix=CHECK_SHUFFLEDOWN
// CHECK_SHUFFLEDOWN:  CUDA API:
// CHECK_SHUFFLEDOWN:    output /*int*/ = cub::ShuffleDown<32>(input /*int*/, src_offset /*int*/, last_thread /*int*/, member_mask /*unsigned int*/);
// CHECK_SHUFFLEDOWN:  Is migrated to (with the option --use-experimental-features=non-uniform-groups):
// CHECK_SHUFFLEDOWN:    output /*int*/ = dpct::experimental::shift_sub_group_left<32>(sycl::ext::oneapi::this_work_item::get_sub_group(), input, src_offset, last_thread, member_mask);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::ShuffleIndex | FileCheck %s -check-prefix=CHECK_SHUFFLEINDEX
// CHECK_SHUFFLEINDEX:  CUDA API:
// CHECK_SHUFFLEINDEX:    output /*int*/ = cub::ShuffleIndex<32>(input /*int*/, src_lane /*int*/, member_mask /*unsigned int*/);
// CHECK_SHUFFLEINDEX:  Is migrated to (with the option --use-experimental-features=non-uniform-groups):
// CHECK_SHUFFLEINDEX:    output /*int*/ = dpct::experimental::select_from_sub_group(member_mask, sycl::ext::oneapi::this_work_item::get_sub_group(), input, src_lane);
