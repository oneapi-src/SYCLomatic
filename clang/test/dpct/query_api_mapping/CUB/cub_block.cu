// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.2, v11.4
// UNSUPPORTED: system-windows
// clang-format off

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockExchange::BlockedToStriped | FileCheck %s -check-prefix=CHECK_BLOCKEXCHANGE_BLOCKEDTOSTRIPED
// CHECK_BLOCKEXCHANGE_BLOCKEDTOSTRIPED:  CUDA API:
// CHECK_BLOCKEXCHANGE_BLOCKEDTOSTRIPED:    __shared__ typename cub::BlockExchange<int, 128, 4>::TempStorage temp_storage;
// CHECK_BLOCKEXCHANGE_BLOCKEDTOSTRIPED:    cub::BlockExchange<int, 128, 4>(temp_storage).BlockedToStriped(thread_data/*int(&)[4]*/, thread_data/*int(&)[4]*/);
// CHECK_BLOCKEXCHANGE_BLOCKEDTOSTRIPED:  Is migrated to:
// CHECK_BLOCKEXCHANGE_BLOCKEDTOSTRIPED:    dpct::group::exchange<int, 4>(temp_storage).blocked_to_striped(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), thread_data, thread_data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockExchange::BlockedToWarpStriped | FileCheck %s -check-prefix=CHECK_BLOCKEXCHANGE_BLOCKEDTOWARPSTRIPED
// CHECK_BLOCKEXCHANGE_BLOCKEDTOWARPSTRIPED:  CUDA API:
// CHECK_BLOCKEXCHANGE_BLOCKEDTOWARPSTRIPED:    __shared__ typename cub::BlockExchange<int, 128, 4>::TempStorage temp_storage;
// CHECK_BLOCKEXCHANGE_BLOCKEDTOWARPSTRIPED:    cub::BlockExchange<int, 128, 4>(temp_storage).BlockedToWarpStriped(thread_data/*int(&)[4]*/, thread_data/*int(&)[4]*/);
// CHECK_BLOCKEXCHANGE_BLOCKEDTOWARPSTRIPED:  Is migrated to:
// CHECK_BLOCKEXCHANGE_BLOCKEDTOWARPSTRIPED:    dpct::group::exchange<int, 4>(temp_storage).blocked_to_sub_group_striped(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), thread_data, thread_data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockExchange::ScatterToBlocked | FileCheck %s -check-prefix=CHECK_BLOCKEXCHANGE_SCATTERTOBLOCKED
// CHECK_BLOCKEXCHANGE_SCATTERTOBLOCKED:  CUDA API:
// CHECK_BLOCKEXCHANGE_SCATTERTOBLOCKED:    __shared__ typename cub::BlockExchange<int, 128, 4>::TempStorage temp_storage;
// CHECK_BLOCKEXCHANGE_SCATTERTOBLOCKED:    cub::BlockExchange<int, 128, 4>(temp_storage).ScatterToBlocked(thread_data/*int(&)[4]*/, thread_rank/*int(&)[4]*/);
// CHECK_BLOCKEXCHANGE_SCATTERTOBLOCKED:  Is migrated to:
// CHECK_BLOCKEXCHANGE_SCATTERTOBLOCKED:    dpct::group::exchange<int, 4>(temp_storage).scatter_to_blocked(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), thread_data, thread_rank);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockExchange::ScatterToStriped | FileCheck %s -check-prefix=CHECK_BLOCKEXCHANGE_SCATTERTOSTRIPED
// CHECK_BLOCKEXCHANGE_SCATTERTOSTRIPED:  CUDA API:
// CHECK_BLOCKEXCHANGE_SCATTERTOSTRIPED:    __shared__ typename cub::BlockExchange<int, 128, 4>::TempStorage temp_storage;
// CHECK_BLOCKEXCHANGE_SCATTERTOSTRIPED:    cub::BlockExchange<int, 128, 4>(temp_storage).ScatterToStriped(thread_data/*int(&)[4]*/, thread_rank/*int(&)[4]*/);
// CHECK_BLOCKEXCHANGE_SCATTERTOSTRIPED:  Is migrated to:
// CHECK_BLOCKEXCHANGE_SCATTERTOSTRIPED:    dpct::group::exchange<int, 4>(temp_storage).scatter_to_striped(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), thread_data, thread_rank);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockExchange::StripedToBlocked | FileCheck %s -check-prefix=CHECK_BLOCKEXCHANGE_STRIPEDTOBLOCKED
// CHECK_BLOCKEXCHANGE_STRIPEDTOBLOCKED:  CUDA API:
// CHECK_BLOCKEXCHANGE_STRIPEDTOBLOCKED:    __shared__ typename cub::BlockExchange<int, 128, 4>::TempStorage temp_storage;
// CHECK_BLOCKEXCHANGE_STRIPEDTOBLOCKED:    cub::BlockExchange<int, 128, 4>(temp_storage).StripedToBlocked(thread_data/*int(&)[4]*/, thread_data/*int(&)[4]*/);
// CHECK_BLOCKEXCHANGE_STRIPEDTOBLOCKED:  Is migrated to:
// CHECK_BLOCKEXCHANGE_STRIPEDTOBLOCKED:    dpct::group::exchange<int, 4>(temp_storage).striped_to_blocked(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), thread_data, thread_data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockExchange::WarpStripedToBlocked | FileCheck %s -check-prefix=CHECK_BLOCKEXCHANGE_WARPSTRIPEDTOBLOCKED
// CHECK_BLOCKEXCHANGE_WARPSTRIPEDTOBLOCKED:  CUDA API:
// CHECK_BLOCKEXCHANGE_WARPSTRIPEDTOBLOCKED:    __shared__ typename cub::BlockExchange<int, 128, 4>::TempStorage temp_storage;
// CHECK_BLOCKEXCHANGE_WARPSTRIPEDTOBLOCKED:    cub::BlockExchange<int, 128, 4>(temp_storage).WarpStripedToBlocked(thread_data/*int(&)[4]*/, thread_data/*int(&)[4]*/);
// CHECK_BLOCKEXCHANGE_WARPSTRIPEDTOBLOCKED:  Is migrated to:
// CHECK_BLOCKEXCHANGE_WARPSTRIPEDTOBLOCKED:    dpct::group::exchange<int, 4>(temp_storage).sub_group_striped_to_blocked(sycl::ext::oneapi::this_work_item::get_nd_item<3>(), thread_data, thread_data);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockLoad::Load | FileCheck %s -check-prefix=CHECK_BLOCKLOAD_LOAD
// CHECK_BLOCKLOAD_LOAD:  CUDA API:
// CHECK_BLOCKLOAD_LOAD:    __shared__ typename cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_DIRECT>::TempStorage temp_storage;
// CHECK_BLOCKLOAD_LOAD:    cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_DIRECT>(temp_storage).Load(src/*int **/, thread_data/*int(&)[4]*/);
// CHECK_BLOCKLOAD_LOAD:    cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_DIRECT>(temp_storage).Load(src/*int **/, thread_data/*int(&)[4]*/, end/*int*/);
// CHECK_BLOCKLOAD_LOAD:    cub::BlockLoad<int, 128, 4, cub::BLOCK_LOAD_DIRECT>(temp_storage).Load(src/*int **/, thread_data/*int(&)[4]*/, end/*int*/, default_value/*int*/);
// CHECK_BLOCKLOAD_LOAD:  Is migrated to:
// CHECK_BLOCKLOAD_LOAD:    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
// CHECK_BLOCKLOAD_LOAD:    dpct::group::group_load<int, 4, dpct::group::group_load_algorithm::blocked>(temp_storage).load(item_ct1, src, thread_data);
// CHECK_BLOCKLOAD_LOAD:    dpct::group::group_load<int, 4, dpct::group::group_load_algorithm::blocked>(temp_storage).load(item_ct1, src, thread_data, end);
// CHECK_BLOCKLOAD_LOAD:    dpct::group::group_load<int, 4, dpct::group::group_load_algorithm::blocked>(temp_storage).load(item_ct1, src, thread_data, end, default_value);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::BlockStore::Store | FileCheck %s -check-prefix=CHECK_BLOCKSTORE_STORE
// CHECK_BLOCKSTORE_STORE:  CUDA API:
// CHECK_BLOCKSTORE_STORE:    __shared__ typename cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_DIRECT>::TempStorage temp_storage;
// CHECK_BLOCKSTORE_STORE:    cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_DIRECT>(temp_storage).Store(dst/*int **/, thread_data/*int(&)[4]*/);
// CHECK_BLOCKSTORE_STORE:    cub::BlockStore<int, 128, 4, cub::BLOCK_STORE_DIRECT>(temp_storage).Store(dst/*int **/, thread_data/*int(&)[4]*/, end/*int*/);
// CHECK_BLOCKSTORE_STORE:  Is migrated to:
// CHECK_BLOCKSTORE_STORE:    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
// CHECK_BLOCKSTORE_STORE:    dpct::group::group_store<int, 4, dpct::group::group_store_algorithm::blocked>(temp_storage).store(item_ct1, dst, thread_data);
// CHECK_BLOCKSTORE_STORE:    dpct::group::group_store<int, 4, dpct::group::group_store_algorithm::blocked>(temp_storage).store(item_ct1, dst, thread_data, end);
