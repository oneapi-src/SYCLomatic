// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.4
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.2, v11.4
// clang-format off
// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceReduce::ArgMax | FileCheck %s -check-prefix=CHECK_REDUCE_ARG_MAX
// CHECK_REDUCE_ARG_MAX: CUDA API:
// CHECK_REDUCE_ARG_MAX-NEXT:   cudaStream_t stream;
// CHECK_REDUCE_ARG_MAX-NEXT:   cudaStreamCreate(&stream);
// CHECK_REDUCE_ARG_MAX-NEXT:  cub::DeviceReduce::ArgMax(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, num_items/*int*/, stream/*cudaStream_t*/);
// CHECK_REDUCE_ARG_MAX-NEXT: Is migrated to:
// CHECK_REDUCE_ARG_MAX-NEXT:  dpct::queue_ptr stream;
// CHECK_REDUCE_ARG_MAX-NEXT:  stream = dpct::get_current_device().create_queue();
// CHECK_REDUCE_ARG_MAX-NEXT:  dpct::reduce_argmax(oneapi::dpl::execution::device_policy(*stream), d_in, d_out, num_items);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceReduce::ArgMin | FileCheck %s -check-prefix=CHECK_REDUCE_ARG_MIN
// CHECK_REDUCE_ARG_MIN: CUDA API:
// CHECK_REDUCE_ARG_MIN-NEXT:   cudaStream_t stream;
// CHECK_REDUCE_ARG_MIN-NEXT:   cudaStreamCreate(&stream);
// CHECK_REDUCE_ARG_MIN-NEXT:   cub::DeviceReduce::ArgMin(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, num_items/*int*/, stream/*cudaStream_t*/);
// CHECK_REDUCE_ARG_MIN-NEXT: Is migrated to:
// CHECK_REDUCE_ARG_MIN-NEXT:   dpct::queue_ptr stream;
// CHECK_REDUCE_ARG_MIN-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_REDUCE_ARG_MIN-NEXT:   dpct::reduce_argmin(oneapi::dpl::execution::device_policy(*stream), d_in, d_out, num_items);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceReduce::Max | FileCheck %s -check-prefix=CHECK_REDUCE_MAX
// CHECK_REDUCE_MAX: CUDA API:
// CHECK_REDUCE_MAX-NEXT:   cudaStream_t stream;
// CHECK_REDUCE_MAX-NEXT:   cudaStreamCreate(&stream);
// CHECK_REDUCE_MAX-NEXT:   cub::DeviceReduce::Max(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, num_items/*int*/, stream/*cudaStream_t*/);
// CHECK_REDUCE_MAX-NEXT: Is migrated to:
// CHECK_REDUCE_MAX-NEXT:   dpct::queue_ptr stream;
// CHECK_REDUCE_MAX-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_REDUCE_MAX-NEXT:   stream->fill(d_out, oneapi::dpl::reduce(oneapi::dpl::execution::device_policy(*stream), d_in, d_in + num_items, typename std::iterator_traits<decltype(d_out)>::value_type{}, sycl::maximum<>()), 1).wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceReduce::Min | FileCheck %s -check-prefix=CHECK_REDUCE_MIN
// CHECK_REDUCE_MIN: CUDA API:
// CHECK_REDUCE_MIN-NEXT:   cudaStream_t stream;
// CHECK_REDUCE_MIN-NEXT:   cudaStreamCreate(&stream);
// CHECK_REDUCE_MIN-NEXT:   cub::DeviceReduce::Min(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, num_items/*int*/, stream/*cudaStream_t*/);
// CHECK_REDUCE_MIN-NEXT: Is migrated to:
// CHECK_REDUCE_MIN-NEXT:   dpct::queue_ptr stream;
// CHECK_REDUCE_MIN-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_REDUCE_MIN-NEXT:   stream->fill(d_out, oneapi::dpl::reduce(oneapi::dpl::execution::device_policy(*stream), d_in, d_in + num_items, typename std::iterator_traits<decltype(d_out)>::value_type{}, sycl::minimum<>()), 1).wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceReduce::Reduce | FileCheck %s -check-prefix=CHECK_REDUCE
// CHECK_REDUCE: CUDA API:
// CHECK_REDUCE-NEXT:   cudaStream_t stream;
// CHECK_REDUCE-NEXT:   cudaStreamCreate(&stream);
// CHECK_REDUCE-NEXT:   cub::DeviceReduce::Reduce(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, num_items/*int*/, op/*ReductionOpT*/, init_value/*T*/, stream/*cudaStream_t*/);
// CHECK_REDUCE-NEXT: Is migrated to:
// CHECK_REDUCE-NEXT:   dpct::queue_ptr stream;
// CHECK_REDUCE-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_REDUCE-NEXT:   stream->fill(d_out, oneapi::dpl::reduce(oneapi::dpl::execution::device_policy(*stream), d_in, d_in + num_items, init_value, op), 1).wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceReduce::ReduceByKey | FileCheck %s -check-prefix=CHECK_REDUCE_BY_KEY
// CHECK_REDUCE_BY_KEY: CUDA API:
// CHECK_REDUCE_BY_KEY-NEXT:   cudaStream_t stream;
// CHECK_REDUCE_BY_KEY-NEXT:   cudaStreamCreate(&stream);
// CHECK_REDUCE_BY_KEY-NEXT:   cub::DeviceReduce::ReduceByKey(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_keys_in/*KeysInputIteratorT*/, d_unique_out/*UniqueOutputIteratorT*/, d_values_in/*ValuesInputIteratorT*/, d_aggregates_out/*AggregatesOutputIteratorT*/, d_num_runs_out/*NumRunsOutputIteratorT*/, op/*ReductionOpT*/, num_items/*int*/, stream/*cudaStream_t*/);
// CHECK_REDUCE_BY_KEY-NEXT: Is migrated to:
// CHECK_REDUCE_BY_KEY-NEXT:   dpct::queue_ptr stream;
// CHECK_REDUCE_BY_KEY-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_REDUCE_BY_KEY-NEXT:   stream->fill(d_num_runs_out, std::distance(d_unique_out, oneapi::dpl::reduce_by_key(oneapi::dpl::execution::device_policy(*stream), d_keys_in, d_keys_in + num_items, d_values_in, d_unique_out, d_aggregates_out, std::equal_to<typename std::iterator_traits<decltype(d_keys_in)>::value_type>(), op).first), 1).wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceReduce::Sum | FileCheck %s -check-prefix=CHECK_REDUCE_SUM
// CHECK_REDUCE_SUM: CUDA API:
// CHECK_REDUCE_SUM-NEXT:   cudaStream_t stream;
// CHECK_REDUCE_SUM-NEXT:   cudaStreamCreate(&stream);
// CHECK_REDUCE_SUM-NEXT:   cub::DeviceReduce::Sum(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, num_items/*int*/, stream/*cudaStream_t*/);
// CHECK_REDUCE_SUM-NEXT: Is migrated to:
// CHECK_REDUCE_SUM-NEXT:   dpct::queue_ptr stream;
// CHECK_REDUCE_SUM-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_REDUCE_SUM-NEXT:   stream->fill(d_out, oneapi::dpl::reduce(oneapi::dpl::execution::device_policy(*stream), d_in, d_in + num_items, typename std::iterator_traits<decltype(d_out)>::value_type{}), 1).wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceRunLengthEncode::Encode | FileCheck %s -check-prefix=CHECK_ENCODE
// CHECK_ENCODE: CUDA API:
// CHECK_ENCODE-NEXT:   cudaStream_t stream;
// CHECK_ENCODE-NEXT:   cudaStreamCreate(&stream);
// CHECK_ENCODE-NEXT:   cub::DeviceRunLengthEncode::Encode(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_unique_out/*UniqueOutputIteratorT*/, d_counts_out/*LengthsOutputIteratorT*/, d_num_runs_out/*NumRunsOutputIteratorT*/, num_items/*int*/, stream/*cudaStream_t*/);
// CHECK_ENCODE-NEXT: Is migrated to:
// CHECK_ENCODE-NEXT:   dpct::queue_ptr stream;
// CHECK_ENCODE-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_ENCODE-NEXT:   stream->fill(d_num_runs_out, std::distance(d_unique_out, oneapi::dpl::reduce_by_segment(oneapi::dpl::execution::device_policy(*stream), d_in, d_in + num_items, dpct::constant_iterator<size_t>(1), d_unique_out, d_counts_out).first), 1).wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceRunLengthEncode::NoTrivialRuns | FileCheck %s -check-prefix=CHECK_NO_TRIVIAL_RUNS
// CHECK_NO_TRIVIAL_RUNS: CUDA API:
// CHECK_NO_TRIVIAL_RUNS-NEXT:   cudaStream_t stream;
// CHECK_NO_TRIVIAL_RUNS-NEXT:   cudaStreamCreate(&stream);
// CHECK_NO_TRIVIAL_RUNS-NEXT:   cub::DeviceRunLengthEncode::NonTrivialRuns(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_offsets_out/*OffsetsOutputIteratorT*/, d_lengths_out/*LengthsOutputIteratorT*/, d_num_runs_out/*NumRunsOutputIteratorT*/, num_items/*int*/, stream/*cudaStream_t*/);
// CHECK_NO_TRIVIAL_RUNS-NEXT: Is migrated to:
// CHECK_NO_TRIVIAL_RUNS-NEXT:   dpct::queue_ptr stream;
// CHECK_NO_TRIVIAL_RUNS-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_NO_TRIVIAL_RUNS-NEXT:   dpct::nontrivial_run_length_encode(oneapi::dpl::execution::device_policy(*stream), d_in, d_offsets_out, d_lengths_out, d_num_runs_out, num_items);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceScan::ExclusiveScan | FileCheck %s -check-prefix=CHECK_EXCLUSIVE_SCAN
// CHECK_EXCLUSIVE_SCAN: CUDA API:
// CHECK_EXCLUSIVE_SCAN-NEXT:   cudaStream_t stream;
// CHECK_EXCLUSIVE_SCAN-NEXT:   cudaStreamCreate(&stream);
// CHECK_EXCLUSIVE_SCAN-NEXT:   cub::DeviceScan::ExclusiveScan(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, scan_op/*ScanOpT*/, init_value/*InitValueT*/, num_items/*int*/, stream/*cudaStream_t*/);
// CHECK_EXCLUSIVE_SCAN-NEXT: Is migrated to:
// CHECK_EXCLUSIVE_SCAN-NEXT:   dpct::queue_ptr stream;
// CHECK_EXCLUSIVE_SCAN-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_EXCLUSIVE_SCAN-NEXT:   oneapi::dpl::exclusive_scan(oneapi::dpl::execution::device_policy(*stream), d_in, d_in + num_items, d_out, init_value, scan_op);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceScan::ExclusiveScanByKey | FileCheck %s -check-prefix=CHECK_EXCLUSIVE_SCAN_BY_KEY
// CHECK_EXCLUSIVE_SCAN_BY_KEY: CUDA API:
// CHECK_EXCLUSIVE_SCAN_BY_KEY-NEXT:   cudaStream_t stream;
// CHECK_EXCLUSIVE_SCAN_BY_KEY-NEXT:   cudaStreamCreate(&stream);
// CHECK_EXCLUSIVE_SCAN_BY_KEY-NEXT:   cub::DeviceScan::ExclusiveScanByKey(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_keys_in/*KeysInputIteratorT*/, d_values_in/*ValuesInputIteratorT*/, d_values_out/*ValuesOutputIteratorT*/, op/*ScanOpT*/, init_value/*InitValueT*/, num_items/*int*/, equality_op/*EqualityOpT*/, stream/*cudaStream_t*/);
// CHECK_EXCLUSIVE_SCAN_BY_KEY-NEXT: Is migrated to:
// CHECK_EXCLUSIVE_SCAN_BY_KEY-NEXT:   dpct::queue_ptr stream;
// CHECK_EXCLUSIVE_SCAN_BY_KEY-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_EXCLUSIVE_SCAN_BY_KEY-NEXT:   oneapi::dpl::exclusive_scan_by_key(oneapi::dpl::execution::device_policy(*stream), d_keys_in, d_keys_in + num_items, d_values_in, d_values_out, init_value, equality_op, op);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceScan::ExclusiveSum | FileCheck %s -check-prefix=CHECK_EXCLUSIVE_SUM
// CHECK_EXCLUSIVE_SUM: CUDA API:
// CHECK_EXCLUSIVE_SUM-NEXT:   cudaStream_t stream;
// CHECK_EXCLUSIVE_SUM-NEXT:   cudaStreamCreate(&stream);
// CHECK_EXCLUSIVE_SUM-NEXT:   cub::DeviceScan::ExclusiveSum(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, num_items/*int*/, stream/*cudaStream_t*/);
// CHECK_EXCLUSIVE_SUM-NEXT: Is migrated to:
// CHECK_EXCLUSIVE_SUM-NEXT:   dpct::queue_ptr stream;
// CHECK_EXCLUSIVE_SUM-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_EXCLUSIVE_SUM-NEXT:   oneapi::dpl::exclusive_scan(oneapi::dpl::execution::device_policy(*stream), d_in, d_in + num_items, d_out, typename std::iterator_traits<decltype(d_in)>::value_type{});

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceScan::ExclusiveSumByKey | FileCheck %s -check-prefix=CHECK_EXCLUSIVE_SUM_BY_KEY
// CHECK_EXCLUSIVE_SUM_BY_KEY: CUDA API:
// CHECK_EXCLUSIVE_SUM_BY_KEY-NEXT:   cudaStream_t stream;
// CHECK_EXCLUSIVE_SUM_BY_KEY-NEXT:   cudaStreamCreate(&stream);
// CHECK_EXCLUSIVE_SUM_BY_KEY-NEXT:   cub::DeviceScan::ExclusiveSumByKey(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_keys_in/*KeysInputIteratorT*/, d_values_in/*ValuesInputIteratorT*/, d_values_out/*ValuesOutputIteratorT*/, num_items/*int*/, equality_op/*EqualityOpT*/, stream/*cudaStream_t*/);
// CHECK_EXCLUSIVE_SUM_BY_KEY-NEXT: Is migrated to:
// CHECK_EXCLUSIVE_SUM_BY_KEY-NEXT:   dpct::queue_ptr stream;
// CHECK_EXCLUSIVE_SUM_BY_KEY-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_EXCLUSIVE_SUM_BY_KEY-NEXT:   oneapi::dpl::exclusive_scan_by_key(oneapi::dpl::execution::device_policy(*stream), d_keys_in, d_keys_in + num_items, d_values_in, d_values_out, typename std::iterator_traits<decltype(d_keys_in)>::value_type{}, equality_op);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceScan::InclusiveScan | FileCheck %s -check-prefix=CHECK_INCLUSIVE_SCAN
// CHECK_INCLUSIVE_SCAN: CUDA API:
// CHECK_INCLUSIVE_SCAN-NEXT:   cudaStream_t stream;
// CHECK_INCLUSIVE_SCAN-NEXT:   cudaStreamCreate(&stream);
// CHECK_INCLUSIVE_SCAN-NEXT:   cub::DeviceScan::InclusiveScan(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, scan_op/*ScanOpT*/, num_items/*int*/, stream/*cudaStream_t*/);
// CHECK_INCLUSIVE_SCAN-NEXT: Is migrated to:
// CHECK_INCLUSIVE_SCAN-NEXT:   dpct::queue_ptr stream;
// CHECK_INCLUSIVE_SCAN-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_INCLUSIVE_SCAN-NEXT:   oneapi::dpl::inclusive_scan(oneapi::dpl::execution::device_policy(*stream), d_in, d_in + num_items, d_out, scan_op);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceScan::InclusiveScanByKey | FileCheck %s -check-prefix=CHECK_INCLUSIVE_SCAN_BY_KEY
// CHECK_INCLUSIVE_SCAN_BY_KEY: CUDA API:
// CHECK_INCLUSIVE_SCAN_BY_KEY-NEXT:   cudaStream_t stream;
// CHECK_INCLUSIVE_SCAN_BY_KEY-NEXT:   cudaStreamCreate(&stream);
// CHECK_INCLUSIVE_SCAN_BY_KEY-NEXT:   cub::DeviceScan::InclusiveScanByKey(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_keys_in/*KeysInputIteratorT*/, d_values_in/*ValuesInputIteratorT*/, d_values_out/*ValuesOutputIteratorT*/, scan_op/*ScanOpT*/, num_items/*int*/, equality_op/*EqualityOpT*/, stream/*cudaStream_t*/);
// CHECK_INCLUSIVE_SCAN_BY_KEY-NEXT: Is migrated to:
// CHECK_INCLUSIVE_SCAN_BY_KEY-NEXT:   dpct::queue_ptr stream;
// CHECK_INCLUSIVE_SCAN_BY_KEY-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_INCLUSIVE_SCAN_BY_KEY-NEXT:   oneapi::dpl::inclusive_scan_by_key(oneapi::dpl::execution::device_policy(*stream), d_keys_in, d_keys_in + num_items, d_values_in, d_values_out, equality_op, scan_op);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceScan::InclusiveSum | FileCheck %s -check-prefix=CHECK_INCLUSIVE_SUM
// CHECK_INCLUSIVE_SUM: CUDA API:
// CHECK_INCLUSIVE_SUM-NEXT:   cudaStream_t stream;
// CHECK_INCLUSIVE_SUM-NEXT:   cudaStreamCreate(&stream);
// CHECK_INCLUSIVE_SUM-NEXT:   cub::DeviceScan::InclusiveSum(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, num_items/*int*/, stream/*cudaStream_t*/);
// CHECK_INCLUSIVE_SUM-NEXT: Is migrated to:
// CHECK_INCLUSIVE_SUM-NEXT:   dpct::queue_ptr stream;
// CHECK_INCLUSIVE_SUM-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_INCLUSIVE_SUM-NEXT:   oneapi::dpl::inclusive_scan(oneapi::dpl::execution::device_policy(*stream), d_in, d_in + num_items, d_out);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceScan::InclusiveSumByKey | FileCheck %s -check-prefix=CHECK_INCLUSIVE_SUM_BY_KEY
// CHECK_INCLUSIVE_SUM_BY_KEY: CUDA API:
// CHECK_INCLUSIVE_SUM_BY_KEY-NEXT:   cudaStream_t stream;
// CHECK_INCLUSIVE_SUM_BY_KEY-NEXT:   cudaStreamCreate(&stream);
// CHECK_INCLUSIVE_SUM_BY_KEY-NEXT:   cub::DeviceScan::InclusiveSumByKey(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_keys_in/*KeysInputIteratorT*/, d_values_in/*ValuesInputIteratorT*/, d_values_out/*ValuesOutputIteratorT*/, num_items/*int*/, equality_op/*EqualityOpT*/, stream/*cudaStream_t*/);
// CHECK_INCLUSIVE_SUM_BY_KEY-NEXT: Is migrated to:
// CHECK_INCLUSIVE_SUM_BY_KEY-NEXT:   dpct::queue_ptr stream;
// CHECK_INCLUSIVE_SUM_BY_KEY-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_INCLUSIVE_SUM_BY_KEY-NEXT:   oneapi::dpl::inclusive_scan_by_key(oneapi::dpl::execution::device_policy(*stream), d_keys_in, d_keys_in + num_items, d_values_in, d_values_out, equality_op);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSelect::Flagged | FileCheck %s -check-prefix=CHECK_SELECT_FLAGGED
// CHECK_SELECT_FLAGGED: CUDA API:
// CHECK_SELECT_FLAGGED-NEXT:   cudaStream_t stream;
// CHECK_SELECT_FLAGGED-NEXT:   cudaStreamCreate(&stream);
// CHECK_SELECT_FLAGGED-NEXT:   cub::DeviceSelect::Flagged(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_flags/*FlagIterator*/, d_out/*OutputIteratorT*/, d_num_selected_out/*NumSelectedIteratorT*/, num_items/*int*/, stream/*cudaStream_t*/);
// CHECK_SELECT_FLAGGED-NEXT: Is migrated to:
// CHECK_SELECT_FLAGGED-NEXT:   dpct::queue_ptr stream;
// CHECK_SELECT_FLAGGED-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_SELECT_FLAGGED-NEXT:   stream->fill(d_num_selected_out, std::distance(d_out, dpct::copy_if(oneapi::dpl::execution::device_policy(*stream), d_in, d_in + num_items, d_flags, d_out, [](const auto &t) -> bool { return t; })), 1).wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSelect::If | FileCheck %s -check-prefix=CHECK_SELECT_IF
// CHECK_SELECT_IF: CUDA API:
// CHECK_SELECT_IF-NEXT:   cudaStream_t stream;
// CHECK_SELECT_IF-NEXT:   cudaStreamCreate(&stream);
// CHECK_SELECT_IF-NEXT:   cub::DeviceSelect::If(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, d_num_selected_out/*NumSelectedIteratorT*/, num_items/*int*/, select_op/*SelectOp*/, stream/*cudaStream_t*/);
// CHECK_SELECT_IF-NEXT: Is migrated to:
// CHECK_SELECT_IF-NEXT:   dpct::queue_ptr stream;
// CHECK_SELECT_IF-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_SELECT_IF-NEXT:   stream->fill(d_num_selected_out, std::distance(d_out, oneapi::dpl::copy_if(oneapi::dpl::execution::device_policy(*stream), d_in, d_in + num_items, d_out, select_op)), 1).wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSelect::Unique | FileCheck %s -check-prefix=CHECK_SELECT_UNIQUE
// CHECK_SELECT_UNIQUE: CUDA API:
// CHECK_SELECT_UNIQUE-NEXT:   cudaStream_t stream;
// CHECK_SELECT_UNIQUE-NEXT:   cudaStreamCreate(&stream);
// CHECK_SELECT_UNIQUE-NEXT:   cub::DeviceSelect::Unique(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, d_num_selected_out/*NumSelectedIteratorT*/, num_items/*int*/, stream/*cudaStream_t*/);
// CHECK_SELECT_UNIQUE-NEXT: Is migrated to:
// CHECK_SELECT_UNIQUE-NEXT:   dpct::queue_ptr stream;
// CHECK_SELECT_UNIQUE-NEXT:   stream = dpct::get_current_device().create_queue();
// CHECK_SELECT_UNIQUE-NEXT:   stream->fill(d_num_selected_out, std::distance(d_out, oneapi::dpl::unique_copy(oneapi::dpl::execution::device_policy(*stream), d_in, d_in + num_items, d_out)), 1).wait();
