// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.2, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-11.8
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.2, v11.4, v11.5, v11.6, v11.7, v11.8
// UNSUPPORTED: system-windows
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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceHistogram::MultiHistogramRange | FileCheck %s -check-prefix=CHECK_DEVICEHISTOGRAM_MULTIHISTOGRAMRANGE
// CHECK_DEVICEHISTOGRAM_MULTIHISTOGRAMRANGE:  CUDA API:
// CHECK_DEVICEHISTOGRAM_MULTIHISTOGRAMRANGE:    void *d_temp_storage = nullptr;
// CHECK_DEVICEHISTOGRAM_MULTIHISTOGRAMRANGE:    size_t temp_storage_bytes = 0;
// CHECK_DEVICEHISTOGRAM_MULTIHISTOGRAMRANGE:    cub::DeviceHistogram::MultiHistogramRange<4, 3>(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_samples/*unsigned char **/, d_histogram/*int *(&)[3]*/, num_levels/*int(&)[3]*/, d_levels/*unsigned int *(&)[3]*/, num_pixels/*int*/);
// CHECK_DEVICEHISTOGRAM_MULTIHISTOGRAMRANGE:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICEHISTOGRAM_MULTIHISTOGRAMRANGE:    cub::DeviceHistogram::MultiHistogramRange<4, 3>(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_samples/*unsigned char **/, d_histogram/*int *(&)[3]*/, num_levels/*int(&)[3]*/, d_levels/*unsigned int *(&)[3]*/, num_pixels/*int*/);
// CHECK_DEVICEHISTOGRAM_MULTIHISTOGRAMRANGE:  Is migrated to:
// CHECK_DEVICEHISTOGRAM_MULTIHISTOGRAMRANGE:    dpct::multi_histogram_range<4, 3>(oneapi::dpl::execution::device_policy(q_ct1), d_samples, d_histogram, num_levels, d_levels, num_pixels);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceHistogram::MultiHistogramEven | FileCheck %s -check-prefix=CHECK_DEVICEHISTOGRAM_MULTIHISTOGRAMEVEN
// CHECK_DEVICEHISTOGRAM_MULTIHISTOGRAMEVEN:  CUDA API:
// CHECK_DEVICEHISTOGRAM_MULTIHISTOGRAMEVEN:    void *d_temp_storage = nullptr;
// CHECK_DEVICEHISTOGRAM_MULTIHISTOGRAMEVEN:    size_t temp_storage_bytes = 0;
// CHECK_DEVICEHISTOGRAM_MULTIHISTOGRAMEVEN:    cub::DeviceHistogram::MultiHistogramEven<4, 3>(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_samples/*unsigned char **/, d_histogram/*int *(&)[3]*/, num_levels/*int(&)[3]*/, lower_level/*unsigned int(&)[3]*/, upper_level/*unsigned int(&)[3]*/, num_pixels/*int*/, S/*cudaStream_t*/);
// CHECK_DEVICEHISTOGRAM_MULTIHISTOGRAMEVEN:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICEHISTOGRAM_MULTIHISTOGRAMEVEN:    cub::DeviceHistogram::MultiHistogramEven<4, 3>(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_samples/*unsigned char **/, d_histogram/*int *(&)[3]*/, num_levels/*int(&)[3]*/, lower_level/*unsigned int(&)[3]*/, upper_level/*unsigned int(&)[3]*/, num_pixels/*int*/, S/*cudaStream_t*/);
// CHECK_DEVICEHISTOGRAM_MULTIHISTOGRAMEVEN:  Is migrated to:
// CHECK_DEVICEHISTOGRAM_MULTIHISTOGRAMEVEN:    dpct::multi_histogram_even<4, 3>(oneapi::dpl::execution::device_policy(*S), d_samples, d_histogram, num_levels, lower_level, upper_level, num_pixels);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceHistogram::HistogramEven | FileCheck %s -check-prefix=CHECK_DEVICEHISTOGRAM_HISTOGRAMEVEN
// CHECK_DEVICEHISTOGRAM_HISTOGRAMEVEN:  CUDA API:
// CHECK_DEVICEHISTOGRAM_HISTOGRAMEVEN:    void *d_temp_storage = nullptr;
// CHECK_DEVICEHISTOGRAM_HISTOGRAMEVEN:    size_t temp_storage_bytes = 0;
// CHECK_DEVICEHISTOGRAM_HISTOGRAMEVEN:    cub::DeviceHistogram::HistogramEven(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_samples/*float **/, d_histogram/*int **/, num_levels/*int*/, lower_level/*float*/, upper_level/*float*/, num_samples/*int*/);
// CHECK_DEVICEHISTOGRAM_HISTOGRAMEVEN:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICEHISTOGRAM_HISTOGRAMEVEN:    cub::DeviceHistogram::HistogramEven(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_samples/*float **/, d_histogram/*int **/, num_levels/*int*/, lower_level/*float*/, upper_level/*float*/, num_samples/*int*/);
// CHECK_DEVICEHISTOGRAM_HISTOGRAMEVEN:  Is migrated to:
// CHECK_DEVICEHISTOGRAM_HISTOGRAMEVEN:    dpct::histogram_even(oneapi::dpl::execution::device_policy(q_ct1), d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceHistogram::HistogramRange | FileCheck %s -check-prefix=CHECK_DEVICEHISTOGRAM_HISTOGRAMRANGE
// CHECK_DEVICEHISTOGRAM_HISTOGRAMRANGE:  CUDA API:
// CHECK_DEVICEHISTOGRAM_HISTOGRAMRANGE:    void *d_temp_storage = nullptr;
// CHECK_DEVICEHISTOGRAM_HISTOGRAMRANGE:    size_t temp_storage_bytes = 0;
// CHECK_DEVICEHISTOGRAM_HISTOGRAMRANGE:    cub::DeviceHistogram::HistogramRange(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_samples/*float **/, d_histogram/*int **/, num_levels/*int*/, d_levels/*float **/, num_samples/*int*/);
// CHECK_DEVICEHISTOGRAM_HISTOGRAMRANGE:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICEHISTOGRAM_HISTOGRAMRANGE:    cub::DeviceHistogram::HistogramRange(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_samples/*float **/, d_histogram/*int **/, num_levels/*int*/, d_levels/*float **/, num_samples/*int*/);
// CHECK_DEVICEHISTOGRAM_HISTOGRAMRANGE:  Is migrated to:
// CHECK_DEVICEHISTOGRAM_HISTOGRAMRANGE:    dpct::histogram_range(oneapi::dpl::execution::device_policy(q_ct1), d_samples, d_histogram, num_levels, d_levels, num_samples);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceMergeSort::SortKeys | FileCheck %s -check-prefix=CHECK_DEVICEMERGESORT_SORTKEYS
// CHECK_DEVICEMERGESORT_SORTKEYS:  CUDA API:
// CHECK_DEVICEMERGESORT_SORTKEYS:    void *temp_storage = nullptr;
// CHECK_DEVICEMERGESORT_SORTKEYS:    size_t temp_storage_size;
// CHECK_DEVICEMERGESORT_SORTKEYS:    cub::DeviceMergeSort::SortKeys(temp_storage/*void **/, temp_storage_size/*size_t*/, d_keys/*int **/, num_items/*int*/, op/*CustomOpT*/);
// CHECK_DEVICEMERGESORT_SORTKEYS:    cudaMalloc(&temp_storage, temp_storage_size);
// CHECK_DEVICEMERGESORT_SORTKEYS:    cub::DeviceMergeSort::SortKeys(temp_storage/*void **/, temp_storage_size/*size_t*/, d_keys/*int **/, num_items/*int*/, op/*CustomOpT*/);
// CHECK_DEVICEMERGESORT_SORTKEYS:  Is migrated to:
// CHECK_DEVICEMERGESORT_SORTKEYS:    oneapi::dpl::sort(oneapi::dpl::execution::device_policy(q_ct1), d_keys, d_keys + num_items, op);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceMergeSort::SortKeysCopy | FileCheck %s -check-prefix=CHECK_DEVICEMERGESORT_SORTKEYSCOPY
// CHECK_DEVICEMERGESORT_SORTKEYSCOPY:  CUDA API:
// CHECK_DEVICEMERGESORT_SORTKEYSCOPY:    void *temp_storage = nullptr;
// CHECK_DEVICEMERGESORT_SORTKEYSCOPY:    size_t temp_storage_size;
// CHECK_DEVICEMERGESORT_SORTKEYSCOPY:    cub::DeviceMergeSort::SortKeysCopy(temp_storage/*void **/, temp_storage_size/*size_t*/, d_keys/*int **/, d_outs/*int **/, num_items/*int*/, op/*CustomOpT*/);
// CHECK_DEVICEMERGESORT_SORTKEYSCOPY:    cudaMalloc(&temp_storage, temp_storage_size);
// CHECK_DEVICEMERGESORT_SORTKEYSCOPY:    cub::DeviceMergeSort::SortKeysCopy(temp_storage/*void **/, temp_storage_size/*size_t*/, d_keys/*int **/, d_outs/*int **/, num_items/*int*/, op/*CustomOpT*/);
// CHECK_DEVICEMERGESORT_SORTKEYSCOPY:  Is migrated to:
// CHECK_DEVICEMERGESORT_SORTKEYSCOPY:    oneapi::dpl::partial_sort_copy(oneapi::dpl::execution::device_policy(q_ct1), d_keys, d_keys + num_items, d_outs, d_outs + num_items, op);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceMergeSort::SortPairs | FileCheck %s -check-prefix=CHECK_DEVICEMERGESORT_SORTPAIRS
// CHECK_DEVICEMERGESORT_SORTPAIRS:  CUDA API:
// CHECK_DEVICEMERGESORT_SORTPAIRS:    void *temp_storage = nullptr;
// CHECK_DEVICEMERGESORT_SORTPAIRS:    size_t temp_storage_size;
// CHECK_DEVICEMERGESORT_SORTPAIRS:    cub::DeviceMergeSort::SortPairs(temp_storage/*void **/, temp_storage_size/*size_t*/, d_keys/*int **/, d_values/*int **/, num_items/*int*/, op/*CustomOpT*/);
// CHECK_DEVICEMERGESORT_SORTPAIRS:    cudaMalloc(&temp_storage, temp_storage_size);
// CHECK_DEVICEMERGESORT_SORTPAIRS:    cub::DeviceMergeSort::SortPairs(temp_storage/*void **/, temp_storage_size/*size_t*/, d_keys/*int **/, d_values/*int **/, num_items/*int*/, op/*CustomOpT*/);
// CHECK_DEVICEMERGESORT_SORTPAIRS:  Is migrated to:
// CHECK_DEVICEMERGESORT_SORTPAIRS:    dpct::sort(oneapi::dpl::execution::device_policy(q_ct1), d_keys, d_keys + num_items, d_values, op);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceMergeSort::StableSortKeys | FileCheck %s -check-prefix=CHECK_DEVICEMERGESORT_STABLESORTKEYS
// CHECK_DEVICEMERGESORT_STABLESORTKEYS:  CUDA API:
// CHECK_DEVICEMERGESORT_STABLESORTKEYS:    void *temp_storage = nullptr;
// CHECK_DEVICEMERGESORT_STABLESORTKEYS:    size_t temp_storage_size;
// CHECK_DEVICEMERGESORT_STABLESORTKEYS:    cub::DeviceMergeSort::StableSortKeys(temp_storage/*void **/, temp_storage_size/*size_t*/, d_keys/*int **/, num_items/*int*/, op/*CustomOpT*/);
// CHECK_DEVICEMERGESORT_STABLESORTKEYS:    cudaMalloc(&temp_storage, temp_storage_size);
// CHECK_DEVICEMERGESORT_STABLESORTKEYS:    cub::DeviceMergeSort::StableSortKeys(temp_storage/*void **/, temp_storage_size/*size_t*/, d_keys/*int **/, num_items/*int*/, op/*CustomOpT*/);
// CHECK_DEVICEMERGESORT_STABLESORTKEYS:  Is migrated to:
// CHECK_DEVICEMERGESORT_STABLESORTKEYS:    oneapi::dpl::stable_sort(oneapi::dpl::execution::device_policy(q_ct1), d_keys, d_keys + num_items, op);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceMergeSort::StableSortPairs | FileCheck %s -check-prefix=CHECK_DEVICEMERGESORT_STABLESORTPAIRS
// CHECK_DEVICEMERGESORT_STABLESORTPAIRS:  CUDA API:
// CHECK_DEVICEMERGESORT_STABLESORTPAIRS:    void *temp_storage = nullptr;
// CHECK_DEVICEMERGESORT_STABLESORTPAIRS:    size_t temp_storage_size;
// CHECK_DEVICEMERGESORT_STABLESORTPAIRS:    cub::DeviceMergeSort::StableSortPairs(temp_storage/*void **/, temp_storage_size/*size_t*/, d_keys/*int **/, d_values/*int **/, num_items/*int*/, op/*CustomOpT*/);
// CHECK_DEVICEMERGESORT_STABLESORTPAIRS:    cudaMalloc(&temp_storage, temp_storage_size);
// CHECK_DEVICEMERGESORT_STABLESORTPAIRS:    cub::DeviceMergeSort::StableSortPairs(temp_storage/*void **/, temp_storage_size/*size_t*/, d_keys/*int **/, d_values/*int **/, num_items/*int*/, op/*CustomOpT*/);
// CHECK_DEVICEMERGESORT_STABLESORTPAIRS:  Is migrated to:
// CHECK_DEVICEMERGESORT_STABLESORTPAIRS:    dpct::stable_sort(oneapi::dpl::execution::device_policy(q_ct1), d_keys, d_keys + num_items, d_values, op);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DevicePartition::Flagged | FileCheck %s -check-prefix=CHECK_DEVICEPARTITION_FLAGGED
// CHECK_DEVICEPARTITION_FLAGGED:  CUDA API:
// CHECK_DEVICEPARTITION_FLAGGED:    void *d_temp_storage = nullptr;
// CHECK_DEVICEPARTITION_FLAGGED:    size_t temp_storage_bytes = 0;
// CHECK_DEVICEPARTITION_FLAGGED:    cub::DevicePartition::Flagged(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_in/*int **/, d_flags/*int **/, d_out/*int **/, d_num_selected_out/*int **/, num_items/*int*/);
// CHECK_DEVICEPARTITION_FLAGGED:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICEPARTITION_FLAGGED:    cub::DevicePartition::Flagged(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_in/*int **/, d_flags/*int **/, d_out/*int **/, d_num_selected_out/*int **/, num_items/*int*/);
// CHECK_DEVICEPARTITION_FLAGGED:  Is migrated to:
// CHECK_DEVICEPARTITION_FLAGGED:    dpct::partition_flagged(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_flags, d_out, d_num_selected_out, num_items);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DevicePartition::If | FileCheck %s -check-prefix=CHECK_DEVICEPARTITION_IF
// CHECK_DEVICEPARTITION_IF:  CUDA API:
// CHECK_DEVICEPARTITION_IF:    void *d_temp_storage = nullptr;
// CHECK_DEVICEPARTITION_IF:    size_t temp_storage_bytes = 0;
// CHECK_DEVICEPARTITION_IF:    cub::DevicePartition::If(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_in/*int **/, d_out/*int **/, d_num_selected_out/*int **/, num_items/*int*/, select_op/*SelectOp*/);
// CHECK_DEVICEPARTITION_IF:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICEPARTITION_IF:    cub::DevicePartition::If(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_in/*int **/, d_out/*int **/, d_num_selected_out/*int **/, num_items/*int*/, select_op/*SelectOp*/);
// CHECK_DEVICEPARTITION_IF:  Is migrated to:
// CHECK_DEVICEPARTITION_IF:  dpct::partition_if(oneapi::dpl::execution::device_policy(q_ct1), d_in, d_out, d_num_selected_out, num_items, select_op);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceRadixSort::SortKeys | FileCheck %s -check-prefix=CHECK_DEVICERADIXSORT_SORTKEYS
// CHECK_DEVICERADIXSORT_SORTKEYS:  CUDA API:
// CHECK_DEVICERADIXSORT_SORTKEYS:    void *d_temp_storage = nullptr;
// CHECK_DEVICERADIXSORT_SORTKEYS:    size_t temp_storage_bytes;
// CHECK_DEVICERADIXSORT_SORTKEYS:    cub::DeviceRadixSort::SortKeys(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, n/*int*/);
// CHECK_DEVICERADIXSORT_SORTKEYS:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICERADIXSORT_SORTKEYS:    cub::DeviceRadixSort::SortKeys(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, n/*int*/);
// CHECK_DEVICERADIXSORT_SORTKEYS:  Is migrated to:
// CHECK_DEVICERADIXSORT_SORTKEYS:    dpct::sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, false);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceRadixSort::SortKeysDescending | FileCheck %s -check-prefix=CHECK_DEVICERADIXSORT_SORTKEYSDESCENDING
// CHECK_DEVICERADIXSORT_SORTKEYSDESCENDING:  CUDA API:
// CHECK_DEVICERADIXSORT_SORTKEYSDESCENDING:    void *d_temp_storage = nullptr;
// CHECK_DEVICERADIXSORT_SORTKEYSDESCENDING:    size_t temp_storage_bytes;
// CHECK_DEVICERADIXSORT_SORTKEYSDESCENDING:    cub::DeviceRadixSort::SortKeysDescending(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, n/*int*/);
// CHECK_DEVICERADIXSORT_SORTKEYSDESCENDING:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICERADIXSORT_SORTKEYSDESCENDING:    cub::DeviceRadixSort::SortKeysDescending(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, n/*int*/);
// CHECK_DEVICERADIXSORT_SORTKEYSDESCENDING:  Is migrated to:
// CHECK_DEVICERADIXSORT_SORTKEYSDESCENDING:    dpct::sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceRadixSort::SortPairs | FileCheck %s -check-prefix=CHECK_DEVICERADIXSORT_SORTPAIRS
// CHECK_DEVICERADIXSORT_SORTPAIRS:  CUDA API:
// CHECK_DEVICERADIXSORT_SORTPAIRS:    void *d_temp_storage = nullptr;
// CHECK_DEVICERADIXSORT_SORTPAIRS:    size_t temp_storage_bytes;
// CHECK_DEVICERADIXSORT_SORTPAIRS:    cub::DeviceRadixSort::SortPairs(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, d_values_in/*int **/, d_values_out/*int ***/, n/*int*/);
// CHECK_DEVICERADIXSORT_SORTPAIRS:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICERADIXSORT_SORTPAIRS:    cub::DeviceRadixSort::SortPairs(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, d_values_in/*int **/, d_values_out/*int ***/, n/*int*/);
// CHECK_DEVICERADIXSORT_SORTPAIRS:  Is migrated to:
// CHECK_DEVICERADIXSORT_SORTPAIRS:    dpct::sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, n, false);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceRadixSort::SortPairsDescending | FileCheck %s -check-prefix=CHECK_DEVICERADIXSORT_SORTPAIRSDESCENDING
// CHECK_DEVICERADIXSORT_SORTPAIRSDESCENDING:  CUDA API:
// CHECK_DEVICERADIXSORT_SORTPAIRSDESCENDING:    void *d_temp_storage = nullptr;
// CHECK_DEVICERADIXSORT_SORTPAIRSDESCENDING:    size_t temp_storage_bytes;
// CHECK_DEVICERADIXSORT_SORTPAIRSDESCENDING:    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, d_values_in/*int **/, d_values_out/*int ***/, n/*int*/);
// CHECK_DEVICERADIXSORT_SORTPAIRSDESCENDING:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICERADIXSORT_SORTPAIRSDESCENDING:    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, d_values_in/*int **/, d_values_out/*int ***/, n/*int*/);
// CHECK_DEVICERADIXSORT_SORTPAIRSDESCENDING:  Is migrated to:
// CHECK_DEVICERADIXSORT_SORTPAIRSDESCENDING:    dpct::sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, n, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSegmentedRadixSort::SortKeys | FileCheck %s -check-prefix=CHECK_DEVICESEGMENTEDRADIXSORT_SORTKEYS
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTKEYS:  CUDA API:
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTKEYS:    void *d_temp_storage = nullptr;
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTKEYS:    size_t temp_storage_bytes;
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTKEYS:    cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTKEYS:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTKEYS:    cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTKEYS:  Is migrated to:
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTKEYS:    dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, false);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSegmentedRadixSort::SortKeysDescending | FileCheck %s -check-prefix=CHECK_DEVICESEGMENTEDRADIXSORT_SORTKEYSDESCENDING
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTKEYSDESCENDING:  CUDA API:
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTKEYSDESCENDING:    void *d_temp_storage = nullptr;
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTKEYSDESCENDING:    size_t temp_storage_bytes;
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTKEYSDESCENDING:    cub::DeviceSegmentedRadixSort::SortKeysDescending(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTKEYSDESCENDING:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTKEYSDESCENDING:    cub::DeviceSegmentedRadixSort::SortKeysDescending(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTKEYSDESCENDING:  Is migrated to:
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTKEYSDESCENDING:    dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSegmentedRadixSort::SortPairs | FileCheck %s -check-prefix=CHECK_DEVICESEGMENTEDRADIXSORT_SORTPAIRS
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTPAIRS:  CUDA API:
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTPAIRS:    void *d_temp_storage = nullptr;
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTPAIRS:    size_t temp_storage_bytes;
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTPAIRS:    cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, d_values_in/*int **/, d_values_out/*int ***/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTPAIRS:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTPAIRS:    cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, d_values_in/*int **/, d_values_out/*int ***/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTPAIRS:  Is migrated to:
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTPAIRS:    dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, n, num_segments, d_offsets, d_offsets + 1, false);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSegmentedRadixSort::SortPairsDescending | FileCheck %s -check-prefix=CHECK_DEVICESEGMENTEDRADIXSORT_SORTPAIRSDESCENDING
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTPAIRSDESCENDING:  CUDA API:
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTPAIRSDESCENDING:    void *d_temp_storage = nullptr;
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTPAIRSDESCENDING:    size_t temp_storage_bytes;
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTPAIRSDESCENDING:    cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, d_values_in/*int **/, d_values_out/*int ***/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTPAIRSDESCENDING:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTPAIRSDESCENDING:    cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, d_values_in/*int **/, d_values_out/*int ***/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTPAIRSDESCENDING:  Is migrated to:
// CHECK_DEVICESEGMENTEDRADIXSORT_SORTPAIRSDESCENDING:    dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, n, num_segments, d_offsets, d_offsets + 1, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSegmentedReduce::ArgMax | FileCheck %s -check-prefix=CHECK_DEVICESEGMENTEDREDUCE_ARGMAX
// CHECK_DEVICESEGMENTEDREDUCE_ARGMAX:  CUDA API:
// CHECK_DEVICESEGMENTEDREDUCE_ARGMAX:    cub::DeviceSegmentedReduce::ArgMax(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDREDUCE_ARGMAX:  Is migrated to:
// CHECK_DEVICESEGMENTEDREDUCE_ARGMAX:    dpct::segmented_reduce_argmax(oneapi::dpl::execution::device_policy(dpct::get_in_order_queue()), d_in, d_out, num_segments, d_offsets, d_offsets + 1);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSegmentedReduce::ArgMin | FileCheck %s -check-prefix=CHECK_DEVICESEGMENTEDREDUCE_ARGMIN
// CHECK_DEVICESEGMENTEDREDUCE_ARGMIN:  CUDA API:
// CHECK_DEVICESEGMENTEDREDUCE_ARGMIN:    cub::DeviceSegmentedReduce::ArgMin(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDREDUCE_ARGMIN:  Is migrated to:
// CHECK_DEVICESEGMENTEDREDUCE_ARGMIN:    dpct::segmented_reduce_argmin(oneapi::dpl::execution::device_policy(dpct::get_in_order_queue()), d_in, d_out, num_segments, d_offsets, d_offsets + 1);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSegmentedReduce::Max | FileCheck %s -check-prefix=CHECK_DEVICESEGMENTEDREDUCE_MAX
// CHECK_DEVICESEGMENTEDREDUCE_MAX:  CUDA API:
// CHECK_DEVICESEGMENTEDREDUCE_MAX:    cub::DeviceSegmentedReduce::Max(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDREDUCE_MAX:  Is migrated to:
// CHECK_DEVICESEGMENTEDREDUCE_MAX:    dpct::device::segmented_reduce<128>(dpct::get_in_order_queue(), d_in, d_out, num_segments, d_offsets, d_offsets + 1, sycl::maximum<>(), std::numeric_limits<typename std::iterator_traits<decltype(d_out)>::value_type>::lowest());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSegmentedReduce::Min | FileCheck %s -check-prefix=CHECK_DEVICESEGMENTEDREDUCE_MIN
// CHECK_DEVICESEGMENTEDREDUCE_MIN:  CUDA API:
// CHECK_DEVICESEGMENTEDREDUCE_MIN:    cub::DeviceSegmentedReduce::Min(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDREDUCE_MIN:  Is migrated to:
// CHECK_DEVICESEGMENTEDREDUCE_MIN:    dpct::device::segmented_reduce<128>(dpct::get_in_order_queue(), d_in, d_out, num_segments, d_offsets, d_offsets + 1, sycl::minimum<>(), std::numeric_limits<typename std::iterator_traits<decltype(d_out)>::value_type>::max());

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSegmentedReduce::Reduce | FileCheck %s -check-prefix=CHECK_DEVICESEGMENTEDREDUCE_REDUCE
// CHECK_DEVICESEGMENTEDREDUCE_REDUCE:  CUDA API:
// CHECK_DEVICESEGMENTEDREDUCE_REDUCE:    cub::DeviceSegmentedReduce::Reduce(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/, op/*ReductionOpT*/, init_value/*T*/);
// CHECK_DEVICESEGMENTEDREDUCE_REDUCE:  Is migrated to:
// CHECK_DEVICESEGMENTEDREDUCE_REDUCE:    dpct::device::segmented_reduce<128>(dpct::get_in_order_queue(), d_in, d_out, num_segments, d_offsets, d_offsets + 1, dpct_placeholder, init_value);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSegmentedReduce::Sum | FileCheck %s -check-prefix=CHECK_DEVICESEGMENTEDREDUCE_SUM
// CHECK_DEVICESEGMENTEDREDUCE_SUM:  CUDA API:
// CHECK_DEVICESEGMENTEDREDUCE_SUM:    cub::DeviceSegmentedReduce::Sum(temp_storage/*void **/, temp_storage_bytes/*size_t &*/, d_in/*InputIteratorT*/, d_out/*OutputIteratorT*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDREDUCE_SUM:  Is migrated to:
// CHECK_DEVICESEGMENTEDREDUCE_SUM:    dpct::device::segmented_reduce<128>(dpct::get_in_order_queue(), d_in, d_out, num_segments, d_offsets, d_offsets + 1, sycl::plus<>(), typename std::iterator_traits<decltype(d_out)>::value_type{});

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSegmentedSort::SortKeys | FileCheck %s -check-prefix=CHECK_DEVICESEGMENTEDSORT_SORTKEYS
// CHECK_DEVICESEGMENTEDSORT_SORTKEYS:  CUDA API:
// CHECK_DEVICESEGMENTEDSORT_SORTKEYS:    void *d_temp_storage = nullptr;
// CHECK_DEVICESEGMENTEDSORT_SORTKEYS:    size_t temp_storage_bytes;
// CHECK_DEVICESEGMENTEDSORT_SORTKEYS:    cub::DeviceSegmentedSort::SortKeys(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDSORT_SORTKEYS:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICESEGMENTEDSORT_SORTKEYS:    cub::DeviceSegmentedSort::SortKeys(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDSORT_SORTKEYS:  Is migrated to:
// CHECK_DEVICESEGMENTEDSORT_SORTKEYS:    dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, false);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSegmentedSort::SortKeysDescending | FileCheck %s -check-prefix=CHECK_DEVICESEGMENTEDSORT_SORTKEYSDESCENDING
// CHECK_DEVICESEGMENTEDSORT_SORTKEYSDESCENDING:  CUDA API:
// CHECK_DEVICESEGMENTEDSORT_SORTKEYSDESCENDING:    void *d_temp_storage = nullptr;
// CHECK_DEVICESEGMENTEDSORT_SORTKEYSDESCENDING:    size_t temp_storage_bytes;
// CHECK_DEVICESEGMENTEDSORT_SORTKEYSDESCENDING:    cub::DeviceSegmentedSort::SortKeysDescending(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDSORT_SORTKEYSDESCENDING:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICESEGMENTEDSORT_SORTKEYSDESCENDING:    cub::DeviceSegmentedSort::SortKeysDescending(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDSORT_SORTKEYSDESCENDING:  Is migrated to:
// CHECK_DEVICESEGMENTEDSORT_SORTKEYSDESCENDING:    dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSegmentedSort::SortPairs | FileCheck %s -check-prefix=CHECK_DEVICESEGMENTEDSORT_SORTPAIRS
// CHECK_DEVICESEGMENTEDSORT_SORTPAIRS:  CUDA API:
// CHECK_DEVICESEGMENTEDSORT_SORTPAIRS:    void *d_temp_storage = nullptr;
// CHECK_DEVICESEGMENTEDSORT_SORTPAIRS:    size_t temp_storage_bytes;
// CHECK_DEVICESEGMENTEDSORT_SORTPAIRS:    cub::DeviceSegmentedSort::SortPairs(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, d_values_in/*int **/, d_values_out/*int ***/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDSORT_SORTPAIRS:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICESEGMENTEDSORT_SORTPAIRS:    cub::DeviceSegmentedSort::SortPairs(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, d_values_in/*int **/, d_values_out/*int ***/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDSORT_SORTPAIRS:  Is migrated to:
// CHECK_DEVICESEGMENTEDSORT_SORTPAIRS:    dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, n, num_segments, d_offsets, d_offsets + 1, false);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSegmentedSort::SortPairsDescending | FileCheck %s -check-prefix=CHECK_DEVICESEGMENTEDSORT_SORTPAIRSDESCENDING
// CHECK_DEVICESEGMENTEDSORT_SORTPAIRSDESCENDING:  CUDA API:
// CHECK_DEVICESEGMENTEDSORT_SORTPAIRSDESCENDING:    void *d_temp_storage = nullptr;
// CHECK_DEVICESEGMENTEDSORT_SORTPAIRSDESCENDING:    size_t temp_storage_bytes;
// CHECK_DEVICESEGMENTEDSORT_SORTPAIRSDESCENDING:    cub::DeviceSegmentedSort::SortPairsDescending(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, d_values_in/*int **/, d_values_out/*int ***/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDSORT_SORTPAIRSDESCENDING:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICESEGMENTEDSORT_SORTPAIRSDESCENDING:    cub::DeviceSegmentedSort::SortPairsDescending(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, d_values_in/*int **/, d_values_out/*int ***/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDSORT_SORTPAIRSDESCENDING:  Is migrated to:
// CHECK_DEVICESEGMENTEDSORT_SORTPAIRSDESCENDING:    dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, n, num_segments, d_offsets, d_offsets + 1, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSegmentedSort::StableSortKeys | FileCheck %s -check-prefix=CHECK_DEVICESEGMENTEDSORT_STABLESORTKEYS
// CHECK_DEVICESEGMENTEDSORT_STABLESORTKEYS:  CUDA API:
// CHECK_DEVICESEGMENTEDSORT_STABLESORTKEYS:    void *d_temp_storage = nullptr;
// CHECK_DEVICESEGMENTEDSORT_STABLESORTKEYS:    size_t temp_storage_bytes;
// CHECK_DEVICESEGMENTEDSORT_STABLESORTKEYS:    cub::DeviceSegmentedSort::StableSortKeys(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDSORT_STABLESORTKEYS:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICESEGMENTEDSORT_STABLESORTKEYS:    cub::DeviceSegmentedSort::StableSortKeys(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDSORT_STABLESORTKEYS:  Is migrated to:
// CHECK_DEVICESEGMENTEDSORT_STABLESORTKEYS:    dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, false);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSegmentedSort::StableSortKeysDescending | FileCheck %s -check-prefix=CHECK_DEVICESEGMENTEDSORT_STABLESORTKEYSDESCENDING
// CHECK_DEVICESEGMENTEDSORT_STABLESORTKEYSDESCENDING:  CUDA API:
// CHECK_DEVICESEGMENTEDSORT_STABLESORTKEYSDESCENDING:    void *d_temp_storage = nullptr;
// CHECK_DEVICESEGMENTEDSORT_STABLESORTKEYSDESCENDING:    size_t temp_storage_bytes;
// CHECK_DEVICESEGMENTEDSORT_STABLESORTKEYSDESCENDING:    cub::DeviceSegmentedSort::StableSortKeysDescending(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDSORT_STABLESORTKEYSDESCENDING:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICESEGMENTEDSORT_STABLESORTKEYSDESCENDING:    cub::DeviceSegmentedSort::StableSortKeysDescending(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDSORT_STABLESORTKEYSDESCENDING:  Is migrated to:
// CHECK_DEVICESEGMENTEDSORT_STABLESORTKEYSDESCENDING:    dpct::segmented_sort_keys(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, n, num_segments, d_offsets, d_offsets + 1, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSegmentedSort::StableSortPairs | FileCheck %s -check-prefix=CHECK_DEVICESEGMENTEDSORT_STABLESORTPAIRS
// CHECK_DEVICESEGMENTEDSORT_STABLESORTPAIRS:  CUDA API:
// CHECK_DEVICESEGMENTEDSORT_STABLESORTPAIRS:    void *d_temp_storage = nullptr;
// CHECK_DEVICESEGMENTEDSORT_STABLESORTPAIRS:    size_t temp_storage_bytes;
// CHECK_DEVICESEGMENTEDSORT_STABLESORTPAIRS:    cub::DeviceSegmentedSort::StableSortPairs(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, d_values_in/*int **/, d_values_out/*int ***/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDSORT_STABLESORTPAIRS:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICESEGMENTEDSORT_STABLESORTPAIRS:    cub::DeviceSegmentedSort::StableSortPairs(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, d_values_in/*int **/, d_values_out/*int ***/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDSORT_STABLESORTPAIRS:  Is migrated to:
// CHECK_DEVICESEGMENTEDSORT_STABLESORTPAIRS:    dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, n, num_segments, d_offsets, d_offsets + 1);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSegmentedSort::StableSortPairsDescending | FileCheck %s -check-prefix=CHECK_DEVICESEGMENTEDSORT_STABLESORTPAIRSDESCENDING
// CHECK_DEVICESEGMENTEDSORT_STABLESORTPAIRSDESCENDING:  CUDA API:
// CHECK_DEVICESEGMENTEDSORT_STABLESORTPAIRSDESCENDING:    void *d_temp_storage = nullptr;
// CHECK_DEVICESEGMENTEDSORT_STABLESORTPAIRSDESCENDING:    size_t temp_storage_bytes;
// CHECK_DEVICESEGMENTEDSORT_STABLESORTPAIRSDESCENDING:    cub::DeviceSegmentedSort::StableSortPairsDescending(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, d_values_in/*int **/, d_values_out/*int ***/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDSORT_STABLESORTPAIRSDESCENDING:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICESEGMENTEDSORT_STABLESORTPAIRSDESCENDING:    cub::DeviceSegmentedSort::StableSortPairsDescending(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_keys_out/*int **/, d_values_in/*int **/, d_values_out/*int ***/, n/*int*/, num_segments/*int*/, d_offsets/*int **/, d_offsets + 1/*int **/);
// CHECK_DEVICESEGMENTEDSORT_STABLESORTPAIRSDESCENDING:  Is migrated to:
// CHECK_DEVICESEGMENTEDSORT_STABLESORTPAIRSDESCENDING:    dpct::segmented_sort_pairs(oneapi::dpl::execution::device_policy(q_ct1), d_keys_in, d_keys_out, d_values_in, d_values_out, n, num_segments, d_offsets, d_offsets + 1, true);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSelect::UniqueByKey | FileCheck %s -check-prefix=CHECK_DEVICESELECT_UNIQUEBYKEY
// CHECK_DEVICESELECT_UNIQUEBYKEY:  CUDA API:
// CHECK_DEVICESELECT_UNIQUEBYKEY:    void *d_temp_storage = nullptr;
// CHECK_DEVICESELECT_UNIQUEBYKEY:    size_t temp_storage_bytes = 0;
// CHECK_DEVICESELECT_UNIQUEBYKEY:    cub::DeviceSelect::UniqueByKey(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_values_in/*int **/, d_keys_out/*int **/, d_values_out/*int **/, d_num_selected_out/*int **/, num_items/*int*/, s/*cudaStream_t*/);
// CHECK_DEVICESELECT_UNIQUEBYKEY:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICESELECT_UNIQUEBYKEY:    cub::DeviceSelect::UniqueByKey(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_keys_in/*int **/, d_values_in/*int **/, d_keys_out/*int **/, d_values_out/*int **/, d_num_selected_out/*int **/, num_items/*int*/, s/*cudaStream_t*/);
// CHECK_DEVICESELECT_UNIQUEBYKEY:  Is migrated to:
// CHECK_DEVICESELECT_UNIQUEBYKEY:    s->fill(d_num_selected_out, std::distance(d_keys_out, std::get<0>(dpct::unique_copy(oneapi::dpl::execution::device_policy(*s), d_keys_in, d_keys_in + num_items, d_values_in, d_keys_out, d_values_out))), 1).wait();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cub::DeviceSpmv::CsrMV | FileCheck %s -check-prefix=CHECK_DEVICESPMV_CSRMV
// CHECK_DEVICESPMV_CSRMV:  CUDA API:
// CHECK_DEVICESPMV_CSRMV:    void *d_temp_storage = nullptr;
// CHECK_DEVICESPMV_CSRMV:    size_t temp_storage_bytes = 0;
// CHECK_DEVICESPMV_CSRMV:    cub::DeviceSpmv::CsrMV(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_values/*float **/, d_row_offsets/*int **/, d_column_indices/*int **/, d_vector_x/*float **/, d_vector_y/*float **/, num_rows/*int*/, num_cols/*int*/, num_nonzeros/*int*/);
// CHECK_DEVICESPMV_CSRMV:    cudaMalloc(&d_temp_storage, temp_storage_bytes);
// CHECK_DEVICESPMV_CSRMV:    cub::DeviceSpmv::CsrMV(d_temp_storage/*void **/, temp_storage_bytes/*size_t*/, d_values/*float **/, d_row_offsets/*int **/, d_column_indices/*int **/, d_vector_x/*float **/, d_vector_y/*float **/, num_rows/*int*/, num_cols/*int*/, num_nonzeros/*int*/);
// CHECK_DEVICESPMV_CSRMV:  Is migrated to:
// CHECK_DEVICESPMV_CSRMV:    dpct::sparse::csrmv(q_ct1, d_values, d_row_offsets, d_column_indices, d_vector_x, d_vector_y, num_rows, num_cols);
