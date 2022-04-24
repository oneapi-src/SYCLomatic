// RUN: dpct -out-root %T/driver-stream-and-event %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/driver-stream-and-event/driver-stream-and-event.dp.cpp %s

void foo(){
  CUfunction f;
  CUstream s;
  CUevent e;

  //CHECK: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuFuncSetCacheConfig was removed because DPC++ currently does not support configuring shared memory on devices.
  //CHECK-NEXT: */
  cuFuncSetCacheConfig(f, CU_FUNC_CACHE_PREFER_NONE);

  //CHECK: s = dpct::get_current_device().create_queue();
  //CHECK-NEXT: s->wait();
  cuStreamCreate(&s, CU_STREAM_DEFAULT);
  cuStreamSynchronize(s);

  //CHECK: /*
  //CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuEventCreate was removed because this call is redundant in DPC++.
  //CHECK-NEXT: */
  //CHECK-NEXT: e = s->ext_oneapi_submit_barrier({e});
  cuEventCreate(&e, CU_EVENT_DEFAULT);
  cuStreamWaitEvent(s, e, 0);

  //CHECK: /*
  //CHECK-NEXT: DPCT1012:{{[0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  //CHECK-NEXT: */
  //CHECK-NEXT: e_ct1 = std::chrono::steady_clock::now();
  //CHECK-NEXT: e = s->ext_oneapi_submit_barrier();
  //CHECK-NEXT: e.wait_and_throw();
  cuEventRecord(e, s);
  cuEventSynchronize(e);

  //CHECK: sycl::info::event_command_status r;
  //CHECK-NEXT: r = e.get_info<sycl::info::event::command_execution_status>();
  CUresult r;
  r = cuEventQuery(e);

  //CHECK: sycl::event start, end;
  //CHECK-NEXT: std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  //CHECK-NEXT: std::chrono::time_point<std::chrono::steady_clock> end_ct1;
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1012:{{[0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  //CHECK-NEXT: */
  //CHECK-NEXT: start_ct1 = std::chrono::steady_clock::now();
  //CHECK-NEXT: /*
  //CHECK-NEXT: DPCT1012:{{[0-9]+}}: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
  //CHECK-NEXT: */
  //CHECK-NEXT: end_ct1 = std::chrono::steady_clock::now();
  //CHECK-NEXT: float result_time;
  //CHECK-NEXT: result_time = std::chrono::duration<float, std::milli>(end_ct1 - start_ct1).count();
  CUevent start, end;
  cuEventRecord(start, s);
  cuEventRecord(end, s);
  cuEventSynchronize(start);
  cuEventSynchronize(end);
  float result_time;
  cuEventElapsedTime(&result_time, start, end);

  int rr;
  //CHECK: rr = dpct::get_kernel_function_info((const void *)f).max_work_group_size;
  cuFuncGetAttribute(&rr, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, f);
}
