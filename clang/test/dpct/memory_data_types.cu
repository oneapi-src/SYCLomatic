// RUN: dpct --format-range=none -usm-level=none -out-root %T/memory_data_types %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --match-full-lines --input-file %T/memory_data_types/memory_data_types.dp.cpp %s

void foo(int *data, int x, int y) {
  // CHECK: dpct::pitched_data p1 = dpct::pitched_data(data, x, x, y);
  cudaPitchedPtr p1 = make_cudaPitchedPtr(data, x, x, y);

  size_t p1_pitch, p1_x, p1_y;

  // CHECK: data = (int *)p1.get_data_ptr();
  // CHECK-NEXT: p1.set_data_ptr(data);
  // CHECK-NEXT: p1_pitch = p1.get_pitch();
  // CHECK-NEXT: p1.set_pitch(p1_pitch);
  // CHECK-NEXT: p1_x = p1.get_x();
  // CHECK-NEXT: p1.set_x(p1_x);
  // CHECK-NEXT: p1_y = p1.get_y();
  // CHECK-NEXT: p1.set_y(p1_y);
  data = (int *)p1.ptr;
  p1.ptr = data;
  p1_pitch = p1.pitch;
  p1.pitch = p1_pitch;
  p1_x = p1.xsize;
  p1.xsize = p1_x;
  p1_y = p1.ysize;
  p1.ysize = p1_y;

  // CHECK: sycl::range<3> extent = sycl::range<3>(x, y, 1);
  cudaExtent extent = make_cudaExtent(x, y, 1);
  // CHECK: sycl::id<3> pos = sycl::id<3>(0, 0, 0);
  cudaPos pos = make_cudaPos(0, 0, 0);

  // CHECK: dpct::pitched_data p2_from_data_ct1, p2_to_data_ct1;
  // CHECK-NEXT: sycl::id<3> p2_from_pos_ct1(0, 0, 0), p2_to_pos_ct1(0, 0, 0);
  // CHECK-NEXT: sycl::range<3> p2_size_ct1(1, 1, 1);
  // CHECK-NEXT: dpct::memcpy_direction p2_direction_ct1;
  cudaMemcpy3DParms p2;
  cudaArray_t a1;

  // CHECK: p2_from_data_ct1 = a1->to_pitched_data();
  p2.srcArray = a1;
  // CHECK: p2_from_pos_ct1 = pos;
  p2.srcPos = pos;
  // CHECK: p2_to_data_ct1 = p1;
  p2.dstPtr = p1;
  // CHECK: p2_to_pos_ct1 = sycl::id<3>(0, 0, 0);
  p2.dstPos = make_cudaPos(0, 0, 0);
  // CHECK: p2_size_ct1 = extent;
  p2.extent = extent;
  // CHECK: p2_direction_ct1 = dpct::device_to_host;
  p2.kind = cudaMemcpyDeviceToHost;
  // CHECK: dpct::dpct_memcpy(p2_to_data_ct1, p2_to_pos_ct1, p2_from_data_ct1, p2_from_pos_ct1, p2_size_ct1, p2_direction_ct1);
  cudaMemcpy3D(&p2);

  // CHECK: dpct::pitched_data p3;
  cudaPitchedPtr p3;
  
  // CHECK: DPCT_CHECK_ERROR(p3 = dpct::dpct_malloc(sycl::range<3>(x, y, 1)));
  cudaMalloc3D(&p3, make_cudaExtent(x, y, 1));

  // CHECK: p2_from_data_ct1 = dpct::pitched_data(data, x, x, y);
  p2.srcPtr = make_cudaPitchedPtr(data, x, x, y);
  // CHECK: p2_to_data_ct1 = p3;
  p2.dstPtr = p3;
  // CHECK: p2_size_ct1[0] = x;
  p2.extent.width = x;
  // CHECK: p2_size_ct1[1] = y;
  p2.extent.height = y;
  // CHECK: p2_size_ct1[2] = 1;
  p2.extent.depth = 1;
  // CHECK: p2_direction_ct1 = dpct::host_to_device;
  p2.kind = cudaMemcpyHostToDevice;
  // CHECK: dpct::dpct_memcpy(p2_to_data_ct1, p2_to_pos_ct1, p2_from_data_ct1, p2_from_pos_ct1, p2_size_ct1, p2_direction_ct1);
  cudaMemcpy3D(&p2);
}

