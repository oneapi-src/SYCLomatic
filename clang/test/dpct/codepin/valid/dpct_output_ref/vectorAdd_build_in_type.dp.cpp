// RUN: echo 0
// CHECK: d_A = sycl::malloc_device<float>(VECTOR_SIZE, q_ct1);
// CHECK-NEXT: d_B = sycl::malloc_device<float>(VECTOR_SIZE, q_ct1);
// CHECK-NEXT: d_C = sycl::malloc_device<float>(VECTOR_SIZE, q_ct1);
// CHECK-NEXT: q_ct1.parallel_for(
