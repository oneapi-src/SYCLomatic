// RUN: dpct --format-range=none --usm-level=none -out-root %T %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/accessor-offset.dp.cpp --match-full-lines %s

__global__ void hello(int *d) {}
__global__ void hello(int *d, int i) {}

void mod(int **p) {
    p++;
}

void mod2(int *&p) {
    p++;
}

void nonmod(int *p) {
    p++;
}

int *d_a_global;

void foo() {
  // CHECK: dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // CHECK-NEXT: sycl::queue &q_ct1 = dev_ct1.default_queue();
  int *d_a;
  int **p = &d_a;
  int n;
  int x, y;

  // No offset: use right after memory allocation
  // CHECK: {
  // CHECK-NEXT:   dpct::dpct_malloc(&d_a, n * sizeof(float));
  // CHECK-NEXT:   {
  // CHECK-NEXT:     dpct::buffer_t d_a_buf_ct0 = dpct::get_buffer(d_a);
  // CHECK-NEXT:     q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto d_a_acc_ct0 = d_a_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             hello((int *)(&d_a_acc_ct0[0]), 23);
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  {
    cudaMalloc(&d_a, n * sizeof(float));
    hello<<<1, 1>>>(d_a, 23);
  }

  // Offset: compound assign operator (+=)
  // CHECK: {
  // CHECK-NEXT:   dpct::dpct_malloc(&d_a, n * sizeof(float));
  // CHECK-NEXT:   d_a += 2;
  // CHECK-NEXT:   {
  // CHECK-NEXT:     std::pair<dpct::buffer_t, size_t> d_a_buf_ct0 = dpct::get_buffer_and_offset(d_a);
  // CHECK-NEXT:     size_t d_a_offset_ct0 = d_a_buf_ct0.second;
  // CHECK-NEXT:     q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto d_a_acc_ct0 = d_a_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             int *d_a_ct0 = (int *)(&d_a_acc_ct0[0] + d_a_offset_ct0);
  // CHECK-NEXT:             hello(d_a_ct0, 23);
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  {
    cudaMalloc(&d_a, n * sizeof(float));
    d_a += 2;
    hello<<<1, 1>>>(d_a, 23);
  }

  // Offset: compound assign operator (-=)
  // CHECK: {
  // CHECK-NEXT:   dpct::dpct_malloc(&d_a, n * sizeof(float));
  // CHECK-NEXT:   d_a -= 2;
  // CHECK-NEXT:   {
  // CHECK-NEXT:     std::pair<dpct::buffer_t, size_t> d_a_buf_ct0 = dpct::get_buffer_and_offset(d_a);
  // CHECK-NEXT:     size_t d_a_offset_ct0 = d_a_buf_ct0.second;
  // CHECK-NEXT:     q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto d_a_acc_ct0 = d_a_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             int *d_a_ct0 = (int *)(&d_a_acc_ct0[0] + d_a_offset_ct0);
  // CHECK-NEXT:             hello(d_a_ct0, 23);
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  {
    cudaMalloc(&d_a, n * sizeof(float));
    d_a -= 2;
    hello<<<1, 1>>>(d_a, 23);
  }

  // Offset: assign operator (=)
  // CHECK: {
  // CHECK-NEXT:   dpct::dpct_malloc((void **)&d_a, n * sizeof(float));
  // CHECK-NEXT:   d_a = d_a + 2;
  // CHECK-NEXT:   {
  // CHECK-NEXT:     std::pair<dpct::buffer_t, size_t> d_a_buf_ct0 = dpct::get_buffer_and_offset(d_a);
  // CHECK-NEXT:     size_t d_a_offset_ct0 = d_a_buf_ct0.second;
  // CHECK-NEXT:     q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto d_a_acc_ct0 = d_a_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             int *d_a_ct0 = (int *)(&d_a_acc_ct0[0] + d_a_offset_ct0);
  // CHECK-NEXT:             hello(d_a_ct0, 23);
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  {
    cudaMalloc((void **)&d_a, n * sizeof(float));
    d_a = d_a + 2;
    hello<<<1, 1>>>(d_a, 23);
  }

  // Offset: assign operator (=)
  // CHECK: {
  // CHECK-NEXT:   dpct::dpct_malloc((void **)&d_a, n * sizeof(float));
  // CHECK-NEXT:   d_a = d_a - 2;
  // CHECK-NEXT:   {
  // CHECK-NEXT:     std::pair<dpct::buffer_t, size_t> d_a_buf_ct0 = dpct::get_buffer_and_offset(d_a);
  // CHECK-NEXT:     size_t d_a_offset_ct0 = d_a_buf_ct0.second;
  // CHECK-NEXT:     q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto d_a_acc_ct0 = d_a_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             int *d_a_ct0 = (int *)(&d_a_acc_ct0[0] + d_a_offset_ct0);
  // CHECK-NEXT:             hello(d_a_ct0, 23);
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  {
    cudaMalloc((void **)&d_a, n * sizeof(float));
    d_a = d_a - 2;
    hello<<<1, 1>>>(d_a, 23);
  }

  // Offset: reference doesn't match memory allocation exactly
  // CHECK: {
  // CHECK-NEXT:   d_a = d_a;
  // CHECK-NEXT:   dpct::dpct_malloc(&d_a + 1, n * sizeof(float));
  // CHECK-NEXT:   {
  // CHECK-NEXT:     std::pair<dpct::buffer_t, size_t> d_a_buf_ct0 = dpct::get_buffer_and_offset(d_a);
  // CHECK-NEXT:     size_t d_a_offset_ct0 = d_a_buf_ct0.second;
  // CHECK-NEXT:     q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto d_a_acc_ct0 = d_a_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             int *d_a_ct0 = (int *)(&d_a_acc_ct0[0] + d_a_offset_ct0);
  // CHECK-NEXT:             hello(d_a_ct0, 23);
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  {
    d_a = d_a;
    cudaMalloc(&d_a + 1, n * sizeof(float));
    hello<<<1, 1>>>(d_a, 23);
  }

  // Offset: reference doesn't match memory allocation exactly
  // CHECK: {
  // CHECK-NEXT:   dpct::dpct_malloc(&d_a, n * sizeof(float));
  // CHECK-NEXT:   {
  // CHECK-NEXT:     std::pair<dpct::buffer_t, size_t> d_a_buf_ct0 = dpct::get_buffer_and_offset(d_a + 1);
  // CHECK-NEXT:     size_t d_a_offset_ct0 = d_a_buf_ct0.second;
  // CHECK-NEXT:     q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto d_a_acc_ct0 = d_a_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             int *d_a_ct0 = (int *)(&d_a_acc_ct0[0] + d_a_offset_ct0);
  // CHECK-NEXT:             hello(d_a_ct0, 23);
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  {
    cudaMalloc(&d_a, n * sizeof(float));
    hello<<<1, 1>>>(d_a + 1, 23);
  }

  // Offset: reference doesn't match memory allocation exactly
  // CHECK: {
  // CHECK-NEXT:   d_a = d_a;
  // CHECK-NEXT:   dpct::dpct_malloc(p, n * sizeof(float));
  // CHECK-NEXT:   {
  // CHECK-NEXT:     std::pair<dpct::buffer_t, size_t> d_a_buf_ct0 = dpct::get_buffer_and_offset(d_a);
  // CHECK-NEXT:     size_t d_a_offset_ct0 = d_a_buf_ct0.second;
  // CHECK-NEXT:     q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto d_a_acc_ct0 = d_a_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             int *d_a_ct0 = (int *)(&d_a_acc_ct0[0] + d_a_offset_ct0);
  // CHECK-NEXT:             hello(d_a_ct0, 23);
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  {
    d_a = d_a;
    cudaMalloc(p, n * sizeof(float));
    hello<<<1, 1>>>(d_a, 23);
  }

  // No offset: ignore parens
  // CHECK: {
  // CHECK-NEXT:   dpct::dpct_malloc((&d_a), n * sizeof(float));
  // CHECK-NEXT:   {
  // CHECK-NEXT:     dpct::buffer_t d_a_buf_ct0 = dpct::get_buffer(d_a);
  // CHECK-NEXT:     q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto d_a_acc_ct0 = d_a_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             hello((int *)(&d_a_acc_ct0[0]), 23);
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  {
    cudaMalloc((&d_a), n * sizeof(float));
    hello<<<1, 1>>>(d_a, 23);
  }

  // No offset: cudaMalloc resets d_a
  // CHECK: {
  // CHECK-NEXT:   d_a -= 4;
  // CHECK-NEXT:     {
  // CHECK-NEXT:       dpct::dpct_malloc((void **)&d_a, n * sizeof(float));
  // CHECK-NEXT:     }
  // CHECK-NEXT:     {
  // CHECK-NEXT:         {
  // CHECK-NEXT:           dpct::buffer_t d_a_buf_ct0 = dpct::get_buffer(d_a);
  // CHECK-NEXT:           q_ct1.submit(
  // CHECK-NEXT:             [&](sycl::handler &cgh) {
  // CHECK-NEXT:               auto d_a_acc_ct0 = d_a_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:               cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:                 [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:                   hello((int *)(&d_a_acc_ct0[0]), 23);
  // CHECK-NEXT:                 });
  // CHECK-NEXT:             });
  // CHECK-NEXT:         }
  // CHECK-NEXT:     }
  // CHECK-NEXT: }
  {
    d_a -= 4;
      {
        cudaMalloc((void **)&d_a, n * sizeof(float));
      }
      {
        {
          hello<<<1, 1>>>(d_a, 23);
        }
      }
  }

  // Offset: d_a is used as lvalue by passing its address to mod
  // CHECK: {
  // CHECK-NEXT:   dpct::dpct_malloc((void **)&d_a, n * sizeof(float));
  // CHECK-NEXT:   mod(&d_a);
  // CHECK-NEXT:   {
  // CHECK-NEXT:     std::pair<dpct::buffer_t, size_t> d_a_buf_ct0 = dpct::get_buffer_and_offset(d_a);
  // CHECK-NEXT:     size_t d_a_offset_ct0 = d_a_buf_ct0.second;
  // CHECK-NEXT:     q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto d_a_acc_ct0 = d_a_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             int *d_a_ct0 = (int *)(&d_a_acc_ct0[0] + d_a_offset_ct0);
  // CHECK-NEXT:             hello(d_a_ct0, 23);
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  {
    cudaMalloc((void **)&d_a, n * sizeof(float));
    mod(&d_a);
    hello<<<1, 1>>>(d_a, 23);
  }

  // Offset: d_a is used as lvalue by passing its reference to mod2
  // CHECK: {
  // CHECK-NEXT:   dpct::dpct_malloc((void **)&d_a, n * sizeof(float));
  // CHECK-NEXT:   mod2(d_a);
  // CHECK-NEXT:   {
  // CHECK-NEXT:     std::pair<dpct::buffer_t, size_t> d_a_buf_ct0 = dpct::get_buffer_and_offset(d_a);
  // CHECK-NEXT:     size_t d_a_offset_ct0 = d_a_buf_ct0.second;
  // CHECK-NEXT:     q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto d_a_acc_ct0 = d_a_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             int *d_a_ct0 = (int *)(&d_a_acc_ct0[0] + d_a_offset_ct0);
  // CHECK-NEXT:             hello(d_a_ct0, 23);
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  {
    cudaMalloc((void **)&d_a, n * sizeof(float));
    mod2(d_a);
    hello<<<1, 1>>>(d_a, 23);
  }

  // No offset: d_a is used as rvalue by passing its value to nonmod
  // CHECK: {
  // CHECK-NEXT:   dpct::dpct_malloc((void **)&d_a, n * sizeof(float));
  // CHECK-NEXT:   nonmod(d_a);
  // CHECK-NEXT:   {
  // CHECK-NEXT:     dpct::buffer_t d_a_buf_ct0 = dpct::get_buffer(d_a);
  // CHECK-NEXT:     q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto d_a_acc_ct0 = d_a_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             hello((int *)(&d_a_acc_ct0[0]), 23);
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  {
    cudaMalloc((void **)&d_a, n * sizeof(float));
    nonmod(d_a);
    hello<<<1, 1>>>(d_a, 23);
  }

  // Offset: although d_a remains the same after four pointer arithmetic
  // operations, because it is used as lvalue; this is a trade-off between
  // correctness and complexity.
  // CHECK: {
  // CHECK-NEXT:   dpct::dpct_malloc(&d_a, n * sizeof(float));
  // CHECK-NEXT:   d_a += 2;
  // CHECK-NEXT:   d_a -= 2;
  // CHECK-NEXT:   d_a = d_a + 2;
  // CHECK-NEXT:   d_a = d_a - 2;
  // CHECK-NEXT:   {
  // CHECK-NEXT:     std::pair<dpct::buffer_t, size_t> d_a_buf_ct0 = dpct::get_buffer_and_offset(d_a);
  // CHECK-NEXT:     size_t d_a_offset_ct0 = d_a_buf_ct0.second;
  // CHECK-NEXT:     q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto d_a_acc_ct0 = d_a_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),  
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             int *d_a_ct0 = (int *)(&d_a_acc_ct0[0] + d_a_offset_ct0);
  // CHECK-NEXT:             hello(d_a_ct0, 23);
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  {
    cudaMalloc(&d_a, n * sizeof(float));
    d_a += 2;
    d_a -= 2;
    d_a = d_a + 2;
    d_a = d_a - 2;
    hello<<<1, 1>>>(d_a, 23);
  }

  // No offset: although d_a is used as lvalue several times, the second
  // cudaMalloc makes d_a point to the beginning of a new piece of memory,
  // which makes the offset unnecessary.
  // CHECK: {
  // CHECK-NEXT:   dpct::dpct_malloc(&d_a, n * sizeof(float));
  // CHECK-NEXT:   d_a += 4;
  // CHECK-NEXT:   d_a -= 2;
  // CHECK-NEXT:   d_a = d_a + 8;
  // CHECK-NEXT:   d_a = d_a - 4;
  // CHECK-NEXT:   mod(&d_a);
  // CHECK-NEXT:   mod2(d_a);
  // CHECK-NEXT:   dpct::dpct_malloc(&d_a, n * sizeof(float));
  // CHECK-NEXT:   {
  // CHECK-NEXT:     dpct::buffer_t d_a_buf_ct0 = dpct::get_buffer(d_a);
  // CHECK-NEXT:     q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto d_a_acc_ct0 = d_a_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             hello((int *)(&d_a_acc_ct0[0]), 23);
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  {
    cudaMalloc(&d_a, n * sizeof(float));
    d_a += 4;
    d_a -= 2;
    d_a = d_a + 8;
    d_a = d_a - 4;
    mod(&d_a);
    mod2(d_a);
    cudaMalloc(&d_a, n * sizeof(float));
    hello<<<1, 1>>>(d_a, 23);
  }

  // CHECK: {
  // CHECK-NEXT:   d_a += 2;
  // CHECK-NEXT:   if (n > 23) {
  // CHECK-NEXT:     dpct::dpct_malloc((void **)&d_a, n * sizeof(float));
  // CHECK-NEXT:     {
  // CHECK-NEXT:       dpct::buffer_t d_a_buf_ct0 = dpct::get_buffer(d_a);
  // CHECK-NEXT:       q_ct1.submit(
  // CHECK-NEXT:         [&](sycl::handler &cgh) {
  // CHECK-NEXT:           auto d_a_acc_ct0 = d_a_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:           cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:               hello((int *)(&d_a_acc_ct0[0]));
  // CHECK-NEXT:             });
  // CHECK-NEXT:         });
  // CHECK-NEXT:     }
  // CHECK-NEXT:     if (n > 45) {
  // CHECK-NEXT:         dpct::buffer_t d_a_buf_ct0 = dpct::get_buffer(d_a);
  // CHECK-NEXT:         q_ct1.submit(
  // CHECK-NEXT:           [&](sycl::handler &cgh) {
  // CHECK-NEXT:             auto d_a_acc_ct0 = d_a_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:             cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:               [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:                 hello((int *)(&d_a_acc_ct0[0]));
  // CHECK-NEXT:               });
  // CHECK-NEXT:           });
  // CHECK-NEXT:     }
  // CHECK-NEXT:     if (n > 67) {
  // CHECK-NEXT:       d_a += 2;
  // CHECK-NEXT:       {
  // CHECK-NEXT:         std::pair<dpct::buffer_t, size_t> d_a_buf_ct0 = dpct::get_buffer_and_offset(d_a);
  // CHECK-NEXT:         size_t d_a_offset_ct0 = d_a_buf_ct0.second;
  // CHECK-NEXT:         q_ct1.submit(
  // CHECK-NEXT:           [&](sycl::handler &cgh) {
  // CHECK-NEXT:             auto d_a_acc_ct0 = d_a_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:             cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:               [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:                 int *d_a_ct0 = (int *)(&d_a_acc_ct0[0] + d_a_offset_ct0);
  // CHECK-NEXT:                 hello(d_a_ct0);
  // CHECK-NEXT:               });
  // CHECK-NEXT:           });
  // CHECK-NEXT:       }
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  {
    d_a += 2;
    if (n > 23) {
      cudaMalloc((void **)&d_a, n * sizeof(float));
      hello<<<1, 1>>>(d_a);
      if (n > 45) {
        hello<<<1, 1>>>(d_a);
      }
      if (n > 67) {
        d_a += 2;
        hello<<<1, 1>>>(d_a);
      }
    }
  }
  // CHECK: {
  // CHECK-NEXT:   d_a += 2;
  // CHECK-NEXT:   while (1) {
  // CHECK-NEXT:     dpct::dpct_malloc((void **)&d_a, n * sizeof(float));
  // CHECK-NEXT:     {
  // CHECK-NEXT:       dpct::buffer_t d_a_buf_ct0 = dpct::get_buffer(d_a);
  // CHECK-NEXT:       q_ct1.submit(
  // CHECK-NEXT:         [&](sycl::handler &cgh) {
  // CHECK-NEXT:           auto d_a_acc_ct0 = d_a_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:           cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:               hello((int *)(&d_a_acc_ct0[0]));
  // CHECK-NEXT:             });
  // CHECK-NEXT:         });
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT:   {
  // CHECK-NEXT:     std::pair<dpct::buffer_t, size_t> d_a_buf_ct0 = dpct::get_buffer_and_offset(d_a);
  // CHECK-NEXT:     size_t d_a_offset_ct0 = d_a_buf_ct0.second;
  // CHECK-NEXT:     q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto d_a_acc_ct0 = d_a_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             int *d_a_ct0 = (int *)(&d_a_acc_ct0[0] + d_a_offset_ct0);
  // CHECK-NEXT:             hello(d_a_ct0);
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  {
    d_a += 2;
    while (1) {
      cudaMalloc((void **)&d_a, n * sizeof(float));
      hello<<<1, 1>>>(d_a);
    }
    hello<<<1, 1>>>(d_a);
  }
  // CHECK: {
  // CHECK-NEXT:   d_a += 2;
  // CHECK-NEXT:   for (; 1;) {
  // CHECK-NEXT:     dpct::dpct_malloc((void **)&d_a, n * sizeof(float));
  // CHECK-NEXT:     {
  // CHECK-NEXT:       dpct::buffer_t d_a_buf_ct0 = dpct::get_buffer(d_a);
  // CHECK-NEXT:       q_ct1.submit(
  // CHECK-NEXT:         [&](sycl::handler &cgh) {
  // CHECK-NEXT:           auto d_a_acc_ct0 = d_a_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:           cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:             [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:               hello((int *)(&d_a_acc_ct0[0]), 23);
  // CHECK-NEXT:             });
  // CHECK-NEXT:         });
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT:   {
  // CHECK-NEXT:     std::pair<dpct::buffer_t, size_t> d_a_buf_ct0 = dpct::get_buffer_and_offset(d_a);
  // CHECK-NEXT:     size_t d_a_offset_ct0 = d_a_buf_ct0.second;
  // CHECK-NEXT:     q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto d_a_acc_ct0 = d_a_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             int *d_a_ct0 = (int *)(&d_a_acc_ct0[0] + d_a_offset_ct0);
  // CHECK-NEXT:             hello(d_a_ct0, 23);
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  {
    d_a += 2;
    for (; 1;) {
      cudaMalloc((void **)&d_a, n * sizeof(float));
      hello<<<1, 1>>>(d_a, 23);
    }
    hello<<<1, 1>>>(d_a, 23);
  }

  // Offset: offsets are always there for global variables
  // CHECK: {
  // CHECK-NEXT:   dpct::dpct_malloc((void **)&d_a_global, n * sizeof(float));
  // CHECK-NEXT:   {
  // CHECK-NEXT:     std::pair<dpct::buffer_t, size_t> d_a_global_buf_ct0 = dpct::get_buffer_and_offset(d_a_global);
  // CHECK-NEXT:     size_t d_a_global_offset_ct0 = d_a_global_buf_ct0.second;
  // CHECK-NEXT:     q_ct1.submit(
  // CHECK-NEXT:       [&](sycl::handler &cgh) {
  // CHECK-NEXT:         auto d_a_global_acc_ct0 = d_a_global_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
  // CHECK-EMPTY:
  // CHECK-NEXT:         cgh.parallel_for<dpct_kernel_name<class hello_{{[a-f0-9]+}}>>(
  // CHECK-NEXT:           sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
  // CHECK-NEXT:           [=](sycl::nd_item<3> item_ct1) {
  // CHECK-NEXT:             int *d_a_global_ct0 = (int *)(&d_a_global_acc_ct0[0] + d_a_global_offset_ct0);
  // CHECK-NEXT:             hello(d_a_global_ct0, 23);
  // CHECK-NEXT:           });
  // CHECK-NEXT:       });
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  {
    cudaMalloc((void **)&d_a_global, n * sizeof(float));
    hello<<<1, 1>>>(d_a_global, 23);
  }
}

__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements) {
}

int testVectorAdd(void)
{
    cudaError_t err = cudaSuccess;
    int numElements = 50000;
    size_t size = numElements * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    err = cudaMalloc((void **)&d_A, size);
    err = cudaMalloc((void **)&d_B, size);
    err = cudaMalloc((void **)&d_C, size);

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    // CHECK: /*
    // CHECK: DPCT1049:{{[0-9]+}}: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit query info::device::max_work_group_size. Adjust the workgroup size if needed.
    // CHECK: */
    // CHECK-NEXT: {
    // CHECK-NEXT:   dpct::buffer_t d_A_buf_ct0 = dpct::get_buffer(d_A);
    // CHECK-NEXT:   dpct::buffer_t d_B_buf_ct1 = dpct::get_buffer(d_B);
    // CHECK-NEXT:   dpct::buffer_t d_C_buf_ct2 = dpct::get_buffer(d_C);
    // CHECK-NEXT:   dpct::get_default_queue().submit(
    // CHECK-NEXT:     [&](sycl::handler &cgh) {
    // CHECK-NEXT:       auto d_A_acc_ct0 = d_A_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
    // CHECK-NEXT:       auto d_B_acc_ct1 = d_B_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
    // CHECK-NEXT:       auto d_C_acc_ct2 = d_C_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);
    // CHECK-EMPTY:
    // CHECK-NEXT:       cgh.parallel_for<dpct_kernel_name<class vectorAdd_{{[a-f0-9]+}}>>(
    // CHECK-NEXT:         sycl::nd_range<3>(sycl::range<3>(1, 1, blocksPerGrid) * sycl::range<3>(1, 1, threadsPerBlock), sycl::range<3>(1, 1, threadsPerBlock)), 
    // CHECK-NEXT:         [=](sycl::nd_item<3> item_ct1) {
    // CHECK-NEXT:           vectorAdd((const float *)(&d_A_acc_ct0[0]), (const float *)(&d_B_acc_ct1[0]), (float *)(&d_C_acc_ct2[0]), numElements);
    // CHECK-NEXT:         });
    // CHECK-NEXT:     });
    // CHECK-NEXT: }
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    err = cudaFree(d_A);
    err = cudaFree(d_B);
    err = cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
