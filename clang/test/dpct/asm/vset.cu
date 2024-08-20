// RUN: dpct --format-range=none -out-root %T/asm/vset %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/asm/vset/vset.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl  %T/asm/vset/vset.dp.cpp -o %T/asm/vset/vset.dp.o %}

// clang-format off
__global__ void vset(unsigned *d) {
  unsigned a, b;
  int e, f;

  // CHECK: *d = dpct::extend_compare<int32_t, int32_t>(e, f, std::equal_to<>());
  // CHECK: *d = dpct::extend_compare<int32_t, int32_t>(e, f, std::not_equal_to<>());
  // CHECK: *d = dpct::extend_compare<int32_t, int32_t>(e, f, std::less<>());
  // CHECK: *d = dpct::extend_compare<int32_t, int32_t>(e, f, std::less_equal<>());
  // CHECK: *d = dpct::extend_compare<int32_t, int32_t>(e, f, std::greater<>());
  // CHECK: *d = dpct::extend_compare<int32_t, int32_t>(e, f, std::greater_equal<>());
  asm("vset.s32.s32.eq %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(0));
  asm("vset.s32.s32.ne %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(0));
  asm("vset.s32.s32.lt %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(0));
  asm("vset.s32.s32.le %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(0));
  asm("vset.s32.s32.gt %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(0));
  asm("vset.s32.s32.ge %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(0));
  
  // CHECK: *d = dpct::extend_compare<uint32_t, int32_t>(a, f, std::equal_to<>());
  // CHECK: *d = dpct::extend_compare<uint32_t, int32_t>(a, f, std::not_equal_to<>());
  // CHECK: *d = dpct::extend_compare<uint32_t, int32_t>(a, f, std::less<>());
  // CHECK: *d = dpct::extend_compare<uint32_t, int32_t>(a, f, std::less_equal<>());
  // CHECK: *d = dpct::extend_compare<uint32_t, int32_t>(a, f, std::greater<>());
  // CHECK: *d = dpct::extend_compare<uint32_t, int32_t>(a, f, std::greater_equal<>());
  asm("vset.u32.s32.eq %0, %1, %2, %3;" : "=r"(*d) : "r"(a), "r"(f), "r"(0));
  asm("vset.u32.s32.ne %0, %1, %2, %3;" : "=r"(*d) : "r"(a), "r"(f), "r"(0));
  asm("vset.u32.s32.lt %0, %1, %2, %3;" : "=r"(*d) : "r"(a), "r"(f), "r"(0));
  asm("vset.u32.s32.le %0, %1, %2, %3;" : "=r"(*d) : "r"(a), "r"(f), "r"(0));
  asm("vset.u32.s32.gt %0, %1, %2, %3;" : "=r"(*d) : "r"(a), "r"(f), "r"(0));
  asm("vset.u32.s32.ge %0, %1, %2, %3;" : "=r"(*d) : "r"(a), "r"(f), "r"(0));

  // CHECK: *d = dpct::extend_compare<int32_t, uint32_t>(e, a, std::equal_to<>());
  // CHECK: *d = dpct::extend_compare<int32_t, uint32_t>(e, a, std::not_equal_to<>());
  // CHECK: *d = dpct::extend_compare<int32_t, uint32_t>(e, a, std::less<>());
  // CHECK: *d = dpct::extend_compare<int32_t, uint32_t>(e, a, std::less_equal<>());
  // CHECK: *d = dpct::extend_compare<int32_t, uint32_t>(e, a, std::greater<>());
  // CHECK: *d = dpct::extend_compare<int32_t, uint32_t>(e, a, std::greater_equal<>());
  asm("vset.s32.u32.eq %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(a), "r"(0));
  asm("vset.s32.u32.ne %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(a), "r"(0));
  asm("vset.s32.u32.lt %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(a), "r"(0));
  asm("vset.s32.u32.le %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(a), "r"(0));
  asm("vset.s32.u32.gt %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(a), "r"(0));
  asm("vset.s32.u32.ge %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(a), "r"(0));

  // CHECK: *d = dpct::extend_compare<uint32_t, uint32_t>(a, b, std::equal_to<>());
  // CHECK: *d = dpct::extend_compare<uint32_t, uint32_t>(a, b, std::not_equal_to<>());
  // CHECK: *d = dpct::extend_compare<uint32_t, uint32_t>(a, b, std::less<>());
  // CHECK: *d = dpct::extend_compare<uint32_t, uint32_t>(a, b, std::less_equal<>());
  // CHECK: *d = dpct::extend_compare<uint32_t, uint32_t>(a, b, std::greater<>());
  // CHECK: *d = dpct::extend_compare<uint32_t, uint32_t>(a, b, std::greater_equal<>());
  asm("vset.u32.u32.eq %0, %1, %2, %3;" : "=r"(*d) : "r"(a), "r"(b), "r"(0));
  asm("vset.u32.u32.ne %0, %1, %2, %3;" : "=r"(*d) : "r"(a), "r"(b), "r"(0));
  asm("vset.u32.u32.lt %0, %1, %2, %3;" : "=r"(*d) : "r"(a), "r"(b), "r"(0));
  asm("vset.u32.u32.le %0, %1, %2, %3;" : "=r"(*d) : "r"(a), "r"(b), "r"(0));
  asm("vset.u32.u32.gt %0, %1, %2, %3;" : "=r"(*d) : "r"(a), "r"(b), "r"(0));
  asm("vset.u32.u32.ge %0, %1, %2, %3;" : "=r"(*d) : "r"(a), "r"(b), "r"(0));

  // CHECK: *d = dpct::extend_compare<int32_t, int32_t>(e, f, a, std::equal_to<>(), sycl::plus<>());
  // CHECK: *d = dpct::extend_compare<int32_t, int32_t>(e, f, a, std::not_equal_to<>(), sycl::plus<>());
  // CHECK: *d = dpct::extend_compare<int32_t, int32_t>(e, f, a, std::less<>(), sycl::plus<>());
  // CHECK: *d = dpct::extend_compare<int32_t, int32_t>(e, f, a, std::less_equal<>(), sycl::plus<>());
  // CHECK: *d = dpct::extend_compare<int32_t, int32_t>(e, f, a, std::greater<>(), sycl::plus<>());
  // CHECK: *d = dpct::extend_compare<int32_t, int32_t>(e, f, a, std::greater_equal<>(), sycl::plus<>());
  asm("vset.s32.s32.eq.add %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(a));
  asm("vset.s32.s32.ne.add %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(a));
  asm("vset.s32.s32.lt.add %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(a));
  asm("vset.s32.s32.le.add %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(a));
  asm("vset.s32.s32.gt.add %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(a));
  asm("vset.s32.s32.ge.add %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(a));

  // CHECK: *d = dpct::extend_compare<int32_t, uint32_t>(e, a, b, std::equal_to<>(), sycl::minimum<>());
  // CHECK: *d = dpct::extend_compare<int32_t, uint32_t>(e, a, b, std::not_equal_to<>(), sycl::minimum<>());
  // CHECK: *d = dpct::extend_compare<int32_t, uint32_t>(e, a, b, std::less<>(), sycl::minimum<>());
  // CHECK: *d = dpct::extend_compare<int32_t, uint32_t>(e, a, b, std::less_equal<>(), sycl::minimum<>());
  // CHECK: *d = dpct::extend_compare<int32_t, uint32_t>(e, a, b, std::greater<>(), sycl::minimum<>());
  // CHECK: *d = dpct::extend_compare<int32_t, uint32_t>(e, a, b, std::greater_equal<>(), sycl::minimum<>());
  asm("vset.s32.u32.eq.min %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(a), "r"(b));
  asm("vset.s32.u32.ne.min %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(a), "r"(b));
  asm("vset.s32.u32.lt.min %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(a), "r"(b));
  asm("vset.s32.u32.le.min %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(a), "r"(b));
  asm("vset.s32.u32.gt.min %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(a), "r"(b));
  asm("vset.s32.u32.ge.min %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(a), "r"(b));

  // CHECK: *d = dpct::extend_compare<uint32_t, int32_t>(a, f, a, std::equal_to<>(), sycl::maximum<>());
  // CHECK: *d = dpct::extend_compare<uint32_t, int32_t>(a, f, a, std::not_equal_to<>(), sycl::maximum<>());
  // CHECK: *d = dpct::extend_compare<uint32_t, int32_t>(a, f, a, std::less<>(), sycl::maximum<>());
  // CHECK: *d = dpct::extend_compare<uint32_t, int32_t>(a, f, a, std::less_equal<>(), sycl::maximum<>());
  // CHECK: *d = dpct::extend_compare<uint32_t, int32_t>(a, f, a, std::greater<>(), sycl::maximum<>());
  // CHECK: *d = dpct::extend_compare<uint32_t, int32_t>(a, f, a, std::greater_equal<>(), sycl::maximum<>());
  asm("vset.u32.s32.eq.max %0, %1, %2, %3;" : "=r"(*d) : "r"(a), "r"(f), "r"(a));
  asm("vset.u32.s32.ne.max %0, %1, %2, %3;" : "=r"(*d) : "r"(a), "r"(f), "r"(a));
  asm("vset.u32.s32.lt.max %0, %1, %2, %3;" : "=r"(*d) : "r"(a), "r"(f), "r"(a));
  asm("vset.u32.s32.le.max %0, %1, %2, %3;" : "=r"(*d) : "r"(a), "r"(f), "r"(a));
  asm("vset.u32.s32.gt.max %0, %1, %2, %3;" : "=r"(*d) : "r"(a), "r"(f), "r"(a));
  asm("vset.u32.s32.ge.max %0, %1, %2, %3;" : "=r"(*d) : "r"(a), "r"(f), "r"(a));
}

__device__ int vset2() {
  int i;
  // clang-format off
  // CHECK: { i = dpct::extend_compare<int32_t, int32_t>(32, 40, std::equal_to<>()); if (!(i == 0)) { return 1; } };
  // CHECK: { i = dpct::extend_compare<int32_t, int32_t>(32, 32, std::not_equal_to<>()); if (!(i == 0)) { return 2; } };
  // CHECK: { i = dpct::extend_compare<int32_t, int32_t>(32, 32, std::less<>()); if (!(i == 0)) { return 3; } };
  // CHECK: { i = dpct::extend_compare<int32_t, int32_t>(30, 32, std::less_equal<>()); if (!(i == 1)) { return 4; } };
  // CHECK: { i = dpct::extend_compare<int32_t, int32_t>(32, 31, std::greater<>()); if (!(i == 1)) { return 5; } };
  // CHECK: { i = dpct::extend_compare<int32_t, int32_t>(32, 32, std::greater_equal<>()); if (!(i == 1)) { return 6; } };
  // CHECK: { i = dpct::extend_compare<int32_t, int32_t>(32, 40, 1, std::equal_to<>(), sycl::plus<>()); if (!(i == 1)) { return 7; } };
  // CHECK: { i = dpct::extend_compare<int32_t, int32_t>(32, 32, 3, std::not_equal_to<>(), sycl::minimum<>()); if (!(i == 0)) { return 8; } };
  // CHECK: { i = dpct::extend_compare<int32_t, int32_t>(32, 32, 4, std::less<>(), sycl::maximum<>()); if (!(i == 4)) { return 9; } };
  { asm("vset.s32.s32.eq %0, %1, %2;" : "=r"(i) : "r"(32), "r"(40)); if (!(i == 0)) { return 1; } };
  { asm("vset.s32.s32.ne %0, %1, %2;" : "=r"(i) : "r"(32), "r"(32)); if (!(i == 0)) { return 2; } };
  { asm("vset.s32.s32.lt %0, %1, %2;" : "=r"(i) : "r"(32), "r"(32)); if (!(i == 0)) { return 3; } };
  { asm("vset.s32.s32.le %0, %1, %2;" : "=r"(i) : "r"(30), "r"(32)); if (!(i == 1)) { return 4; } };
  { asm("vset.s32.s32.gt %0, %1, %2;" : "=r"(i) : "r"(32), "r"(31)); if (!(i == 1)) { return 5; } };
  { asm("vset.s32.s32.ge %0, %1, %2;" : "=r"(i) : "r"(32), "r"(32)); if (!(i == 1)) { return 6; } };
  { asm("vset.s32.s32.eq.add %0, %1, %2, %3;" : "=r"(i) : "r"(32), "r"(40), "r"(1)); if (!(i == 1)) { return 7; } };
  { asm("vset.s32.s32.ne.min %0, %1, %2, %3;" : "=r"(i) : "r"(32), "r"(32), "r"(3)); if (!(i == 0)) { return 8; } };
  { asm("vset.s32.s32.lt.max %0, %1, %2, %3;" : "=r"(i) : "r"(32), "r"(32), "r"(4)); if (!(i == 4)) { return 9; } };
  // clang-format on
  return 0;
}
