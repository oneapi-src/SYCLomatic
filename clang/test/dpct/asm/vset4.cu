// RUN: dpct --format-range=none -out-root %T/asm/vset4 %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/asm/vset4/vset4.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl  %T/asm/vset4/vset4.dp.cpp -o %T/asm/vset4/vset4.dp.o %}

// clang-format off
__global__ void vset4(unsigned *d) {
  unsigned a, b, c;
  int e, f, g;

  // CHECK: *d = dpct::extend_vcompare4(e, f, std::equal_to<>());
  // CHECK: *d = dpct::extend_vcompare4(e, f, std::not_equal_to<>());
  // CHECK: *d = dpct::extend_vcompare4(e, f, std::less<>());
  // CHECK: *d = dpct::extend_vcompare4(e, f, std::less_equal<>());
  // CHECK: *d = dpct::extend_vcompare4(e, f, std::greater<>());
  // CHECK: *d = dpct::extend_vcompare4(e, f, std::greater_equal<>());
  asm("vset4.s32.s32.eq %0, %1, %2;" : "=r"(*d) : "r"(e), "r"(f));
  asm("vset4.s32.s32.ne %0, %1, %2;" : "=r"(*d) : "r"(e), "r"(f));
  asm("vset4.s32.s32.lt %0, %1, %2;" : "=r"(*d) : "r"(e), "r"(f));
  asm("vset4.s32.s32.le %0, %1, %2;" : "=r"(*d) : "r"(e), "r"(f));
  asm("vset4.s32.s32.gt %0, %1, %2;" : "=r"(*d) : "r"(e), "r"(f));
  asm("vset4.s32.s32.ge %0, %1, %2;" : "=r"(*d) : "r"(e), "r"(f));
  
  // CHECK: *d = dpct::extend_vcompare4(a, f, std::equal_to<>());
  // CHECK: *d = dpct::extend_vcompare4(a, f, std::not_equal_to<>());
  // CHECK: *d = dpct::extend_vcompare4(a, f, std::less<>());
  // CHECK: *d = dpct::extend_vcompare4(a, f, std::less_equal<>());
  // CHECK: *d = dpct::extend_vcompare4(a, f, std::greater<>());
  // CHECK: *d = dpct::extend_vcompare4(a, f, std::greater_equal<>());
  asm("vset4.u32.s32.eq %0, %1, %2;" : "=r"(*d) : "r"(a), "r"(f));
  asm("vset4.u32.s32.ne %0, %1, %2;" : "=r"(*d) : "r"(a), "r"(f));
  asm("vset4.u32.s32.lt %0, %1, %2;" : "=r"(*d) : "r"(a), "r"(f));
  asm("vset4.u32.s32.le %0, %1, %2;" : "=r"(*d) : "r"(a), "r"(f));
  asm("vset4.u32.s32.gt %0, %1, %2;" : "=r"(*d) : "r"(a), "r"(f));
  asm("vset4.u32.s32.ge %0, %1, %2;" : "=r"(*d) : "r"(a), "r"(f));

  // CHECK: *d = dpct::extend_vcompare4(e, a, std::equal_to<>());
  // CHECK: *d = dpct::extend_vcompare4(e, a, std::not_equal_to<>());
  // CHECK: *d = dpct::extend_vcompare4(e, a, std::less<>());
  // CHECK: *d = dpct::extend_vcompare4(e, a, std::less_equal<>());
  // CHECK: *d = dpct::extend_vcompare4(e, a, std::greater<>());
  // CHECK: *d = dpct::extend_vcompare4(e, a, std::greater_equal<>());
  asm("vset4.s32.u32.eq %0, %1, %2;" : "=r"(*d) : "r"(e), "r"(a));
  asm("vset4.s32.u32.ne %0, %1, %2;" : "=r"(*d) : "r"(e), "r"(a));
  asm("vset4.s32.u32.lt %0, %1, %2;" : "=r"(*d) : "r"(e), "r"(a));
  asm("vset4.s32.u32.le %0, %1, %2;" : "=r"(*d) : "r"(e), "r"(a));
  asm("vset4.s32.u32.gt %0, %1, %2;" : "=r"(*d) : "r"(e), "r"(a));
  asm("vset4.s32.u32.ge %0, %1, %2;" : "=r"(*d) : "r"(e), "r"(a));

  // CHECK: *d = dpct::extend_vcompare4(a, b, std::equal_to<>());
  // CHECK: *d = dpct::extend_vcompare4(a, b, std::not_equal_to<>());
  // CHECK: *d = dpct::extend_vcompare4(a, b, std::less<>());
  // CHECK: *d = dpct::extend_vcompare4(a, b, std::less_equal<>());
  // CHECK: *d = dpct::extend_vcompare4(a, b, std::greater<>());
  // CHECK: *d = dpct::extend_vcompare4(a, b, std::greater_equal<>());
  asm("vset4.u32.u32.eq %0, %1, %2;" : "=r"(*d) : "r"(a), "r"(b));
  asm("vset4.u32.u32.ne %0, %1, %2;" : "=r"(*d) : "r"(a), "r"(b));
  asm("vset4.u32.u32.lt %0, %1, %2;" : "=r"(*d) : "r"(a), "r"(b));
  asm("vset4.u32.u32.le %0, %1, %2;" : "=r"(*d) : "r"(a), "r"(b));
  asm("vset4.u32.u32.gt %0, %1, %2;" : "=r"(*d) : "r"(a), "r"(b));
  asm("vset4.u32.u32.ge %0, %1, %2;" : "=r"(*d) : "r"(a), "r"(b));

  // CHECK: *d = dpct::extend_vcompare4(e, f, a, std::equal_to<>(), sycl::plus<>());
  // CHECK: *d = dpct::extend_vcompare4(e, f, a, std::not_equal_to<>(), sycl::plus<>());
  // CHECK: *d = dpct::extend_vcompare4(e, f, a, std::less<>(), sycl::plus<>());
  // CHECK: *d = dpct::extend_vcompare4(e, f, a, std::less_equal<>(), sycl::plus<>());
  // CHECK: *d = dpct::extend_vcompare4(e, f, a, std::greater<>(), sycl::plus<>());
  // CHECK: *d = dpct::extend_vcompare4(e, f, a, std::greater_equal<>(), sycl::plus<>());
  asm("vset4.s32.s32.eq.add %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(a));
  asm("vset4.s32.s32.ne.add %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(a));
  asm("vset4.s32.s32.lt.add %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(a));
  asm("vset4.s32.s32.le.add %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(a));
  asm("vset4.s32.s32.gt.add %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(a));
  asm("vset4.s32.s32.ge.add %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(a));
}
