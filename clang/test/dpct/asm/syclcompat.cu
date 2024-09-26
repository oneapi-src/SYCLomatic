// RUN: dpct --use-syclcompat --format-range=none -out-root %T/asm/syclcompat %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/asm/syclcompat/syclcompat.dp.cpp

// clang-format off
__global__ void vset(unsigned *d) {
  int e, f;
  // CHECK: DPCT1131:{{.*}}: The migration of "vset.s32.s32.eq %0, %1, %2, %3;" is not currently supported with SYCLcompat. Please adjust the code manually.
  asm("vset.s32.s32.eq %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(0));
}

__device__ int vset2() {
  int i;
  // CHECK: DPCT1131:{{.*}}: The migration of "vset.s32.s32.eq %0, %1, %2;" is not currently supported with SYCLcompat. Please adjust the code manually.
  asm("vset.s32.s32.eq %0, %1, %2;" : "=r"(i) : "r"(32), "r"(40));
  return 0;
}

// clang-format off
__global__ void vset4(unsigned *d) {
  int e, f;
  // CHECK: DPCT1131:{{.*}}: The migration of "vset4.s32.s32.eq %0, %1, %2, %3;" is not currently supported with SYCLcompat. Please adjust the code manually.
  asm("vset4.s32.s32.eq %0, %1, %2, %3;" : "=r"(*d) : "r"(e), "r"(f), "r"(0));
}
// clang-format on
