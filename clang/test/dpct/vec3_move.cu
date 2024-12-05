// RUN: dpct --format-range=none -out-root %T/vec3_move %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/vec3_move/vec3_move.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/vec3_move/vec3_move.dp.cpp -o %T/vec3_move/vec3_move.dp.o %}

#include <cuda_runtime.h>
#include <vector>

void test() {
  char *dataHandle;
  int width, height;
  {
    std::vector<uchar3> data;
    // CHECK:  /*
    // CHECK-NEXT:  DPCT1083:{{[0-9]+}}: The size of uchar3 in the migrated code may be different from the original code. You may need to adjust the code.
    // CHECK-NEXT:  */
    data.assign(reinterpret_cast<uchar3 *>(dataHandle),
                reinterpret_cast<uchar3 *>(dataHandle) + width * height);
  }
  {
    // CHECK:  /*
    // CHECK-NEXT:  DPCT1083:{{[0-9]+}}: The size of double3 in the migrated code may be different from the original code. You may need to adjust the code.
    // CHECK-NEXT:  */
    std::vector<double3> data;
    data.assign(reinterpret_cast<double3 *>(dataHandle),
                reinterpret_cast<double3 *>(dataHandle) + width * height);
  }
  {
    std::vector<long3> data;
    // CHECK:  /*
    // CHECK-NEXT:  DPCT1083:{{[0-9]+}}: The size of long3 in the migrated code may be different from the original code. You may need to adjust the code.
    // CHECK-NEXT:  */
    data.assign(reinterpret_cast<long3 *>(dataHandle),
                reinterpret_cast<long3 *>(dataHandle) + width * height);
  }
  {
    std::vector<ulonglong3> data;
    // CHECK:  /*
    // CHECK-NEXT:  DPCT1083:{{[0-9]+}}: The size of ulonglong3 in the migrated code may be different from the original code. You may need to adjust the code.
    // CHECK-NEXT:  */
    data.assign(reinterpret_cast<ulonglong3 *>(dataHandle),
                reinterpret_cast<ulonglong3 *>(dataHandle) + width * height);
  }
}
