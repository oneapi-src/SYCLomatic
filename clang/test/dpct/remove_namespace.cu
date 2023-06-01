// RUN: dpct --format-range=none -out-root %T/remove_namespace %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/remove_namespace/remove_namespace.dp.cpp

#include "cuda_runtime.h"
#include <algorithm>

namespace aaa {
// CHECK: // AAA
// CHECK-NEXT: using std::max;
// CHECK-NEXT: // BBB
// AAA
using std::max;
// BBB

void foo(size_t len) {
  size_t maxlen = 0;
  // CHECK: maxlen = max(maxlen, len);
  maxlen = max(maxlen, len);
}
}

namespace bbb {
// CHECK: // AAA
// CHECK-NEXT: using ::std::max;
// CHECK-NEXT: // BBB
// AAA
using ::std::max;
// BBB

void foo(size_t len) {
  size_t maxlen = 0;
  // CHECK: maxlen = max(maxlen, len);
  maxlen = max(maxlen, len);
}
}

namespace ccc {
// CHECK: // AAA
// CHECK-EMPTY:
// CHECK-NEXT: // BBB
// AAA
using ::max;
// BBB

void foo(size_t len) {
  size_t maxlen = 0;
  // CHECK: maxlen = std::max(maxlen, len);
  maxlen = max(maxlen, len);
}
}
