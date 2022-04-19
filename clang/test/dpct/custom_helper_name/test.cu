// RUN: dpct --format-range=none -out-root %T/out %s --cuda-include-path="%cuda-path/include" --use-custom-helper=file --custom-helper-name=aaa -- -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/out/include/aaa/aaa.hpp

//      CHECK: //==---- aaa.hpp ----------------------------------*- C++ -*----------------==//
// CHECK-NEXT: //
// CHECK-NEXT: // Copyright (C) Intel Corporation
// CHECK-NEXT: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// CHECK-NEXT: // See https://llvm.org/LICENSE.txt for license information.
// CHECK-NEXT: //
// CHECK-NEXT: //===----------------------------------------------------------------------===//
// CHECK-EMPTY:
// CHECK-NEXT: #ifndef __AAA_HPP__
// CHECK-NEXT: #define __AAA_HPP__
// CHECK: #endif // __AAA_HPP__

int main() {
  float2 f2;
  return 0;
}
