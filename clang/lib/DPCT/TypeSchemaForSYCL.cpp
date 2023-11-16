//===---------------------- TypeSchemaForSYCL.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

#include "Schema.h"
using namespace clang;
using namespace clang::dpct;
void clang::dpct::setTypeSchemaMap() {
  TypeSchemaMap["sycl::int2"] = TypeSchema(
      "sycl::int2", 1, 8, false, "sycl header path",
      std::vector<FieldSchema>(1, FieldSchema("m_Data", ValType::ArrayValue,
                                              "int", true, 8, 0, "None")));
}
