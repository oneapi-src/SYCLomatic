//===---------------------- TypeSchemaForSYCL.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

// TODO:  Add more TypeSechma For SYCl Type.

/*************************How to add a new typeschema************************
*
STypeSchemaMap[TypeName] = TypeSchema(TypeName, FieldNumInType, TypeAlignNum, 
                                      TypeSizeInByte, IsVirtual, FilePath,
                                      std::vector<FieldSchema>)

For SYCL, TypeSchema, IsVirtual always is false
And FilePath always be setted as "sycl header path"

How to construct a FieldSchema:
FieldSchema(MemberVarName, ValTypeOfMemberVar, TypeOfMemberVar, IsBasicType,
            SizeOfMemberVar, SizeOfVarType, OffsetOfMemberVar, "None")

ValTypeOfMemberVar : ValType::{ScalarValue, ArrayValue, 
                               Pointer, PointerOfPointer}

IsBasicType : true/false
  true: this type of field var is basic type of c++, like int or float

*
****************************************************************************/

#include "Schema.h"
using namespace clang;
using namespace clang::dpct;
void clang::dpct::setSTypeSchemaMap() {
  STypeSchemaMap["sycl::int2"] = TypeSchema(
      "sycl::int2", 1, 4, 8, false, "sycl header path",
      std::vector<FieldSchema>(1, FieldSchema("m_Data", ValType::ArrayValue,
                                              "int", true, 8, 4, 0, "None")));
}
