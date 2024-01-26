//===--------------- QueryAPIMappingFast.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QueryAPIMapping.h"

namespace clang {
namespace dpct {

void APIMapping::initEntryMap() {
  FastMode = true;
#include "APIMappingRegisterFastCUBPart1.def"
#include "APIMappingRegisterFastDriverPart1.def"
#include "APIMappingRegisterFastMathPart1.def"
#include "APIMappingRegisterFastMathPart2.def"
#include "APIMappingRegisterFastMathPart3.def"
#include "APIMappingRegisterFastMathPart4.def"
#include "APIMappingRegisterFastNCCLPart1.def"
#include "APIMappingRegisterFastRuntimePart1.def"
#include "APIMappingRegisterFastThrustPart1.def"
#include "APIMappingRegisterFastcuBLASPart1.def"
#include "APIMappingRegisterFastcuBLASPart2.def"
#include "APIMappingRegisterFastcuDNNPart1.def"
#include "APIMappingRegisterFastcuFFTPart1.def"
#include "APIMappingRegisterFastcuRANDPart1.def"
#include "APIMappingRegisterFastcuSPARSEPart1.def"
#include "APIMappingRegisterFastcuSolverPart1.def"
  FastMode = false;
#include "APIMappingRegisterSlow.def"
}

} // namespace dpct
} // namespace clang
