//===--------------- MigrationStatistics..cpp-----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "MigrationStatistics.h"

std::map<std::string, bool> MigrationStatistics::MigrationTable{
#define ENTRY(INTERFACENAME, APINAME, VALUE, FLAG, TARGET, COMMENT)            \
  {#APINAME, VALUE},
#define ENTRY_MEMBER_FUNCTION(INTERFACEOBJNAME, OBJNAME, INTERFACENAME,        \
                              APINAME, VALUE, FLAG, TARGET, COMMENT)           \
  {#OBJNAME "::" #APINAME, VALUE},
#include "SrcAPI/APINames.inc"
#include "SrcAPI/APINames_CUB.inc"
#include "SrcAPI/APINames_NCCL.inc"
#include "SrcAPI/APINames_NVML.inc"
#include "SrcAPI/APINames_NVTX.inc"
#include "SrcAPI/APINames_cuBLAS.inc"
#include "SrcAPI/APINames_cuDNN.inc"
#include "SrcAPI/APINames_cuFFT.inc"
#include "SrcAPI/APINames_cuRAND.inc"
#include "SrcAPI/APINames_cuSOLVER.inc"
#include "SrcAPI/APINames_cuSPARSE.inc"
#include "SrcAPI/APINames_cudnn_frontend.inc"
#include "SrcAPI/APINames_nvGRAPH.inc"
#include "SrcAPI/APINames_nvJPEG.inc"
#include "SrcAPI/APINames_thrust.inc"
#include "SrcAPI/APINames_wmma.inc"
#undef ENTRY_MEMBER_FUNCTION
#undef ENTRY
};

std::map<std::string, bool> MigrationStatistics::TypeMigrationTable{
#define ENTRY_TYPE(TYPENAME, VALUE, FLAG, TARGET, COMMENT) {#TYPENAME, VALUE},
#include "SrcAPI/TypeNames.inc"
#undef ENTRY_TYPE
};

bool MigrationStatistics::IsMigrated(const std::string &APIName) {
  auto Search = MigrationTable.find(APIName);
  if (Search != MigrationTable.end()) {
    return Search->second;
  } else {
#ifdef DPCT_DEBUG_BUILD
    llvm::errs() << "[NOTE] Find new API \"" << APIName
                 << "\" , please update migrated API database.\n";
    ShowStatus(MigrationError);
    dpctExit(MigrationError);
#endif
    return false;
  }
}

std::vector<std::string> MigrationStatistics::GetAllAPINames(void) {
  std::vector<std::string> AllAPINames;
  for (const auto &APIName : MigrationTable) {
    AllAPINames.push_back(APIName.first);
  }

  return AllAPINames;
}
std::map<std::string, bool> &MigrationStatistics::GetTypeTable(void) {
  return TypeMigrationTable;
}