//===--------------- MapNames.h -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_MAPNAMES_H
#define DPCT_MAPNAMES_H

#include "CustomHelperFiles.h"
#include "Rules.h"
#include "Utility.h"
#include "ValidateArguments.h"
#include <map>
#include <set>

namespace clang {
namespace dpct {
enum class KernelArgType;
enum class HelperFileEnum : unsigned int;
struct HelperFunc;
} // namespace dpct
} // namespace clang

const std::string StringLiteralUnsupported{"UNSUPPORTED"};

#define SUPPORTEDVECTORTYPENAMES                                               \
  "char1", "uchar1", "char2", "uchar2", "char3", "uchar3", "char4", "uchar4",  \
      "short1", "ushort1", "short2", "ushort2", "short3", "ushort3", "short4", \
      "ushort4", "int1", "uint1", "int2", "uint2", "int3", "uint3", "int4",    \
      "uint4", "long1", "ulong1", "long2", "ulong2", "long3", "ulong3",        \
      "long4", "ulong4", "float1", "float2", "float3", "float4", "longlong1",  \
      "ulonglong1", "longlong2", "ulonglong2", "longlong3", "ulonglong3",      \
      "longlong4", "ulonglong4", "double1", "double2", "double3", "double4"

/// Record mapping between names
class MapNames {
  static std::vector<std::string> ClNamespace;
  static std::vector<std::string> DpctNamespace;

public:
  static void setExplicitNamespaceMap();
  // KeepNamespace = true for function or type that need avoid ambiguous.
  // Example: sycl::exception <--> std::exception
  // IsMathFunc = true for namespace before math functions.
  // Example: sycl::exp
  static std::string getClNamespace(bool KeepNamespace = false,
                                    bool IsMathFunc = false);
  static std::string getDpctNamespace(bool KeepNamespace = false);

  struct SOLVERFuncReplInfo {
    static SOLVERFuncReplInfo migrateBuffer(std::vector<int> bi,
                                            std::vector<std::string> bt,
                                            std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = bi;
      repl.BufferTypeInfo = bt;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo
    migrateBufferAndRedundant(std::vector<int> bi, std::vector<std::string> bt,
                              std::vector<int> ri, std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = bi;
      repl.BufferTypeInfo = bt;
      repl.RedundantIndexInfo = ri;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo migrateBufferMoveRedundantAndWSS(
        std::vector<int> bi, std::vector<std::string> bt, std::vector<int> ri,
        std::vector<int> mfi, std::vector<int> mti, std::vector<int> wssid,
        std::vector<int> wssi, std::string wssfn, std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = bi;
      repl.BufferTypeInfo = bt;
      repl.RedundantIndexInfo = ri;
      repl.MoveFrom = mfi;
      repl.MoveTo = mti;
      repl.WSSizeInsertAfter = wssid;
      repl.WSSizeInfo = wssi;
      repl.WSSFuncName = wssfn;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo
    migrateReturnAndRedundant(bool q2d, std::vector<int> ri, std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.ReturnValue = true;
      repl.RedundantIndexInfo = ri;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo migrateDeviceAndCopy(bool q2d,
                                                   std::vector<int> cfi,
                                                   std::vector<int> cti,
                                                   std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.ReturnValue = q2d;
      repl.CopyFrom = cfi;
      repl.CopyTo = cti;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo
    migrateBufferAndMissed(std::vector<int> bi, std::vector<std::string> bt,
                           std::vector<int> mafl, std::vector<int> mai,
                           std::vector<bool> mab, std::vector<std::string> mat,
                           std::vector<std::string> man, std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = bi;
      repl.BufferTypeInfo = bt;
      repl.MissedArgumentFinalLocation = mafl;
      repl.MissedArgumentInsertBefore = mai;
      repl.MissedArgumentIsBuffer = mab;
      repl.MissedArgumentType = mat;
      repl.MissedArgumentName = man;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo migrateReturnCopyRedundantAndMissed(
        bool q2d, std::vector<int> ri, std::vector<int> cfi,
        std::vector<int> cti, std::vector<int> mafl, std::vector<int> mai,
        std::vector<bool> mab, std::vector<std::string> mat,
        std::vector<std::string> man, std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.ReturnValue = q2d;
      repl.RedundantIndexInfo = ri;
      repl.CopyFrom = cfi;
      repl.CopyTo = cti;
      repl.MissedArgumentFinalLocation = mafl;
      repl.MissedArgumentInsertBefore = mai;
      repl.MissedArgumentIsBuffer = mab;
      repl.MissedArgumentType = mat;
      repl.MissedArgumentName = man;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo migrateReturnRedundantAndMissed(
        bool q2d, std::vector<int> ri, std::vector<int> mafl,
        std::vector<int> mai, std::vector<bool> mab,
        std::vector<std::string> mat, std::vector<std::string> man,
        std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.ReturnValue = q2d;
      repl.RedundantIndexInfo = ri;
      repl.MissedArgumentFinalLocation = mafl;
      repl.MissedArgumentInsertBefore = mai;
      repl.MissedArgumentIsBuffer = mab;
      repl.MissedArgumentType = mat;
      repl.MissedArgumentName = man;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo migrateBufferAndCast(std::vector<int> bi,
                                                   std::vector<std::string> bt,
                                                   std::vector<int> ci,
                                                   std::vector<std::string> ct,
                                                   std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = bi;
      repl.BufferTypeInfo = bt;
      repl.CastIndexInfo = ci;
      repl.CastTypeInfo = ct;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo migrateBufferRedundantAndCast(
        std::vector<int> bi, std::vector<std::string> bt, std::vector<int> ri,
        std::vector<int> ci, std::vector<std::string> ct, std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = bi;
      repl.BufferTypeInfo = bt;
      repl.RedundantIndexInfo = ri;
      repl.CastIndexInfo = ci;
      repl.CastTypeInfo = ct;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo migrateBufferRedundantAndWS(
        std::vector<int> bi, std::vector<std::string> bt, std::vector<int> ri,
        std::vector<int> wsi, std::vector<int> wss, std::string wsn,
        std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = bi;
      repl.BufferTypeInfo = bt;
      repl.RedundantIndexInfo = ri;
      repl.WorkspaceIndexInfo = wsi;
      repl.WorkspaceSizeInfo = wss;
      repl.WorkspaceSizeFuncName = wsn;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo migrateBufferMissedAndCast(
        std::vector<int> bi, std::vector<std::string> bt, std::vector<int> mafl,
        std::vector<int> mai, std::vector<bool> mab,
        std::vector<std::string> mat, std::vector<std::string> man,
        std::vector<int> ci, std::vector<std::string> ct, std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = bi;
      repl.BufferTypeInfo = bt;
      repl.MissedArgumentFinalLocation = mafl;
      repl.MissedArgumentInsertBefore = mai;
      repl.MissedArgumentIsBuffer = mab;
      repl.MissedArgumentType = mat;
      repl.MissedArgumentName = man;
      repl.CastIndexInfo = ci;
      repl.CastTypeInfo = ct;
      repl.ReplName = s;
      return repl;
    };

    static SOLVERFuncReplInfo
    migrateReturnRedundantAndCast(bool q2d, std::vector<int> ri,
                                  std::vector<int> ci,
                                  std::vector<std::string> ct, std::string s) {
      MapNames::SOLVERFuncReplInfo repl;
      repl.ReturnValue = q2d;
      repl.RedundantIndexInfo = ri;
      repl.CastIndexInfo = ci;
      repl.CastTypeInfo = ct;
      repl.ReplName = s;
      return repl;
    };

    std::vector<int> BufferIndexInfo;
    std::vector<std::string> BufferTypeInfo;

    // will be replaced by empty string""
    std::vector<int> RedundantIndexInfo;

    std::vector<int> CastIndexInfo;
    std::vector<std::string> CastTypeInfo;

    std::vector<int> MissedArgumentFinalLocation;
    std::vector<int> MissedArgumentInsertBefore; // index of original argument
    std::vector<bool> MissedArgumentIsBuffer;
    std::vector<std::string> MissedArgumentType;
    std::vector<std::string> MissedArgumentName;

    std::vector<int> WorkspaceIndexInfo;
    std::vector<int> WorkspaceSizeInfo;
    std::string WorkspaceSizeFuncName;

    std::vector<int> WSSizeInsertAfter;
    std::vector<int> WSSizeInfo;
    std::string WSSFuncName;

    std::vector<int> CopyFrom;
    std::vector<int> CopyTo;
    std::vector<int> MoveFrom;
    std::vector<int> MoveTo;
    bool ReturnValue = false;
    std::string ReplName;
  };

  struct BLASFuncReplInfo {
    std::vector<int> BufferIndexInfo;
    std::vector<int> PointerIndexInfo;
    std::vector<std::string> BufferTypeInfo;
    std::vector<int> OperationIndexInfo;
    int FillModeIndexInfo;
    int SideModeIndexInfo;
    int DiagTypeIndexInfo;
    std::string ReplName;
  };

  struct BLASFuncComplexReplInfo {
    std::vector<int> BufferIndexInfo;
    std::vector<int> PointerIndexInfo;
    std::vector<std::string> BufferTypeInfo;
    std::vector<std::string> PointerTypeInfo;
    std::vector<int> OperationIndexInfo;
    int FillModeIndexInfo;
    int SideModeIndexInfo;
    int DiagTypeIndexInfo;
    std::string ReplName;
  };

  struct RandomGenerateFuncReplInfo {
    std::string BufferTypeInfo;
    std::string DistributeType;
    std::string ValueType;
  };

  struct BLASGemmExTypeInfo {
    std::string OriginScalarType;
    std::string ScalarType;
    std::string OriginABType;
    std::string ABType;
    std::string OriginCType;
    std::string CType;
  };

  struct ThrustFuncReplInfo {
    std::string ReplName;
    std::string ExtraParam;
  };

  using MapTy = std::map<std::string, std::string>;
  using SetTy = std::set<std::string>;
  using ThrustMapTy = std::map<std::string, ThrustFuncReplInfo>;

  static const SetTy SupportedVectorTypes;
  static const std::map<std::string, int> VectorTypeMigratedTypeSizeMap;
  static const std::map<clang::dpct::KernelArgType, int> KernelArgTypeSizeMap;
  static int getArrayTypeSize(const int Dim);
  static const MapTy RemovedAPIWarningMessage;
  static std::unordered_map<std::string, std::shared_ptr<TypeNameRule>>
      TypeNamesMap;
  static std::unordered_map<std::string, std::shared_ptr<ClassFieldRule>>
      ClassFieldMap;
  static std::unordered_map<std::string, std::shared_ptr<TypeNameRule>> 
      CuDNNTypeNamesMap;
  static const MapTy Dim3MemberNamesMap;
  static const MapTy MacrosMap;
  static std::unordered_map<std::string, MacroMigrationRule> MacroRuleMap;
  static std::unordered_map<std::string, MetaRuleObject &> HeaderRuleMap;
  static const MapTy SPBLASEnumsMap;
  static const MapTy BLASEnumsMap;
  static std::map<std::string, MapNames::BLASFuncReplInfo> BLASFuncReplInfoMap;
  static const std::map<std::string, MapNames::BLASFuncComplexReplInfo>
      BLASFuncComplexReplInfoMap;
  static const SetTy ThrustFileExcludeSet;
  static ThrustMapTy ThrustFuncNamesMap;
  static std::map<std::string, clang::dpct::HelperFeatureEnum>
      ThrustFuncNamesHelperFeaturesMap;

  static const std::map<std::string, MapNames::BLASFuncComplexReplInfo>
      LegacyBLASFuncReplInfoMap;

  static const std::set<std::string> MustSyncBLASFunc;
  static const std::map<std::string, std::pair<std::string, int>>
      MaySyncBLASFunc;
  // This map is only used for non-usm.
  static const std::map<std::string, std::map<int, std::string>>
      MaySyncBLASFuncWithMultiArgs;

  static std::map<std::string, MapNames::BLASGemmExTypeInfo>
      BLASTGemmExTypeInfoMap;

  static const MapTy SOLVEREnumsMap;
  static const MapTy DriverEnumsMap;
  static const std::map<std::string, MapNames::SOLVERFuncReplInfo>
      SOLVERFuncReplInfoMap;

  static MapTy ITFName;
  static const MapTy RandomEngineTypeMap;
  static const std::map<std::string, MapNames::RandomGenerateFuncReplInfo>
      RandomGenerateFuncReplInfoMap;

  static const MapTy DeviceRandomGeneratorTypeMap;

  static const std::map<std::string, std::vector<unsigned int>>
      FFTPlanAPINeedParenIdxMap;

  static MapTy BLASComputingAPIWithRewriter;
  static std::unordered_set<std::string> SOLVERAPIWithRewriter;
  static std::unordered_set<std::string> SPARSEAPIWithRewriter;

  static const std::unordered_set<std::string> CooperativeGroupsAPISet;

  template<class T>
  inline static const std::string &findReplacedName(
      const std::unordered_map<std::string, std::shared_ptr<T>> &Map,
      const std::string &Name) {
    static const std::string EmptyString;

    auto Itr = Map.find(Name);
    if (Itr == Map.end())
      return EmptyString;
    return Itr->second->NewName;
  }
  inline static const std::string &findReplacedName(const MapTy &Map,
                                                    const std::string &Name) {
    static const std::string EmptyString;

    auto Itr = Map.find(Name);
    if (Itr == Map.end())
      return EmptyString;
    return Itr->second;
  }
  template<class T>
  static bool replaceName(
      const std::unordered_map<std::string, std::shared_ptr<T>> &Map,
      std::string &Name) {
    auto &Result = findReplacedName(Map, Name);
    if (Result.empty())
      return false;
    Name = Result;
    return true;
  }
  static bool replaceName(const MapTy &Map, std::string &Name) {
    auto &Result = findReplacedName(Map, Name);
    if (Result.empty())
      return false;
    Name = Result;
    return true;
  }
  static bool isInSet(const SetTy &Set, std::string &Name) {
    return Set.find(Name) != Set.end();
  }

  static const MapNames::MapTy MemberNamesMap;
  static const MapNames::MapTy FunctionAttrMap;
  static const MapNames::SetTy HostAllocSet;
  static MapNames::MapTy MathFuncNameMap;

  static std::unordered_map<std::string, std::string> AtomicFuncNamesMap;
  static const MapNames::SetTy PredefinedStreamName;

  /// {Original API, {ToType, FromType}}
  static std::unordered_map<std::string, std::pair<std::string, std::string>>
      MathTypeCastingMap;
};

class MigrationStatistics {
private:
  static std::map<std::string /*API Name*/, bool /*Is Migrated*/>
      MigrationTable;
  static std::map<std::string /*Type Name*/, bool /*Is Migrated*/>
      TypeMigrationTable;

public:
  static bool IsMigrated(const std::string &APIName);
  static std::vector<std::string> GetAllAPINames(void);
  static std::map<std::string, bool> &GetTypeTable(void);
};

#endif
