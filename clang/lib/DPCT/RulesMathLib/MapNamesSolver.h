//===--------------- MapNamesSolver.h ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_RULESMATHLIB_MAPNAMES_SOLVER_H
#define DPCT_RULESMATHLIB_MAPNAMES_SOLVER_H
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

namespace clang {
namespace dpct {

class MapNamesSolver {

  using MapTy = std::map<std::string, std::string>;

public:
  struct SOLVERFuncReplInfo {
    static SOLVERFuncReplInfo migrateBuffer(std::vector<int> bi,
                                            std::vector<std::string> bt,
                                            std::string s) {
      MapNamesSolver::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = std::move(bi);
      repl.BufferTypeInfo = std::move(bt);
      repl.ReplName = std::move(s);
      return repl;
    };

    static SOLVERFuncReplInfo
    migrateBufferAndRedundant(std::vector<int> bi, std::vector<std::string> bt,
                              std::vector<int> ri, std::string s) {
      MapNamesSolver::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = std::move(bi);
      repl.BufferTypeInfo = std::move(bt);
      repl.RedundantIndexInfo = std::move(ri);
      repl.ReplName = std::move(s);
      return repl;
    };

    static SOLVERFuncReplInfo migrateBufferMoveRedundantAndWSS(
        std::vector<int> bi, std::vector<std::string> bt, std::vector<int> ri,
        std::vector<int> mfi, std::vector<int> mti, std::vector<int> wssid,
        std::vector<int> wssi, std::string wssfn, std::string s) {
      MapNamesSolver::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = std::move(bi);
      repl.BufferTypeInfo = std::move(bt);
      repl.RedundantIndexInfo = std::move(ri);
      repl.MoveFrom = std::move(mfi);
      repl.MoveTo = std::move(mti);
      repl.WSSizeInsertAfter = std::move(wssid);
      repl.WSSizeInfo = std::move(wssi);
      repl.WSSFuncName = std::move(wssfn);
      repl.ReplName = std::move(s);
      return repl;
    };

    static SOLVERFuncReplInfo
    migrateReturnAndRedundant(bool q2d, std::vector<int> ri, std::string s) {
      MapNamesSolver::SOLVERFuncReplInfo repl;
      repl.ReturnValue = true;
      repl.RedundantIndexInfo = std::move(ri);
      repl.ReplName = std::move(s);
      return repl;
    };

    static SOLVERFuncReplInfo migrateDeviceAndCopy(bool q2d,
                                                   std::vector<int> cfi,
                                                   std::vector<int> cti,
                                                   std::string s) {
      MapNamesSolver::SOLVERFuncReplInfo repl;
      repl.ReturnValue = q2d;
      repl.CopyFrom = std::move(cfi);
      repl.CopyTo = std::move(cti);
      repl.ReplName = std::move(s);
      return repl;
    };

    static SOLVERFuncReplInfo
    migrateBufferAndMissed(std::vector<int> bi, std::vector<std::string> bt,
                           std::vector<int> mafl, std::vector<int> mai,
                           std::vector<bool> mab, std::vector<std::string> mat,
                           std::vector<std::string> man, std::string s) {
      MapNamesSolver::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = std::move(bi);
      repl.BufferTypeInfo = std::move(bt);
      repl.MissedArgumentFinalLocation = std::move(mafl);
      repl.MissedArgumentInsertBefore = std::move(mai);
      repl.MissedArgumentIsBuffer = std::move(mab);
      repl.MissedArgumentType = std::move(mat);
      repl.MissedArgumentName = std::move(man);
      repl.ReplName = std::move(s);
      return repl;
    };

    static SOLVERFuncReplInfo migrateReturnCopyRedundantAndMissed(
        bool q2d, std::vector<int> ri, std::vector<int> cfi,
        std::vector<int> cti, std::vector<int> mafl, std::vector<int> mai,
        std::vector<bool> mab, std::vector<std::string> mat,
        std::vector<std::string> man, std::string s) {
      MapNamesSolver::SOLVERFuncReplInfo repl;
      repl.ReturnValue = q2d;
      repl.RedundantIndexInfo = std::move(ri);
      repl.CopyFrom = std::move(cfi);
      repl.CopyTo = std::move(cti);
      repl.MissedArgumentFinalLocation = std::move(mafl);
      repl.MissedArgumentInsertBefore = std::move(mai);
      repl.MissedArgumentIsBuffer = std::move(mab);
      repl.MissedArgumentType = std::move(mat);
      repl.MissedArgumentName = std::move(man);
      repl.ReplName = std::move(s);
      return repl;
    };

    static SOLVERFuncReplInfo migrateReturnRedundantAndMissed(
        bool q2d, std::vector<int> ri, std::vector<int> mafl,
        std::vector<int> mai, std::vector<bool> mab,
        std::vector<std::string> mat, std::vector<std::string> man,
        std::string s) {
      MapNamesSolver::SOLVERFuncReplInfo repl;
      repl.ReturnValue = q2d;
      repl.RedundantIndexInfo = std::move(ri);
      repl.MissedArgumentFinalLocation = std::move(mafl);
      repl.MissedArgumentInsertBefore = std::move(mai);
      repl.MissedArgumentIsBuffer = std::move(mab);
      repl.MissedArgumentType = std::move(mat);
      repl.MissedArgumentName = std::move(man);
      repl.ReplName = std::move(s);
      return repl;
    };

    static SOLVERFuncReplInfo migrateBufferAndCast(std::vector<int> bi,
                                                   std::vector<std::string> bt,
                                                   std::vector<int> ci,
                                                   std::vector<std::string> ct,
                                                   std::string s) {
      MapNamesSolver::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = std::move(bi);
      repl.BufferTypeInfo = std::move(bt);
      repl.CastIndexInfo = std::move(ci);
      repl.CastTypeInfo = std::move(ct);
      repl.ReplName = std::move(s);
      return repl;
    };

    static SOLVERFuncReplInfo migrateBufferRedundantAndCast(
        std::vector<int> bi, std::vector<std::string> bt, std::vector<int> ri,
        std::vector<int> ci, std::vector<std::string> ct, std::string s) {
      MapNamesSolver::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = std::move(bi);
      repl.BufferTypeInfo = std::move(bt);
      repl.RedundantIndexInfo = std::move(ri);
      repl.CastIndexInfo = std::move(ci);
      repl.CastTypeInfo = std::move(ct);
      repl.ReplName = std::move(s);
      return repl;
    };

    static SOLVERFuncReplInfo migrateBufferRedundantAndWS(
        std::vector<int> bi, std::vector<std::string> bt, std::vector<int> ri,
        std::vector<int> wsi, std::vector<int> wss, std::string wsn,
        std::string s) {
      MapNamesSolver::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = std::move(bi);
      repl.BufferTypeInfo = std::move(bt);
      repl.RedundantIndexInfo = std::move(ri);
      repl.WorkspaceIndexInfo = std::move(wsi);
      repl.WorkspaceSizeInfo = std::move(wss);
      repl.WorkspaceSizeFuncName = std::move(wsn);
      repl.ReplName = std::move(s);
      return repl;
    };

    static SOLVERFuncReplInfo migrateBufferMissedAndCast(
        std::vector<int> bi, std::vector<std::string> bt, std::vector<int> mafl,
        std::vector<int> mai, std::vector<bool> mab,
        std::vector<std::string> mat, std::vector<std::string> man,
        std::vector<int> ci, std::vector<std::string> ct, std::string s) {
      MapNamesSolver::SOLVERFuncReplInfo repl;
      repl.BufferIndexInfo = std::move(bi);
      repl.BufferTypeInfo = std::move(bt);
      repl.MissedArgumentFinalLocation = std::move(mafl);
      repl.MissedArgumentInsertBefore = std::move(mai);
      repl.MissedArgumentIsBuffer = std::move(mab);
      repl.MissedArgumentType = std::move(mat);
      repl.MissedArgumentName = std::move(man);
      repl.CastIndexInfo = std::move(ci);
      repl.CastTypeInfo = std::move(ct);
      repl.ReplName = std::move(s);
      return repl;
    };

    static SOLVERFuncReplInfo
    migrateReturnRedundantAndCast(bool q2d, std::vector<int> ri,
                                  std::vector<int> ci,
                                  std::vector<std::string> ct, std::string s) {
      MapNamesSolver::SOLVERFuncReplInfo repl;
      repl.ReturnValue = q2d;
      repl.RedundantIndexInfo = std::move(ri);
      repl.CastIndexInfo = std::move(ci);
      repl.CastTypeInfo = std::move(ct);
      repl.ReplName = std::move(s);
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

  static const MapTy SOLVEREnumsMap;
  static const std::map<std::string, MapNamesSolver::SOLVERFuncReplInfo>
      SOLVERFuncReplInfoMap;
  static std::unordered_set<std::string> SOLVERAPIWithRewriter;

}; // class MapNamesSolver

} // namespace dpct
} // namespace clang

#endif //! DPCT_RULESMATHLIB_MAPNAMES_SOLVER_H