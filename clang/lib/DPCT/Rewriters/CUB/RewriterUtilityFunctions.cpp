//===--------------- RewriterUtilityFunctions.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInfo.h"
#include "CallExprRewriterCUB.h"
#include "CallExprRewriterCommon.h"

using namespace clang::dpct;

namespace {
class PrettyTemplatedFunctionNamePrinter {
  std::string Name;
  std::vector<TemplateArgumentInfo> Args;

public:
  PrettyTemplatedFunctionNamePrinter(StringRef Name,
                                     std::vector<TemplateArgumentInfo> &&Args)
      : Name(Name.str()), Args(std::move(Args)) {}
  template <class StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, Name);
    if (!Args.empty()) {
      Stream << '<';
      ArgsPrinter<false, std::vector<TemplateArgumentInfo>>(Args).print(Stream);
      Stream << '>';
    }
  }
};

std::function<PrettyTemplatedFunctionNamePrinter(const CallExpr *)>
makePrettyTemplatedCalleeCreator(std::string CalleeName,
                                 std::vector<size_t> Indexes) {
  return PrinterCreator<
      PrettyTemplatedFunctionNamePrinter, std::string,
      std::function<std::vector<TemplateArgumentInfo>(const CallExpr *)>>(
      CalleeName, [=](const CallExpr *C) -> std::vector<TemplateArgumentInfo> {
        std::vector<TemplateArgumentInfo> Ret;
        auto List = getTemplateArgsList(C);
        for (auto Idx : Indexes) {
          if (Idx < List.size()) {
            Ret.emplace_back(List[Idx]);
          }
        }
        return Ret;
      });
}
} // namespace

#define PRETTY_TEMPLATED_CALLEE(FuncName, ...)                                 \
  makePrettyTemplatedCalleeCreator(FuncName, {__VA_ARGS__})

RewriterMap dpct::createUtilityFunctionsRewriterMap() {
  return RewriterMap{
      // cub::IADD3
      CALL_FACTORY_ENTRY(
          "cub::IADD3",
          CALL("", BO(BinaryOperatorKind::BO_Add,
                      BO(BinaryOperatorKind::BO_Add,
                         CAST_IF_NOT_SAME(LITERAL("unsigned int"), ARG(0)),
                         CAST_IF_NOT_SAME(LITERAL("unsigned int"), ARG(1))),
                      CAST_IF_NOT_SAME(LITERAL("unsigned int"), ARG(2)))))
      // cub::SHL_ADD
      CALL_FACTORY_ENTRY(
          "cub::SHL_ADD",
          CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() +
                                              "extend_shl_clamp",
                                          LITERAL("uint32_t")),
               ARG(0), ARG(1), ARG(2),
               LITERAL(MapNames::getClNamespace() + "plus<>()")))
      // cub::SHR_ADD
      CALL_FACTORY_ENTRY(
          "cub::SHR_ADD",
          CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() +
                                              "extend_shr_clamp",
                                          LITERAL("uint32_t")),
               ARG(0), ARG(1), ARG(2),
               LITERAL(MapNames::getClNamespace() + "plus<>()")))
      // cub::BFE
      CALL_FACTORY_ENTRY("cub::BFE",
                         CALL(MapNames::getDpctNamespace() + "bfe_safe", ARG(0),
                              ARG(1), ARG(2)))
      // cub::BFI
      ASSIGN_FACTORY_ENTRY("cub::BFI", ARG(0),
                           CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                    MapNames::getDpctNamespace() + "bfi_safe",
                                    LITERAL("unsigned")),
                                ARG(2), ARG(1), ARG(3), ARG(4)))
      // cub::LaneId
      MEMBER_CALL_FACTORY_ENTRY(
          "cub::LaneId", MEMBER_CALL(NDITEM, false, LITERAL("get_sub_group")),
          false, "get_local_linear_id")

      // cub::WarpId
      MEMBER_CALL_FACTORY_ENTRY(
          "cub::WarpId", MEMBER_CALL(NDITEM, false, LITERAL("get_sub_group")),
          false, "get_group_linear_id")

      // cub::SyncStream
      CONDITIONAL_FACTORY_ENTRY(
          CheckArgIsDefaultCudaStream(0),
          MEMBER_CALL_FACTORY_ENTRY("cub::SyncStream", QUEUESTR, false, "wait"),
          MEMBER_CALL_FACTORY_ENTRY("cub::SyncStream", ARG(0), true, "wait"))
      // cub::DeviceCount
      CALL_FACTORY_ENTRY("cub::DeviceCount",
                         CALL(MapNames::getDpctNamespace() + "device_count"))
      // cub::DeviceCountUncached
      CALL_FACTORY_ENTRY("cub::DeviceCountUncached",
                         CALL(MapNames::getDpctNamespace() + "device_count"))
      // cub::DeviceCountCachedValue
      CALL_FACTORY_ENTRY("cub::DeviceCountCachedValue",
                         CALL(MapNames::getDpctNamespace() + "device_count"))
      // cub::CurrentDevice
      CALL_FACTORY_ENTRY(
          "cub::CurrentDevice",
          CALL(MapNames::getDpctNamespace() + "get_current_device_id"))
      // cub::PtxVersion
      CONDITIONAL_FACTORY_ENTRY(
          UseSYCLCompat,
          ASSIGN_FACTORY_ENTRY("cub::PtxVersion", ARG(0),
                               LITERAL("SYCLCOMPAT_COMPATIBILITY_TEMP")),
          ASSIGN_FACTORY_ENTRY("cub::PtxVersion", ARG(0),
                               LITERAL("DPCT_COMPATIBILITY_TEMP")))
      // cub::PtxVersionUncached
      CONDITIONAL_FACTORY_ENTRY(
          UseSYCLCompat,
          ASSIGN_FACTORY_ENTRY("cub::PtxVersionUncached", ARG(0),
                               LITERAL("SYCLCOMPAT_COMPATIBILITY_TEMP")),
          ASSIGN_FACTORY_ENTRY("cub::PtxVersionUncached", ARG(0),
                               LITERAL("DPCT_COMPATIBILITY_TEMP")))
      // cub::SmVersion
      ASSIGN_FACTORY_ENTRY(
          "cub::SmVersion", ARG(0),
          BO(BO_Add,
             BO(BO_Mul,
                CALL(MapNames::getDpctNamespace() + "get_major_version",
                     makeDeviceStr()),
                LITERAL("100")),
             BO(BO_Mul,
                CALL(MapNames::getDpctNamespace() + "get_minor_version",
                     makeDeviceStr()),
                LITERAL("10"))))
      // cub::SmVersionUncached
      ASSIGN_FACTORY_ENTRY(
          "cub::SmVersionUncached", ARG(0),
          BO(BO_Add,
             BO(BO_Mul,
                CALL(MapNames::getDpctNamespace() + "get_major_version",
                     makeDeviceStr()),
                LITERAL("100")),
             BO(BO_Mul,
                CALL(MapNames::getDpctNamespace() + "get_minor_version",
                     makeDeviceStr()),
                LITERAL("10"))))
      // cub::RowMajorTid
      MEMBER_CALL_FACTORY_ENTRY("cub::RowMajorTid", NDITEM, /*IsArrow=*/false,
                                "get_local_linear_id")
      // cub::LoadDirectBlocked
      HEADER_INSERT_FACTORY(
          HeaderType::HT_DPCT_GROUP_Utils,
          CALL_FACTORY_ENTRY(
              "cub::LoadDirectBlocked",
              CALL(PRETTY_TEMPLATED_CALLEE(MapNames::getDpctNamespace() +
                                               "group::load_direct_blocked",
                                           0, 1, 2),
                   NDITEM, ARG(1), ARG(2))))
      // cub::LoadDirectStriped
      HEADER_INSERT_FACTORY(
          HeaderType::HT_DPCT_GROUP_Utils,
          CALL_FACTORY_ENTRY(
              "cub::LoadDirectStriped",
              CALL(PRETTY_TEMPLATED_CALLEE(MapNames::getDpctNamespace() +
                                               "group::load_direct_striped",
                                           1, 2, 3),
                   NDITEM, ARG(1), ARG(2))))
      // cub::StoreDirectBlocked
      HEADER_INSERT_FACTORY(
          HeaderType::HT_DPCT_GROUP_Utils,
          CALL_FACTORY_ENTRY(
              "cub::StoreDirectBlocked",
              CALL(PRETTY_TEMPLATED_CALLEE(MapNames::getDpctNamespace() +
                                               "group::store_direct_blocked",
                                           0, 1, 2),
                   NDITEM, ARG(1), ARG(2))))
      // cub::StoreDirectStriped
      HEADER_INSERT_FACTORY(
          HeaderType::HT_DPCT_GROUP_Utils,
          CALL_FACTORY_ENTRY(
              "cub::StoreDirectStriped",
              CALL(PRETTY_TEMPLATED_CALLEE(MapNames::getDpctNamespace() +
                                               "group::store_direct_striped",
                                           1, 2, 3),
                   NDITEM, ARG(1), ARG(2))))
      // cub::ShuffleDown
      SUBGROUPSIZE_FACTORY(
          UINT_MAX,
          MapNames::getDpctNamespace() + "experimental::shift_sub_group_left",
          CONDITIONAL_FACTORY_ENTRY(
              UseNonUniformGroups,
              CALL_FACTORY_ENTRY(
                  "cub::ShuffleDown",
                  CALL(
                      TEMPLATED_CALLEE(MapNames::getDpctNamespace() +
                                           "experimental::shift_sub_group_left",
                                       0, 1),
                      SUBGROUP, ARG(0), ARG(1), ARG(2), ARG(3))),
              UNSUPPORT_FACTORY_ENTRY("cub::ShuffleDown",
                                      Diagnostics::API_NOT_MIGRATED,
                                      LITERAL("cub::ShuffleDown"))))
      // cub::ShuffleUp
      SUBGROUPSIZE_FACTORY(
          UINT_MAX,
          MapNames::getDpctNamespace() + "experimental::shift_sub_group_right",
          CONDITIONAL_FACTORY_ENTRY(
              UseNonUniformGroups,
              CALL_FACTORY_ENTRY(
                  "cub::ShuffleUp",
                  CALL(TEMPLATED_CALLEE(
                           MapNames::getDpctNamespace() +
                               "experimental::shift_sub_group_right",
                           0, 1),
                       SUBGROUP, ARG(0), ARG(1), ARG(2), ARG(3))),
              UNSUPPORT_FACTORY_ENTRY("cub::ShuffleUp",
                                      Diagnostics::API_NOT_MIGRATED,
                                      LITERAL("cub::ShuffleUp"))))};
}
