//===--------------- RewriterUtilityFunctions.cpp -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriterCUB.h"
#include "CallExprRewriterCommon.h"

using namespace clang::dpct;

template <class NameT, class... TemplateArgsT>
class PrettyTemplatedNamePrinter {
  NameT Name;
  ArgsPrinter<false, TemplateArgsT...> TAs;

public:
  PrettyTemplatedNamePrinter(NameT Name, TemplateArgsT &&...TAs)
      : Name(Name), TAs(std::forward<TemplateArgsT>(TAs)...) {}
  template <class StreamT> void print(StreamT &Stream) const {
    dpct::print(Stream, Name);
    std::string Tmp;
    llvm::raw_string_ostream OS(Tmp);
    TAs.print(OS);
    if (Tmp.empty())
      return;
    Stream << '<' << Tmp << '>';
  }
};

static inline std::function<PrettyTemplatedNamePrinter<
    StringRef, std::vector<TemplateArgumentInfo>>(const CallExpr *)>
makePrettyTemplatedCalleeCreator(std::string CalleeName,
                                 std::vector<size_t> Indexes) {
  return PrinterCreator<
      PrettyTemplatedNamePrinter<StringRef, std::vector<TemplateArgumentInfo>>,
      std::string,
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
      MEMBER_CALL_FACTORY_ENTRY(
          "cub::DeviceCount",
          CALL(MapNames::getDpctNamespace() + "dev_mgr::instance"), false,
          "device_count")
      // cub::DeviceCountUncached
      MEMBER_CALL_FACTORY_ENTRY(
          "cub::DeviceCountUncached",
          CALL(MapNames::getDpctNamespace() + "dev_mgr::instance"), false,
          "device_count")
      // cub::DeviceCountCachedValue
      MEMBER_CALL_FACTORY_ENTRY(
          "cub::DeviceCountCachedValue",
          CALL(MapNames::getDpctNamespace() + "dev_mgr::instance"), false,
          "device_count")
      // cub::CurrentDevice
      MEMBER_CALL_FACTORY_ENTRY(
          "cub::CurrentDevice",
          CALL(MapNames::getDpctNamespace() + "dev_mgr::instance"), false,
          "current_device_id")
      // cub::PtxVersion
      ASSIGN_FACTORY_ENTRY("cub::PtxVersion", ARG(0),
                           LITERAL("DPCT_COMPATIBILITY_TEMP"))
      // cub::PtxVersionUncached
      ASSIGN_FACTORY_ENTRY("cub::PtxVersionUncached", ARG(0),
                           LITERAL("DPCT_COMPATIBILITY_TEMP"))
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
                   ARG(0), ARG(1), ARG(2))))
      // cub::LoadDirectStriped
      HEADER_INSERT_FACTORY(
          HeaderType::HT_DPCT_GROUP_Utils,
          CALL_FACTORY_ENTRY(
              "cub::LoadDirectStriped",
              CALL(PRETTY_TEMPLATED_CALLEE(MapNames::getDpctNamespace() +
                                               "group::load_direct_striped",
                                           0, 1, 2, 3),
                   ARG(0), ARG(1), ARG(2))))};
}
