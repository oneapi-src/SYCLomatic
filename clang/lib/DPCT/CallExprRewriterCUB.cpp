//===--------------- CallExprRewriterCUB.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CUBAPIMigration.h"
#include "CallExprRewriter.h"
#include "CallExprRewriterCommon.h"

namespace clang {
namespace dpct {

class CheckCubRedundantFunctionCall {
public:
  bool operator()(const CallExpr *C) {
    return CubDeviceLevelRule::isRedundantCallExpr(C);
  }
};

inline std::shared_ptr<CallExprRewriter>
RemoveCubTempStorageFactory::create(const CallExpr *C) const {
  CubDeviceLevelRule::removeRedundantTempVar(C);
  return Inner->create(C);
}

std::function<bool(const CallExpr *)>
checkArgCanMappingToSyclNativeBinaryOp(size_t ArgIdx) {
  return [=](const CallExpr *C) -> bool {
    const Expr *Arg = C->getArg(ArgIdx);
    std::string TypeName =
    DpctGlobalInfo::getUnqualifiedTypeName(Arg->getType().getCanonicalType());
    return CubTypeRule::CanMappingToSyclNativeBinaryOp(TypeName);
  };
}

inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createRemoveCubTempStorageFactory(
std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>> &&Input) {
  return std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
  std::move(Input.first),
  std::make_shared<RemoveCubTempStorageFactory>(Input.second));
}

template <class T>
inline std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>
createRemoveCubTempStorageFactory(
std::pair<std::string, std::shared_ptr<CallExprRewriterFactoryBase>> &&Input,
T) {
  return createRemoveCubTempStorageFactory(std::move(Input));
}

#define REMOVE_CUB_TEMP_STORAGE_FACTORY(INNER)                                 \
  createRemoveCubTempStorageFactory(INNER 0),

void CallExprRewriterFactoryBase::initRewriterMapCUB() {
  RewriterMap->merge(
  std::unordered_map<std::string, std::shared_ptr<CallExprRewriterFactoryBase>>(
  {// cub::DeviceReduce::Sum
   CONDITIONAL_FACTORY_ENTRY(
   CheckCubRedundantFunctionCall(),
   REMOVE_API_FACTORY_ENTRY("cub::DeviceReduce::Sum"),
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Execution,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Algorithm,
   REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
   makeCheckAnd(
   CheckArgCount(6, std::greater_equal<>(), /* IncludeDefaultArg */ false),
   makeCheckNot(CheckArgIsDefaultCudaStream(5))),
   MEMBER_CALL_FACTORY_ENTRY(
   "cub::DeviceReduce::Sum",
   MEMBER_CALL(
   ARG(5), true, "fill", ARG(3),
   CALL("oneapi::dpl::reduce",
        CALL("oneapi::dpl::execution::device_policy", STREAM(5)), ARG(2),
        BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
        ZERO_INITIALIZER(TYPENAME(STATIC_MEMBER_EXPR(
        TEMPLATED_NAME("std::iterator_traits", CALL("decltype", ARG(3))),
        LITERAL("value_type"))))),
   LITERAL("1")),
   false, "wait"),
   MEMBER_CALL_FACTORY_ENTRY(
   "cub::DeviceReduce::Sum",
   MEMBER_CALL(
   QUEUESTR, false, "fill", ARG(3),
   CALL("oneapi::dpl::reduce",
        CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2),
        BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
        ZERO_INITIALIZER(TYPENAME(STATIC_MEMBER_EXPR(
        TEMPLATED_NAME("std::iterator_traits", CALL("decltype", ARG(3))),
        LITERAL("value_type"))))),
   LITERAL("1")),
   false, "wait"))))))

   // cub::DeviceReduce::Reduce
   CONDITIONAL_FACTORY_ENTRY(
   CheckCubRedundantFunctionCall(),
   REMOVE_API_FACTORY_ENTRY("cub::DeviceReduce::Reduce"),
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Execution,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Algorithm,
   REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
   makeCheckAnd(
   CheckArgCount(8, std::greater_equal<>(), /* IncludeDefaultArg */ false),
   makeCheckNot(CheckArgIsDefaultCudaStream(7))),
   MEMBER_CALL_FACTORY_ENTRY(
   "cub::DeviceReduce::Reduce",
   MEMBER_CALL(ARG(7), true, "fill", ARG(3),
               CALL("oneapi::dpl::reduce",
                    CALL("oneapi::dpl::execution::device_policy", STREAM(7)),
                    ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
                    ARG(6), ARG(5)),
               LITERAL("1")),
   false, "wait"),
   MEMBER_CALL_FACTORY_ENTRY(
   "cub::DeviceReduce::Reduce",
   MEMBER_CALL(QUEUESTR, false, "fill", ARG(3),
               CALL("oneapi::dpl::reduce",
                    CALL("oneapi::dpl::execution::device_policy", QUEUESTR),
                    ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)),
                    ARG(6), ARG(5)),
               LITERAL("1")),
   false, "wait"))))))

   // cub::DeviceReduceByKey
   CONDITIONAL_FACTORY_ENTRY(
   CheckCubRedundantFunctionCall(),
   REMOVE_API_FACTORY_ENTRY("cub::DeviceReduce::ReduceByKey"),
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Execution,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Algorithm,
   REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
   makeCheckAnd(
   CheckArgCount(10, std::greater_equal<>(), /* IncludeDefaultArg */ false),
   makeCheckNot(CheckArgIsDefaultCudaStream(9))),
   MEMBER_CALL_FACTORY_ENTRY(
   "cub::DeviceReduce::ReduceByKey",
   MEMBER_CALL(
   ARG(9), true, "fill", ARG(6),
   CALL("std::distance", ARG(3),
        MEMBER_EXPR(
        CALL("oneapi::dpl::reduce_by_key",
             CALL("oneapi::dpl::execution::device_policy", STREAM(9)), ARG(2),
             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(8)), ARG(4), ARG(3),
             ARG(5),
             CALL(TEMPLATED_NAME(
             "std::equal_to",
             TYPENAME(STATIC_MEMBER_EXPR(
             TEMPLATED_NAME("std::iterator_traits", CALL("decltype", ARG(2))),
             LITERAL("value_type"))))),
             ARG(7)),
        false, LITERAL("first"))),
   LITERAL("1")),
   false, "wait"),
   MEMBER_CALL_FACTORY_ENTRY(
   "cub::DeviceReduce::ReduceByKey",
   MEMBER_CALL(
   QUEUESTR, false, "fill", ARG(6),
   CALL("std::distance", ARG(3),
        MEMBER_EXPR(
        CALL("oneapi::dpl::reduce_by_key",
             CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2),
             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(8)), ARG(4), ARG(3),
             ARG(5),
             CALL(TEMPLATED_NAME(
             "std::equal_to",
             TYPENAME(STATIC_MEMBER_EXPR(
             TEMPLATED_NAME("std::iterator_traits", CALL("decltype", ARG(2))),
             LITERAL("value_type"))))),
             ARG(7)),
        false, LITERAL("first"))),
   LITERAL("1")),
   false, "wait"))))))

   // cub::DeviceScan::ExclusiveSum
   CONDITIONAL_FACTORY_ENTRY(
   CheckCubRedundantFunctionCall(),
   REMOVE_API_FACTORY_ENTRY("cub::DeviceScan::ExclusiveSum"),
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Execution,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Algorithm,
   REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
   makeCheckAnd(
   CheckArgCount(6, std::greater_equal<>(), /* IncludeDefaultArg */ false),
   makeCheckNot(CheckArgIsDefaultCudaStream(5))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceScan::ExclusiveSum",
   CALL("oneapi::dpl::exclusive_scan",
        CALL("oneapi::dpl::execution::device_policy", STREAM(5)), ARG(2),
        BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)), ARG(3),
        ZERO_INITIALIZER(TYPENAME(STATIC_MEMBER_EXPR(
        TEMPLATED_NAME("std::iterator_traits", CALL("decltype", ARG(2))),
        LITERAL("value_type")))))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceScan::ExclusiveSum",
   CALL("oneapi::dpl::exclusive_scan",
        CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2),
        BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)), ARG(3),
        ZERO_INITIALIZER(TYPENAME(STATIC_MEMBER_EXPR(
        TEMPLATED_NAME("std::iterator_traits", CALL("decltype", ARG(2))),
        LITERAL("value_type")))))))))))

   // cub::DeviceScan::InclusiveSum
   CONDITIONAL_FACTORY_ENTRY(
   CheckCubRedundantFunctionCall(),
   REMOVE_API_FACTORY_ENTRY("cub::DeviceScan::InclusiveSum"),
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Execution,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Algorithm,
   REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
   makeCheckAnd(
   CheckArgCount(6, std::greater_equal<>(), /* IncludeDefaultArg */ false),
   makeCheckNot(CheckArgIsDefaultCudaStream(5))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceScan::InclusiveSum",
   CALL("oneapi::dpl::inclusive_scan",
        CALL("oneapi::dpl::execution::device_policy", STREAM(5)), ARG(2),
        BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)), ARG(3))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceScan::InclusiveSum",
   CALL("oneapi::dpl::inclusive_scan",
        CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2),
        BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(4)), ARG(3))))))))

   // cub::DeviceScan::ExclusiveScan
   CONDITIONAL_FACTORY_ENTRY(
   CheckCubRedundantFunctionCall(),
   REMOVE_API_FACTORY_ENTRY("cub::DeviceScan::ExclusiveScan"),
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Execution,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Algorithm,
   REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
   makeCheckAnd(
   CheckArgCount(8, std::greater_equal<>(), /* IncludeDefaultArg */ false),
   makeCheckNot(CheckArgIsDefaultCudaStream(7))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceScan::ExclusiveScan",
   CALL("oneapi::dpl::exclusive_scan",
        CALL("oneapi::dpl::execution::device_policy", STREAM(7)), ARG(2),
        BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(6)), ARG(3), ARG(5),
        ARG(4))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceScan::ExclusiveScan",
   CALL("oneapi::dpl::exclusive_scan",
        CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2),
        BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(6)), ARG(3), ARG(5),
        ARG(4))))))))

   // cub::DeviceScan::InclusiveScan
   CONDITIONAL_FACTORY_ENTRY(
   CheckCubRedundantFunctionCall(),
   REMOVE_API_FACTORY_ENTRY("cub::DeviceScan::InclusiveScan"),
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Execution,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Algorithm,
   REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
   makeCheckAnd(
   CheckArgCount(7, std::greater_equal<>(), /* IncludeDefaultArg */ false),
   makeCheckNot(CheckArgIsDefaultCudaStream(6))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceScan::InclusiveScan",
   CALL("oneapi::dpl::inclusive_scan",
        CALL("oneapi::dpl::execution::device_policy", STREAM(6)), ARG(2),
        BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(5)), ARG(3), ARG(4))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceScan::InclusiveScan",
   CALL("oneapi::dpl::inclusive_scan",
        CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2),
        BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(5)), ARG(3), ARG(4))))))))

   // cub::DeviceSelect::Flagged
   CONDITIONAL_FACTORY_ENTRY(
   CheckCubRedundantFunctionCall(),
   REMOVE_API_FACTORY_ENTRY("cub::DeviceSelect::Flagged"),
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Utils,
   REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
   makeCheckAnd(
   CheckArgCount(8, std::greater_equal<>(), /* IncludeDefaultArg */ false),
   makeCheckNot(CheckArgIsDefaultCudaStream(7))),
   MEMBER_CALL_FACTORY_ENTRY(
   "cub::DeviceSelect::Flagged",
   MEMBER_CALL(
   ARG(7), true, "fill", ARG(5),
   CALL("std::distance", ARG(4),
        CALL("dpct::copy_if",
             CALL("oneapi::dpl::execution::device_policy", STREAM(7)), ARG(2),
             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(6)), ARG(3), ARG(4),
             LITERAL("[](const auto &t) -> bool { return t; }"))),
   LITERAL("1")),
   false, "wait"),
   MEMBER_CALL_FACTORY_ENTRY(
   "cub::DeviceSelect::Flagged",
   MEMBER_CALL(
   QUEUESTR, false, "fill", ARG(5),
   CALL("std::distance", ARG(4),
        CALL("dpct::copy_if",
             CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2),
             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(6)), ARG(3), ARG(4),
             LITERAL("[](const auto &t) -> bool { return t; }"))),
   LITERAL("1")),
   false, "wait")))))

   // cub::DeviceSelect::Unique
   CONDITIONAL_FACTORY_ENTRY(
   CheckCubRedundantFunctionCall(),
   REMOVE_API_FACTORY_ENTRY("cub::DeviceSelect::Unique"),
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Execution,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Algorithm,
   REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
   makeCheckAnd(
   CheckArgCount(7, std::greater_equal<>(), /* IncludeDefaultArg */ false),
   makeCheckNot(CheckArgIsDefaultCudaStream(6))),
   MEMBER_CALL_FACTORY_ENTRY(
   "cub::DeviceSelect::Unique",
   MEMBER_CALL(
   ARG(6), true, "fill", ARG(4),
   CALL("std::distance", ARG(3),
        CALL("oneapi::dpl::unique_copy",
             CALL("oneapi::dpl::execution::device_policy", STREAM(6)), ARG(2),
             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(5)), ARG(3))),
   LITERAL("1")),
   false, "wait"),
   MEMBER_CALL_FACTORY_ENTRY(
   "cub::DeviceSelect::Unique",
   MEMBER_CALL(
   QUEUESTR, false, "fill", ARG(4),
   CALL("std::distance", ARG(3),
        CALL("oneapi::dpl::unique_copy",
             CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2),
             BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(5)), ARG(3))),
   LITERAL("1")),
   false, "wait"))))))

   // cub::DeviceRunLengthEncode::Encode
   CONDITIONAL_FACTORY_ENTRY(
   CheckCubRedundantFunctionCall(),
   REMOVE_API_FACTORY_ENTRY("cub::DeviceRunLengthEncode::Encode"),
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Utils,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Execution,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Algorithm,
   REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
   makeCheckAnd(
   CheckArgCount(8, std::greater_equal<>(), /* IncludeDefaultArg */ false),
   makeCheckNot(CheckArgIsDefaultCudaStream(7))),
   MEMBER_CALL_FACTORY_ENTRY(
   "cub::DeviceRunLengthEncode::Encode",
   MEMBER_CALL(
   ARG(7), true, "fill", ARG(5),
   CALL("std::distance", ARG(3),
        MEMBER_EXPR(CALL("oneapi::dpl::reduce_by_segment",
                         CALL("oneapi::dpl::execution::device_"
                              "policy",
                              STREAM(7)),
                         ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(6)),
                         MEMBER_CALL(CALL("dpct::device_vector<size_t>", ARG(6),
                                          LITERAL("1")),
                                     false, "begin"),
                         ARG(3), ARG(4)),
                    false, LITERAL("first"))),
   LITERAL("1")),
   false, "wait"),
   MEMBER_CALL_FACTORY_ENTRY(
   "cub::DeviceRunLengthEncode::Encode",
   MEMBER_CALL(
   QUEUESTR, false, "fill", ARG(5),
   CALL("std::distance", ARG(3),
        MEMBER_EXPR(CALL("oneapi::dpl::reduce_by_segment",
                         CALL("oneapi::dpl::execution::device_"
                              "policy",
                              QUEUESTR),
                         ARG(2), BO(BinaryOperatorKind::BO_Add, ARG(2), ARG(6)),
                         MEMBER_CALL(CALL("dpct::device_vector<size_t>", ARG(6),
                                          LITERAL("1")),
                                     false, "begin"),
                         ARG(3), ARG(4)),
                    false, LITERAL("first"))),
   LITERAL("1")),
   false, "wait")))))))

   // cub::DeviceSegmentedReduce::Reduce
   CONDITIONAL_FACTORY_ENTRY(
   CheckCubRedundantFunctionCall(),
   REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedReduce::Reduce"),
   REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::DplExtrasDpcppExtensions_segmented_reduce,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Utils,
   WARNING_FACTORY_ENTRY(
   "cub::DeviceSegmentedReduce::Reduce",
   CONDITIONAL_FACTORY_ENTRY(
   makeCheckAnd(
   CheckArgCount(10, std::greater_equal<>(), /* IncludeDefaultArg */ false),
   makeCheckNot(CheckArgIsDefaultCudaStream(9))),
   CONDITIONAL_FACTORY_ENTRY(
   checkArgCanMappingToSyclNativeBinaryOp(7),
   CALL_FACTORY_ENTRY(
   "cub::DeviceSegmentedReduce::Reduce",
   CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() +
                                   "device::segmented_reduce",
                                   LITERAL("128")),
        STREAM(9), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8))),
   WARNING_FACTORY_ENTRY(
   "cub::DeviceSegmentedReduce::Reduce",
   CALL_FACTORY_ENTRY(
   "cub::DeviceSegmentedReduce::Reduce",
   CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() +
                                   "device::segmented_reduce",
                                   LITERAL("128")),
        STREAM(9), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
        LITERAL("dpct_placeholder"), ARG(8))),
   Diagnostics::UNSUPPORTED_BINARY_OPERATION)),
   CONDITIONAL_FACTORY_ENTRY(
   checkArgCanMappingToSyclNativeBinaryOp(7),
   CALL_FACTORY_ENTRY(
   "cub::DeviceSegmentedReduce::Reduce",
   CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() +
                                   "device::segmented_reduce",
                                   LITERAL("128")),
        QUEUESTR, ARG(2), ARG(3), ARG(4), ARG(5), ARG(6), ARG(7), ARG(8))),
   WARNING_FACTORY_ENTRY(
   "cub::DeviceSegmentedReduce::Reduce",
   CALL_FACTORY_ENTRY(
   "cub::DeviceSegmentedReduce::Reduce",
   CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() +
                                   "device::segmented_reduce",
                                   LITERAL("128")),
        QUEUESTR, ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
        LITERAL("dpct_placeholder"), ARG(8))),
   Diagnostics::UNSUPPORTED_BINARY_OPERATION))),
   Diagnostics::REDUCE_PERFORMANCE_TUNE)))))

   // cub::DeviceSegmentedReduce::Sum
   CONDITIONAL_FACTORY_ENTRY(
   CheckCubRedundantFunctionCall(),
   REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedReduce::Sum"),
   REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::DplExtrasDpcppExtensions_segmented_reduce,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Utils,
   WARNING_FACTORY_ENTRY(
   "cub::DeviceSegmentedReduce::Sum",
   CONDITIONAL_FACTORY_ENTRY(
   makeCheckAnd(
   CheckArgCount(10, std::greater_equal<>(), /* IncludeDefaultArg */ false),
   makeCheckNot(CheckArgIsDefaultCudaStream(9))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceSegmentedReduce::Sum",
   CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() +
                                   "device::segmented_reduce",
                                   LITERAL("128")),
        STREAM(9), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
        CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getClNamespace() + "plus",
                                        LITERAL(""))),
        ZERO_INITIALIZER(TYPENAME(STATIC_MEMBER_EXPR(
        TEMPLATED_NAME("std::iterator_traits", CALL("decltype", ARG(3))),
        LITERAL("value_type")))))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceSegmentedReduce::Sum",
   CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() +
                                   "device::segmented_reduce",
                                   LITERAL("128")),
        QUEUESTR, ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
        CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getClNamespace() + "plus",
                                        LITERAL(""))),
        ZERO_INITIALIZER(TYPENAME(STATIC_MEMBER_EXPR(
        TEMPLATED_NAME("std::iterator_traits", CALL("decltype", ARG(3))),
        LITERAL("value_type"))))))),
   Diagnostics::REDUCE_PERFORMANCE_TUNE)))))

   // cub::DeviceSegmentedReduce::Min
   CONDITIONAL_FACTORY_ENTRY(
   CheckCubRedundantFunctionCall(),
   REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedReduce::Min"),
   REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::DplExtrasDpcppExtensions_segmented_reduce,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Utils,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_STD_Numeric_Limits,
   WARNING_FACTORY_ENTRY(
   "cub::DeviceSegmentedReduce::Min",
   CONDITIONAL_FACTORY_ENTRY(
   makeCheckAnd(
   CheckArgCount(10, std::greater_equal<>(), /* IncludeDefaultArg */ false),
   makeCheckNot(CheckArgIsDefaultCudaStream(9))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceSegmentedReduce::Min",
   CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() +
                                   "device::segmented_reduce",
                                   LITERAL("128")),
        STREAM(9), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
        CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getClNamespace() + "minimum",
                                        LITERAL(""))),
        CALL(STATIC_MEMBER_EXPR(
        TEMPLATED_NAME(
        "std::numeric_limits",
        TYPENAME(STATIC_MEMBER_EXPR(
        TEMPLATED_NAME("std::iterator_traits", CALL("decltype", ARG(3))),
        LITERAL("value_type")))),
        LITERAL("max"))))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceSegmentedReduce::Min",
   CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() +
                                   "device::segmented_reduce",
                                   LITERAL("128")),
        QUEUESTR, ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
        CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getClNamespace() + "minimum",
                                        LITERAL(""))),
        CALL(STATIC_MEMBER_EXPR(
        TEMPLATED_NAME(
        "std::numeric_limits",
        TYPENAME(STATIC_MEMBER_EXPR(
        TEMPLATED_NAME("std::iterator_traits", CALL("decltype", ARG(3))),
        LITERAL("value_type")))),
        LITERAL("max")))))),
   Diagnostics::REDUCE_PERFORMANCE_TUNE))))))

   // cub::DeviceSegmentedReduce::Max
   CONDITIONAL_FACTORY_ENTRY(
   CheckCubRedundantFunctionCall(),
   REMOVE_API_FACTORY_ENTRY("cub::DeviceSegmentedReduce::Max"),
   REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::DplExtrasDpcppExtensions_segmented_reduce,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Utils,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_STD_Numeric_Limits,
   WARNING_FACTORY_ENTRY(
   "cub::DeviceSegmentedReduce::Max",
   CONDITIONAL_FACTORY_ENTRY(
   makeCheckAnd(
   CheckArgCount(10, std::greater_equal<>(), /* IncludeDefaultArg */ false),
   makeCheckNot(CheckArgIsDefaultCudaStream(9))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceSegmentedReduce::Max",
   CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() +
                                   "device::segmented_reduce",
                                   LITERAL("128")),
        STREAM(9), ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
        CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getClNamespace() + "maximum",
                                        LITERAL(""))),
        CALL(STATIC_MEMBER_EXPR(
        TEMPLATED_NAME(
        "std::numeric_limits",
        TYPENAME(STATIC_MEMBER_EXPR(
        TEMPLATED_NAME("std::iterator_traits", CALL("decltype", ARG(3))),
        LITERAL("value_type")))),
        LITERAL("lowest"))))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceSegmentedReduce::Max",
   CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getDpctNamespace() +
                                   "device::segmented_reduce",
                                   LITERAL("128")),
        QUEUESTR, ARG(2), ARG(3), ARG(4), ARG(5), ARG(6),
        CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getClNamespace() + "maximum",
                                        LITERAL(""))),
        CALL(STATIC_MEMBER_EXPR(
        TEMPLATED_NAME(
        "std::numeric_limits",
        TYPENAME(STATIC_MEMBER_EXPR(
        TEMPLATED_NAME("std::iterator_traits", CALL("decltype", ARG(3))),
        LITERAL("value_type")))),
        LITERAL("lowest")))))),
   Diagnostics::REDUCE_PERFORMANCE_TUNE))))))

   CONDITIONAL_FACTORY_ENTRY(
   CheckCubRedundantFunctionCall(),
   REMOVE_API_FACTORY_ENTRY("cub::DeviceRadixSort::SortKeys"),
   REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::DplExtrasAlgorithm_sort_keys,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Utils,
   REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
   makeCheckAnd(CheckArgCount(8, std::greater_equal<>(),
                              /* IncludeDefaultArg */ false),
                makeCheckNot(CheckArgIsDefaultCudaStream(7))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceRadixSort::SortKeys",
   CALL("dpct::sort_keys",
        CALL("oneapi::dpl::execution::device_policy", STREAM(7)), ARG(2),
        ARG(3), ARG(4), LITERAL("false"), ARG(5), ARG(6))),
   CONDITIONAL_FACTORY_ENTRY(
   CheckArgCount(7, std::greater_equal<>(),
                 /* IncludeDefaultArg */ false),
   CALL_FACTORY_ENTRY(
   "cub::DeviceRadixSort::SortKeys",
   CALL("dpct::sort_keys",
        CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2), ARG(3),
        ARG(4), LITERAL("false"), ARG(5), ARG(6))),
   CONDITIONAL_FACTORY_ENTRY(
   CheckArgCount(6, std::greater_equal<>(),
                 /* IncludeDefaultArg */ false),
   CALL_FACTORY_ENTRY(
   "cub::DeviceRadixSort::SortKeys",
   CALL("dpct::sort_keys",
        CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2), ARG(3),
        ARG(4), LITERAL("false"), ARG(5))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceRadixSort::SortKeys",
   CALL("dpct::sort_keys",
        CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2), ARG(3),
        ARG(4), LITERAL("false")))))))))))

   CONDITIONAL_FACTORY_ENTRY(
   CheckCubRedundantFunctionCall(),
   REMOVE_API_FACTORY_ENTRY("cub::DeviceRadixSort::SortKeysDescending"),
   REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::DplExtrasAlgorithm_sort_keys,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Utils,
   REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
   makeCheckAnd(CheckArgCount(8, std::greater_equal<>(),
                              /* IncludeDefaultArg */ false),
                makeCheckNot(CheckArgIsDefaultCudaStream(7))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceRadixSort::SortKeysDescending",
   CALL("dpct::sort_keys",
        CALL("oneapi::dpl::execution::device_policy", STREAM(7)), ARG(2),
        ARG(3), ARG(4), LITERAL("true"), ARG(5), ARG(6))),
   CONDITIONAL_FACTORY_ENTRY(
   CheckArgCount(7, std::greater_equal<>(),
                 /* IncludeDefaultArg */ false),
   CALL_FACTORY_ENTRY(
   "cub::DeviceRadixSort::SortKeysDescending",
   CALL("dpct::sort_keys",
        CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2), ARG(3),
        ARG(4), LITERAL("true"), ARG(5), ARG(6))),
   CONDITIONAL_FACTORY_ENTRY(
   CheckArgCount(6, std::greater_equal<>(),
                 /* IncludeDefaultArg */ false),
   CALL_FACTORY_ENTRY(
   "cub::DeviceRadixSort::SortKeysDescending",
   CALL("dpct::sort_keys",
        CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2), ARG(3),
        ARG(4), LITERAL("true"), ARG(5))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceRadixSort::SortKeysDescending",
   CALL("dpct::sort_keys",
        CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2), ARG(3),
        ARG(4), LITERAL("true")))))))))))

   CONDITIONAL_FACTORY_ENTRY(
   CheckCubRedundantFunctionCall(),
   REMOVE_API_FACTORY_ENTRY("cub::DeviceRadixSort::SortPairs"),
   REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::DplExtrasAlgorithm_sort_pairs,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Utils,
   REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
   makeCheckAnd(CheckArgCount(10, std::greater_equal<>(),
                              /* IncludeDefaultArg */ false),
                makeCheckNot(CheckArgIsDefaultCudaStream(9))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceRadixSort::SortPairs",
   CALL("dpct::sort_pairs",
        CALL("oneapi::dpl::execution::device_policy", STREAM(9)), ARG(2),
        ARG(3), ARG(4), ARG(5), ARG(6), LITERAL("false"), ARG(7), ARG(8))),
   CONDITIONAL_FACTORY_ENTRY(
   CheckArgCount(9, std::greater_equal<>(),
                 /* IncludeDefaultArg */ false),
   CALL_FACTORY_ENTRY(
   "cub::DeviceRadixSort::SortPairs",
   CALL("dpct::sort_pairs",
        CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2), ARG(3),
        ARG(4), ARG(5), ARG(6), LITERAL("false"), ARG(7), ARG(8))),
   CONDITIONAL_FACTORY_ENTRY(
   CheckArgCount(8, std::greater_equal<>(),
                 /* IncludeDefaultArg */ false),
   CALL_FACTORY_ENTRY(
   "cub::DeviceRadixSort::SortPairs",
   CALL("dpct::sort_pairs",
        CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2), ARG(3),
        ARG(4), ARG(5), ARG(6), LITERAL("false"), ARG(7))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceRadixSort::SortPairs",
   CALL("dpct::sort_pairs",
        CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2), ARG(3),
        ARG(4), ARG(5), ARG(6), LITERAL("false")))))))))))

   CONDITIONAL_FACTORY_ENTRY(
   CheckCubRedundantFunctionCall(),
   REMOVE_API_FACTORY_ENTRY("cub::DeviceRadixSort::SortPairsDescending"),
   REMOVE_CUB_TEMP_STORAGE_FACTORY(FEATURE_REQUEST_FACTORY(
   HelperFeatureEnum::DplExtrasAlgorithm_sort_pairs,
   HEADER_INSERT_FACTORY(
   HeaderType::HT_DPL_Utils,
   REMOVE_CUB_TEMP_STORAGE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
   makeCheckAnd(CheckArgCount(10, std::greater_equal<>(),
                              /* IncludeDefaultArg */ false),
                makeCheckNot(CheckArgIsDefaultCudaStream(9))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceRadixSort::SortPairsDescending",
   CALL("dpct::sort_pairs",
        CALL("oneapi::dpl::execution::device_policy", STREAM(9)), ARG(2),
        ARG(3), ARG(4), ARG(5), ARG(6), LITERAL("true"), ARG(7), ARG(8))),
   CONDITIONAL_FACTORY_ENTRY(
   CheckArgCount(9, std::greater_equal<>(),
                 /* IncludeDefaultArg */ false),
   CALL_FACTORY_ENTRY(
   "cub::DeviceRadixSort::SortPairsDescending",
   CALL("dpct::sort_pairs",
        CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2), ARG(3),
        ARG(4), ARG(5), ARG(6), LITERAL("true"), ARG(7), ARG(8))),
   CONDITIONAL_FACTORY_ENTRY(
   CheckArgCount(8, std::greater_equal<>(),
                 /* IncludeDefaultArg */ false),
   CALL_FACTORY_ENTRY(
   "cub::DeviceRadixSort::SortPairsDescending",
   CALL("dpct::sort_pairs",
        CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2), ARG(3),
        ARG(4), ARG(5), ARG(6), LITERAL("true"), ARG(7))),
   CALL_FACTORY_ENTRY(
   "cub::DeviceRadixSort::SortPairsDescending",
   CALL("dpct::sort_pairs",
        CALL("oneapi::dpl::execution::device_policy", QUEUESTR), ARG(2), ARG(3),
        ARG(4), ARG(5), ARG(6), LITERAL("true")))))))))))

  }));
}

} // namespace dpct
} // namespace clang
