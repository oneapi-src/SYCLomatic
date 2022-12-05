//===--------------- CallExprRewriterMemory.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CallExprRewriter.h"
#include "CallExprRewriterCommon.h"

namespace clang {
namespace dpct {
// clang-format off
void CallExprRewriterFactoryBase::initRewriterMapMemory() {
  RewriterMap->merge(
      std::unordered_map<std::string,
                         std::shared_ptr<CallExprRewriterFactoryBase>>({
ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        CheckCanUseTemplateMalloc(0, 1),
        ASSIGN_FACTORY_ENTRY("cudaMalloc", DEREF(makeDerefArgCreatorWithCall(0)),
                            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                MapNames::getClNamespace() + "malloc_device",
                                getDoubleDerefedType(0)),
                                getSizeForMalloc(0, 1), QUEUESTR)),
        ASSIGN_FACTORY_ENTRY("cudaMalloc", DEREF(makeDerefArgCreatorWithCall(0)),
                            CAST(getDerefedType(0),
                                CALL(MapNames::getClNamespace() + "malloc_device",
                                getSizeForMalloc(0, 1), QUEUESTR)
                             ))),
    CONDITIONAL_FACTORY_ENTRY(
        makeCheckOr(CheckDerefedTypeBeforeCast(0, "NULL TYPE"), CheckDerefedTypeBeforeCast(0, "void *")),
        ASSIGN_FACTORY_ENTRY("cudaMalloc", DEREF(makeDerefArgCreatorWithCall(0)),
                             CALL(MapNames::getDpctNamespace() + "dpct_malloc",
                                  makeCallArgCreatorWithCall(1))),
        ASSIGN_FACTORY_ENTRY("cudaMalloc", DEREF(makeDerefArgCreatorWithCall(0)),
                             CAST(getDerefedType(0),
                                  CALL(MapNames::getDpctNamespace() + "dpct_malloc",
                                       makeCallArgCreatorWithCall(1)))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        CheckCanUseTemplateMalloc(0, 1),
        ASSIGN_FACTORY_ENTRY("cuMemAlloc_v2", DEREF(makeDerefArgCreatorWithCall(0)),
                            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                MapNames::getClNamespace() + "malloc_device",
                                getDoubleDerefedType(0)),
                                getSizeForMalloc(0, 1), QUEUESTR)),
        ASSIGN_FACTORY_ENTRY("cuMemAlloc_v2", DEREF(makeDerefArgCreatorWithCall(0)),
                            CAST(getDerefedType(0),
                                CALL(MapNames::getClNamespace() + "malloc_device",
                                getSizeForMalloc(0, 1), QUEUESTR)
                             ))),
    CONDITIONAL_FACTORY_ENTRY(
        makeCheckOr(CheckDerefedTypeBeforeCast(0, "NULL TYPE"),
                    CheckDerefedTypeBeforeCast(0, "void *")),
        ASSIGN_FACTORY_ENTRY("cuMemAlloc_v2", DEREF(makeDerefArgCreatorWithCall(0)),
                             CALL(MapNames::getDpctNamespace() + "dpct_malloc",
                                  makeCallArgCreatorWithCall(1))),
        ASSIGN_FACTORY_ENTRY("cuMemAlloc_v2", DEREF(makeDerefArgCreatorWithCall(0)),
                             CAST(getDerefedType(0),
                                  CALL(MapNames::getDpctNamespace() + "dpct_malloc",
                                       makeCallArgCreatorWithCall(1)))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        CheckCanUseTemplateMalloc(0, 1),
        ASSIGN_FACTORY_ENTRY(
            "cudaHostAlloc", DEREF(makeDerefArgCreatorWithCall(0)),
            CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getClNamespace() +
                                                "malloc_host",
                                            getDoubleDerefedType(0)),
                 getSizeForMalloc(0, 1), QUEUESTR)),
        ASSIGN_FACTORY_ENTRY(
            "cudaHostAlloc", DEREF(makeDerefArgCreatorWithCall(0)),
            CAST(getDerefedType(0),
                 CALL(MapNames::getClNamespace() + "malloc_host",
                      getSizeForMalloc(0, 1), QUEUESTR)))),
    ASSIGN_FACTORY_ENTRY("cudaHostAlloc", DEREF(makeDerefArgCreatorWithCall(0)),
                         CAST(getDerefedType(0), CALL("malloc", ARG(1))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        CheckCanUseTemplateMalloc(0, 1),
        ASSIGN_FACTORY_ENTRY(
            "cudaMallocHost", DEREF(makeDerefArgCreatorWithCall(0)),
            CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getClNamespace() +
                                                "malloc_host",
                                            getDoubleDerefedType(0)),
                 getSizeForMalloc(0, 1), QUEUESTR)),
        ASSIGN_FACTORY_ENTRY(
            "cudaMallocHost", DEREF(makeDerefArgCreatorWithCall(0)),
            CAST(getDerefedType(0),
                 CALL(MapNames::getClNamespace() + "malloc_host",
                      getSizeForMalloc(0, 1), QUEUESTR)))),
    ASSIGN_FACTORY_ENTRY("cudaMallocHost", DEREF(makeDerefArgCreatorWithCall(0)),
                         CAST(getDerefedType(0), CALL("malloc", ARG(1))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        CheckCanUseTemplateMalloc(0, 1),
        ASSIGN_FACTORY_ENTRY(
            "cuMemHostAlloc", DEREF(makeDerefArgCreatorWithCall(0)),
            CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getClNamespace() +
                                                "malloc_host",
                                            getDoubleDerefedType(0)),
                 getSizeForMalloc(0, 1), QUEUESTR)),
        ASSIGN_FACTORY_ENTRY(
            "cuMemHostAlloc", DEREF(makeDerefArgCreatorWithCall(0)),
            CAST(getDerefedType(0),
                 CALL(MapNames::getClNamespace() + "malloc_host",
                      getSizeForMalloc(0, 1), QUEUESTR)))),
    ASSIGN_FACTORY_ENTRY("cuMemHostAlloc", DEREF(makeDerefArgCreatorWithCall(0)),
                         CAST(getDerefedType(0), CALL("malloc", ARG(1))))))


ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        CheckCanUseTemplateMalloc(0, 1),
        ASSIGN_FACTORY_ENTRY(
            "cuMemAllocHost_v2", DEREF(makeDerefArgCreatorWithCall(0)),
            CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getClNamespace() +
                                                "malloc_host",
                                            getDoubleDerefedType(0)),
                 getSizeForMalloc(0, 1), QUEUESTR)),
        ASSIGN_FACTORY_ENTRY(
            "cuMemAllocHost_v2", DEREF(makeDerefArgCreatorWithCall(0)),
            CAST(getDerefedType(0),
                 CALL(MapNames::getClNamespace() + "malloc_host",
                      getSizeForMalloc(0, 1), QUEUESTR)))),
    ASSIGN_FACTORY_ENTRY("cuMemAllocHost_v2", DEREF(makeDerefArgCreatorWithCall(0)),
                         CAST(getDerefedType(0), CALL("malloc", ARG(1))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    CheckCanUseTemplateMalloc(0, 1),
    CONDITIONAL_FACTORY_ENTRY(
        hasManagedAttr(0),
        ASSIGN_FACTORY_ENTRY(
            "cudaMallocManaged",
            makeCombinedArg(
                makeCombinedArg(ARG("*("),
                                DEREF(makeDerefArgCreatorWithCall(0))),
                ARG(".get_ptr())")),
            CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getClNamespace() +
                                                "malloc_shared",
                                            getDoubleDerefedType(0)),
                 getSizeForMalloc(0, 1), QUEUESTR)),
        ASSIGN_FACTORY_ENTRY(
            "cudaMallocManaged", DEREF(makeDerefArgCreatorWithCall(0)),
            CALL(TEMPLATED_CALLEE_WITH_ARGS(MapNames::getClNamespace() +
                                                "malloc_shared",
                                            getDoubleDerefedType(0)),
                 getSizeForMalloc(0, 1), QUEUESTR))),
    CONDITIONAL_FACTORY_ENTRY(
        hasManagedAttr(0),
        ASSIGN_FACTORY_ENTRY(
            "cudaMallocManaged",
            makeCombinedArg(
                makeCombinedArg(ARG("*("),
                                DEREF(makeDerefArgCreatorWithCall(0))),
                ARG(".get_ptr())")),
            CAST(getDerefedType(0),
                 CALL(MapNames::getClNamespace() + "malloc_shared",
                      getSizeForMalloc(0, 1), QUEUESTR))),
        ASSIGN_FACTORY_ENTRY(
            "cudaMallocManaged", DEREF(makeDerefArgCreatorWithCall(0)),
            CAST(getDerefedType(0),
                 CALL(MapNames::getClNamespace() + "malloc_shared",
                      getSizeForMalloc(0, 1), QUEUESTR))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    CheckCanUseTemplateMalloc(0, 1),
    ASSIGN_FACTORY_ENTRY("cuMemAllocManaged", DEREF(makeDerefArgCreatorWithCall(0)),
                         CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                  MapNames::getClNamespace() + "malloc_shared",
                                  getDoubleDerefedType(0)),
                              getSizeForMalloc(0, 1), QUEUESTR)),
    ASSIGN_FACTORY_ENTRY("cuMemAllocManaged", DEREF(makeDerefArgCreatorWithCall(0)),
                         CAST(getDerefedType(0),
                              CALL(MapNames::getClNamespace() + "malloc_shared",
                                   getSizeForMalloc(0, 1), QUEUESTR)
                             ))))

// use makeCallArgCreatorWithCall instead of makeDerefArgCreatorWithCall to keep the cast information
ASSIGNABLE_FACTORY(
    ASSIGN_FACTORY_ENTRY("cudaHostGetDevicePointer", DEREF(makeCallArgCreatorWithCall(0)),
                         CAST(ARG("char *"), ARG(1))))

ASSIGNABLE_FACTORY(
    ASSIGN_FACTORY_ENTRY("cuMemHostGetDevicePointer_v2", DEREF(makeCallArgCreatorWithCall(0)),
                         CAST(ARG("char *"), ARG(1))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        checkIsArgStream(3),
        MEMBER_CALL_FACTORY_ENTRY(
            "cuMemcpyDtoDAsync_v2", ARG(3), true, "memcpy", ARG(0), ARG(1), ARG(2)),
        MEMBER_CALL_FACTORY_ENTRY(
            "cuMemcpyDtoDAsync_v2", QUEUESTR, false, "memcpy", ARG(0), ARG(1), ARG(2))),
    CONDITIONAL_FACTORY_ENTRY(
        checkIsArgStream(3),
        CALL_FACTORY_ENTRY(
            "cuMemcpyDtoDAsync_v2", CALL(MapNames::getDpctNamespace() + "async_dpct_memcpy",
            ARG(0), ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "automatic"), STREAM(3))),
        CALL_FACTORY_ENTRY(
            "cuMemcpyDtoDAsync_v2", CALL(MapNames::getDpctNamespace() + "async_dpct_memcpy",
            ARG(0), ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "automatic"))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    MEMBER_CALL_FACTORY_ENTRY(
        "cuMemcpyDtoD_v2", MEMBER_CALL(QUEUESTR, false, "memcpy", ARG(0), ARG(1), ARG(2)), false, "wait"),
    CALL_FACTORY_ENTRY(
        "cuMemcpyDtoD_v2", CALL(MapNames::getDpctNamespace() + "dpct_memcpy",
        ARG(0), ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "automatic")))))

ASSIGNABLE_FACTORY(
    ASSIGN_FACTORY_ENTRY(
        "cuMemAllocPitch_v2", DEREF(makeDerefArgCreatorWithCall(0)),
        CAST(getDerefedType(0), CALL(MapNames::getDpctNamespace() + "dpct_malloc", DEREF(makeCallArgCreatorWithCall(1)), ARG(2), ARG(3)))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsArgStream(3),
    MEMBER_CALL_FACTORY_ENTRY(
        "cuMemPrefetchAsync", ARG(3), true, "prefetch", ARG(0), ARG(1)),
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Device_dev_mgr_get_device,
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Device_device_ext_default_queue,
            MEMBER_CALL_FACTORY_ENTRY(
                "cuMemPrefetchAsync", MEMBER_CALL(MEMBER_CALL(CALL(MapNames::getDpctNamespace() + "dev_mgr::instance"),
                false, "get_device", ARG(2)), false, "default_queue"), false, "prefetch", ARG(0), ARG(1))))))


ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkArgSpelling(3, "CU_DEVICE_CPU"),
    CONDITIONAL_FACTORY_ENTRY(
        checkIsArgIntegerLiteral(2),
            WARNING_FACTORY_ENTRY("cuMemAdvise",
                FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Device_cpu_device,
                    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Device_device_ext_default_queue,
                        MEMBER_CALL_FACTORY_ENTRY(
                        "cuMemAdvise",
                        MEMBER_CALL(CALL(MapNames::getDpctNamespace() + "cpu_device"), false, "default_queue"),
                    false, "mem_advise", ARG(0), ARG(1), ARG("0")))),
                Diagnostics::DEFAULT_MEM_ADVICE,
                ARG(" and was set to 0")),

            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Device_cpu_device,
                FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Device_device_ext_default_queue,
                    MEMBER_CALL_FACTORY_ENTRY(
                        "cuMemAdvise",
                        MEMBER_CALL(CALL(MapNames::getDpctNamespace() + "cpu_device"), false, "default_queue"),
                        false, "mem_advise", ARG(0), ARG(1), ARG(2))))),

    CONDITIONAL_FACTORY_ENTRY(
        checkIsArgIntegerLiteral(2),
            WARNING_FACTORY_ENTRY("cuMemAdvise",
                FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Device_dev_mgr_get_device,
                    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Device_device_ext_default_queue,
                        MEMBER_CALL_FACTORY_ENTRY(
                            "cuMemAdvise",
                            MEMBER_CALL(MEMBER_CALL(CALL(MapNames::getDpctNamespace() + "dev_mgr::instance"),
                            false, "get_device", ARG(3)), false, "default_queue"),
                            false, "mem_advise", ARG(0), ARG(1), ARG("0")))),
                Diagnostics::DEFAULT_MEM_ADVICE,
                ARG(" and was set to 0")),

            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Device_dev_mgr_get_device,
                FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Device_device_ext_default_queue,
                    MEMBER_CALL_FACTORY_ENTRY(
                        "cuMemAdvise",
                        MEMBER_CALL(MEMBER_CALL(CALL(MapNames::getDpctNamespace() + "dev_mgr::instance"),
                        false, "get_device", ARG(3)), false, "default_queue"),
                        false, "mem_advise", ARG(0), ARG(1), ARG(2)))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CALL_FACTORY_ENTRY("cuMemFree_v2", CALL(MapNames::getClNamespace() + "free", ARG(0), QUEUESTR)),
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Memory_dpct_free,
        CALL_FACTORY_ENTRY("cuMemFree_v2", CALL(MapNames::getDpctNamespace() + "dpct_free", ARG(0))))))

ASSIGNABLE_FACTORY(
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Device_dev_mgr_get_device,
        FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Device_device_ext_get_device_info_return_info,
            FEATURE_REQUEST_FACTORY(HelperFeatureEnum::Device_device_info_get_global_mem_size,
                ASSIGN_FACTORY_ENTRY("cuDeviceTotalMem_v2", DEREF(makeDerefArgCreatorWithCall(0)),
                    MEMBER_CALL(MEMBER_CALL(MEMBER_CALL(CALL(MapNames::getDpctNamespace() + "dev_mgr::instance"),
                        false, "get_device", ARG(1)),
                        false, "get_device_info"), false, "get_global_mem_size"))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        checkIsArgStream(3),
        MEMBER_CALL_FACTORY_ENTRY(
            "cuMemcpyHtoDAsync_v2", ARG(3), true, "memcpy", ARG(0), ARG(1), ARG(2)),
        MEMBER_CALL_FACTORY_ENTRY(
            "cuMemcpyHtoDAsync_v2", QUEUESTR, false, "memcpy", ARG(0), ARG(1), ARG(2))),
    CONDITIONAL_FACTORY_ENTRY(
        checkIsArgStream(3),
        CALL_FACTORY_ENTRY(
            "cuMemcpyHtoDAsync_v2", CALL(MapNames::getDpctNamespace() + "async_dpct_memcpy",
            ARG(0), ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "automatic"), STREAM(3))),
        CALL_FACTORY_ENTRY(
            "cuMemcpyHtoDAsync_v2", CALL(MapNames::getDpctNamespace() + "async_dpct_memcpy",
            ARG(0), ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "automatic"))))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    MEMBER_CALL_FACTORY_ENTRY(
        "cuMemcpyHtoD_v2", MEMBER_CALL(QUEUESTR, false, "memcpy", ARG(0), ARG(1), ARG(2)), false, "wait"),
    CALL_FACTORY_ENTRY(
        "cuMemcpyHtoD_v2", CALL(MapNames::getDpctNamespace() + "dpct_memcpy",
        ARG(0), ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "automatic")))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    MEMBER_CALL_FACTORY_ENTRY(
        "cuMemcpy", MEMBER_CALL(QUEUESTR, false, "memcpy", ARG(0), ARG(1), ARG(2)), false, "wait"),
    CALL_FACTORY_ENTRY(
        "cuMemcpy", CALL(MapNames::getDpctNamespace() + "dpct_memcpy",
	ARG(0), ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "automatic")))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    checkIsUSM(),
    CONDITIONAL_FACTORY_ENTRY(
        checkIsArgStream(3),
        MEMBER_CALL_FACTORY_ENTRY(
            "cuMemcpyAsync", ARG(3), true, "memcpy", ARG(0), ARG(1), ARG(2)),
        MEMBER_CALL_FACTORY_ENTRY(
            "cuMemcpyAsync", QUEUESTR, false, "memcpy", ARG(0), ARG(1), ARG(2))),
    CONDITIONAL_FACTORY_ENTRY(
        checkIsArgStream(3),
        CALL_FACTORY_ENTRY(
            "cuMemcpyAsync", CALL(MapNames::getDpctNamespace() + "async_dpct_memcpy",
            ARG(0), ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "automatic"), STREAM(3))),
        CALL_FACTORY_ENTRY(
            "cuMemcpyAsync", CALL(MapNames::getDpctNamespace() + "async_dpct_memcpy",
            ARG(0), ARG(1), ARG(2), ARG(MapNames::getDpctNamespace() + "automatic"))))))

CONDITIONAL_FACTORY_ENTRY(
    checkIsCallExprOnly(),
    WARNING_FACTORY_ENTRY("cudaHostUnregister",
        TOSTRING_FACTORY_ENTRY("cudaHostUnregister", LITERAL("")),
         Diagnostics::FUNC_CALL_REMOVED,
        std::string("cudaHostUnregister"), getRemovedAPIWarningMessage("cudaHostUnregister")),
    WARNING_FACTORY_ENTRY("cudaHostUnregister",
        TOSTRING_FACTORY_ENTRY("cudaHostUnregister", LITERAL("0")),
        Diagnostics::FUNC_CALL_REMOVED_0,
        std::string("cudaHostUnregister"), getRemovedAPIWarningMessage("cudaHostUnregister")))

CONDITIONAL_FACTORY_ENTRY(
    checkIsCallExprOnly(),
    WARNING_FACTORY_ENTRY("cudaHostRegister",
        TOSTRING_FACTORY_ENTRY("cudaHostRegister", LITERAL("")),
         Diagnostics::FUNC_CALL_REMOVED,
        std::string("cudaHostRegister"), getRemovedAPIWarningMessage("cudaHostRegister")),
    WARNING_FACTORY_ENTRY("cudaHostRegister",
        TOSTRING_FACTORY_ENTRY("cudaHostRegister", LITERAL("0")),
        Diagnostics::FUNC_CALL_REMOVED_0,
        std::string("cudaHostRegister"), getRemovedAPIWarningMessage("cudaHostRegister")))

ASSIGNABLE_FACTORY(
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Memory_pointer_attributes,
        MEMBER_CALL_FACTORY_ENTRY(
            "cudaPointerGetAttributes", DEREF(0), false, "init", ARG(1))))

      }));
  // clang-format on
}

} // namespace dpct
} // namespace clang
