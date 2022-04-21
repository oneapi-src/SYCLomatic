//===---APINamesMemory.inc--------------------------------------------===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

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
    ASSIGN_FACTORY_ENTRY("cudaMallocManaged", DEREF(makeDerefArgCreatorWithCall(0)),
                         CALL(TEMPLATED_CALLEE_WITH_ARGS(
                                  MapNames::getClNamespace() + "malloc_shared",
                                  getDoubleDerefedType(0)),
                              getSizeForMalloc(0, 1), QUEUESTR)),
    ASSIGN_FACTORY_ENTRY("cudaMallocManaged", DEREF(makeDerefArgCreatorWithCall(0)),
                         CAST(getDerefedType(0),
                              CALL(MapNames::getClNamespace() + "malloc_shared",
                                   getSizeForMalloc(0, 1), QUEUESTR)
                             ))))

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
                         ARG(1)))

ASSIGNABLE_FACTORY(
    ASSIGN_FACTORY_ENTRY("cuMemHostGetDevicePointer_v2", DEREF(makeCallArgCreatorWithCall(0)),
                         ARG(1)))

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
