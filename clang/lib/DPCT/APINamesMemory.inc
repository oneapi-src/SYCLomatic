//===---APINamesMemory.inc--------------------------------------------===//
//
// Copyright (C) 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//


ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
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
                             ))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
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
                             ))))

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