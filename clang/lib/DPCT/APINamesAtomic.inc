//===--------------- APINamesAtomic.inc -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
    CONDITIONAL_FACTORY_ENTRY(
        GetUsingGenericSpace,
        CALL_FACTORY_ENTRY(
            "atomicAdd",
            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                     MapNames::getDpctNamespace() + "atomic_fetch_add",
                     LITERAL(MapNames::getClNamespace() +
                             "access::address_space::generic_space")),
                 makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1))),
        CONDITIONAL_FACTORY_ENTRY(
            GetHasSharedAttr(0),
            CALL_FACTORY_ENTRY(
                "atomicAdd",
                CALL(TEMPLATED_CALLEE_WITH_ARGS(
                         MapNames::getDpctNamespace() + "atomic_fetch_add",
                         LITERAL(MapNames::getClNamespace() +
                                 "access::address_space::local_space")),
                     makeCallArgCreatorWithCall(0),
                     makeCallArgCreatorWithCall(1))),
            CONDITIONAL_FACTORY_ENTRY(
                ReportMemoryAttrDeduce(0),
                WARNING_FACTORY_ENTRY(
                    "atomicAdd",
                    CALL_FACTORY_ENTRY("atomicAdd",
                                       CALL(MapNames::getDpctNamespace() +
                                                "atomic_fetch_add",
                                            makeCallArgCreatorWithCall(0),
                                            makeCallArgCreatorWithCall(1))),
                    Diagnostics::SHARE_MEMORY_ATTR_DEDUCE,
                    makeCallArgCreatorWithCall(0),
                    std::string("atomic_fetch_add"),
                    std::string("atomic_fetch_add<") +
                        MapNames::getClNamespace() +
                        std::string("access::address_space::local_space>")),
                CALL_FACTORY_ENTRY("atomicAdd",
                                   CALL(MapNames::getDpctNamespace() +
                                            "atomic_fetch_add",
                                        makeCallArgCreatorWithCall(0),
                                        makeCallArgCreatorWithCall(1)))))))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
    CONDITIONAL_FACTORY_ENTRY(
        GetUsingGenericSpace,
        CALL_FACTORY_ENTRY(
            "atomicSub",
            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                     MapNames::getDpctNamespace() + "atomic_fetch_sub",
                     LITERAL(MapNames::getClNamespace() +
                             "access::address_space::generic_space")),
                 makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1))),
        CONDITIONAL_FACTORY_ENTRY(
            GetHasSharedAttr(0),
            CALL_FACTORY_ENTRY(
                "atomicSub",
                CALL(TEMPLATED_CALLEE_WITH_ARGS(
                         MapNames::getDpctNamespace() + "atomic_fetch_sub",
                         LITERAL(MapNames::getClNamespace() +
                                 "access::address_space::local_space")),
                     makeCallArgCreatorWithCall(0),
                     makeCallArgCreatorWithCall(1))),
            CONDITIONAL_FACTORY_ENTRY(
                ReportMemoryAttrDeduce(0),
                WARNING_FACTORY_ENTRY(
                    "atomicSub",
                    CALL_FACTORY_ENTRY("atomicSub",
                                       CALL(MapNames::getDpctNamespace() +
                                                "atomic_fetch_sub",
                                            makeCallArgCreatorWithCall(0),
                                            makeCallArgCreatorWithCall(1))),
                    Diagnostics::SHARE_MEMORY_ATTR_DEDUCE,
                    makeCallArgCreatorWithCall(0),
                    std::string("atomic_fetch_sub"),
                    std::string("atomic_fetch_sub<") +
                        MapNames::getClNamespace() +
                        std::string("access::address_space::local_space>")),
                CALL_FACTORY_ENTRY("atomicSub",
                                   CALL(MapNames::getDpctNamespace() +
                                            "atomic_fetch_sub",
                                        makeCallArgCreatorWithCall(0),
                                        makeCallArgCreatorWithCall(1)))))))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
    CONDITIONAL_FACTORY_ENTRY(
        GetUsingGenericSpace,
        CALL_FACTORY_ENTRY(
            "atomicExch",
            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                     MapNames::getDpctNamespace() + "atomic_exchange",
                     LITERAL(MapNames::getClNamespace() +
                             "access::address_space::generic_space")),
                 makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1))),
        CONDITIONAL_FACTORY_ENTRY(
            GetHasSharedAttr(0),
            CALL_FACTORY_ENTRY(
                "atomicExch",
                CALL(TEMPLATED_CALLEE_WITH_ARGS(
                         MapNames::getDpctNamespace() + "atomic_exchange",
                         LITERAL(MapNames::getClNamespace() +
                                 "access::address_space::local_space")),
                     makeCallArgCreatorWithCall(0),
                     makeCallArgCreatorWithCall(1))),
            CONDITIONAL_FACTORY_ENTRY(
                ReportMemoryAttrDeduce(0),
                WARNING_FACTORY_ENTRY(
                    "atomicExch",
                    CALL_FACTORY_ENTRY("atomicExch",
                                       CALL(MapNames::getDpctNamespace() +
                                                "atomic_exchange",
                                            makeCallArgCreatorWithCall(0),
                                            makeCallArgCreatorWithCall(1))),
                    Diagnostics::SHARE_MEMORY_ATTR_DEDUCE,
                    makeCallArgCreatorWithCall(0),
                    std::string("atomic_exchange"),
                    std::string("atomic_exchange<") +
                        MapNames::getClNamespace() +
                        std::string("access::address_space::local_space>")),
                CALL_FACTORY_ENTRY("atomicExch",
                                   CALL(MapNames::getDpctNamespace() +
                                            "atomic_exchange",
                                        makeCallArgCreatorWithCall(0),
                                        makeCallArgCreatorWithCall(1)))))))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
    CONDITIONAL_FACTORY_ENTRY(
        GetUsingGenericSpace,
        CALL_FACTORY_ENTRY(
            "atomicMax",
            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                     MapNames::getDpctNamespace() + "atomic_fetch_max",
                     LITERAL(MapNames::getClNamespace() +
                             "access::address_space::generic_space")),
                 makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1))),
        CONDITIONAL_FACTORY_ENTRY(
            GetHasSharedAttr(0),
            CALL_FACTORY_ENTRY(
                "atomicMax",
                CALL(TEMPLATED_CALLEE_WITH_ARGS(
                         MapNames::getDpctNamespace() + "atomic_fetch_max",
                         LITERAL(MapNames::getClNamespace() +
                                 "access::address_space::local_space")),
                     makeCallArgCreatorWithCall(0),
                     makeCallArgCreatorWithCall(1))),
            CONDITIONAL_FACTORY_ENTRY(
                ReportMemoryAttrDeduce(0),
                WARNING_FACTORY_ENTRY(
                    "atomicMax",
                    CALL_FACTORY_ENTRY("atomicMax",
                                       CALL(MapNames::getDpctNamespace() +
                                                "atomic_fetch_max",
                                            makeCallArgCreatorWithCall(0),
                                            makeCallArgCreatorWithCall(1))),
                    Diagnostics::SHARE_MEMORY_ATTR_DEDUCE,
                    makeCallArgCreatorWithCall(0),
                    std::string("atomic_fetch_max"),
                    std::string("atomic_fetch_max<") +
                        MapNames::getClNamespace() +
                        std::string("access::address_space::local_space>")),
                CALL_FACTORY_ENTRY("atomicMax",
                                   CALL(MapNames::getDpctNamespace() +
                                            "atomic_fetch_max",
                                        makeCallArgCreatorWithCall(0),
                                        makeCallArgCreatorWithCall(1)))))))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
    CONDITIONAL_FACTORY_ENTRY(
        GetUsingGenericSpace,
        CALL_FACTORY_ENTRY(
            "atomicMin",
            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                     MapNames::getDpctNamespace() + "atomic_fetch_min",
                     LITERAL(MapNames::getClNamespace() +
                             "access::address_space::generic_space")),
                 makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1))),
        CONDITIONAL_FACTORY_ENTRY(
            GetHasSharedAttr(0),
            CALL_FACTORY_ENTRY(
                "atomicMin",
                CALL(TEMPLATED_CALLEE_WITH_ARGS(
                         MapNames::getDpctNamespace() + "atomic_fetch_min",
                         LITERAL(MapNames::getClNamespace() +
                                 "access::address_space::local_space")),
                     makeCallArgCreatorWithCall(0),
                     makeCallArgCreatorWithCall(1))),
            CONDITIONAL_FACTORY_ENTRY(
                ReportMemoryAttrDeduce(0),
                WARNING_FACTORY_ENTRY(
                    "atomicMin",
                    CALL_FACTORY_ENTRY("atomicMin",
                                       CALL(MapNames::getDpctNamespace() +
                                                "atomic_fetch_min",
                                            makeCallArgCreatorWithCall(0),
                                            makeCallArgCreatorWithCall(1))),
                    Diagnostics::SHARE_MEMORY_ATTR_DEDUCE,
                    makeCallArgCreatorWithCall(0),
                    std::string("atomic_fetch_min"),
                    std::string("atomic_fetch_min<") +
                        MapNames::getClNamespace() +
                        std::string("access::address_space::local_space>")),
                CALL_FACTORY_ENTRY("atomicMin",
                                   CALL(MapNames::getDpctNamespace() +
                                            "atomic_fetch_min",
                                        makeCallArgCreatorWithCall(0),
                                        makeCallArgCreatorWithCall(1)))))))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
    CONDITIONAL_FACTORY_ENTRY(
        GetUsingGenericSpace,
        CALL_FACTORY_ENTRY(
            "atomicCAS",
            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                     MapNames::getDpctNamespace() +
                         "atomic_compare_exchange_strong",
                     LITERAL(MapNames::getClNamespace() +
                             "access::address_space::generic_space")),
                 makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1),
                 makeCallArgCreatorWithCall(2))),
        CONDITIONAL_FACTORY_ENTRY(
            GetHasSharedAttr(0),
            CALL_FACTORY_ENTRY(
                "atomicCAS",
                CALL(TEMPLATED_CALLEE_WITH_ARGS(
                         MapNames::getDpctNamespace() +
                             "atomic_compare_exchange_strong",
                         LITERAL(MapNames::getClNamespace() +
                                 "access::address_space::local_space")),
                     makeCallArgCreatorWithCall(0),
                     makeCallArgCreatorWithCall(1),
                     makeCallArgCreatorWithCall(2))),
            CONDITIONAL_FACTORY_ENTRY(
                ReportMemoryAttrDeduce(0),
                WARNING_FACTORY_ENTRY(
                    "atomicCAS",
                    CALL_FACTORY_ENTRY(
                        "atomicCAS", CALL(MapNames::getDpctNamespace() +
                                              "atomic_compare_exchange_strong",
                                          makeCallArgCreatorWithCall(0),
                                          makeCallArgCreatorWithCall(1),
                                          makeCallArgCreatorWithCall(2))),
                    Diagnostics::SHARE_MEMORY_ATTR_DEDUCE,
                    makeCallArgCreatorWithCall(0),
                    std::string("atomic_compare_exchange_strong"),
                    std::string("atomic_compare_exchange_strong<") +
                        MapNames::getClNamespace() +
                        std::string("access::address_space::local_space>")),
                CALL_FACTORY_ENTRY("atomicCAS",
                                   CALL(MapNames::getDpctNamespace() +
                                            "atomic_compare_exchange_strong",
                                        makeCallArgCreatorWithCall(0),
                                        makeCallArgCreatorWithCall(1),
                                        makeCallArgCreatorWithCall(2)))))))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
    CONDITIONAL_FACTORY_ENTRY(
        GetUsingGenericSpace,
        CALL_FACTORY_ENTRY(
            "atomicInc",
            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                     MapNames::getDpctNamespace() + "atomic_fetch_compare_inc",
                     LITERAL(MapNames::getClNamespace() +
                             "access::address_space::generic_space")),
                 makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1))),
        CONDITIONAL_FACTORY_ENTRY(
            GetHasSharedAttr(0),
            CALL_FACTORY_ENTRY(
                "atomicInc",
                CALL(TEMPLATED_CALLEE_WITH_ARGS(
                         MapNames::getDpctNamespace() +
                             "atomic_fetch_compare_inc",
                         LITERAL(MapNames::getClNamespace() +
                                 "access::address_space::local_space")),
                     makeCallArgCreatorWithCall(0),
                     makeCallArgCreatorWithCall(1))),
            CONDITIONAL_FACTORY_ENTRY(
                ReportMemoryAttrDeduce(0),
                WARNING_FACTORY_ENTRY(
                    "atomicInc",
                    CALL_FACTORY_ENTRY("atomicInc",
                                       CALL(MapNames::getDpctNamespace() +
                                                "atomic_fetch_compare_inc",
                                            makeCallArgCreatorWithCall(0),
                                            makeCallArgCreatorWithCall(1))),
                    Diagnostics::SHARE_MEMORY_ATTR_DEDUCE,
                    makeCallArgCreatorWithCall(0),
                    std::string("atomic_fetch_compare_inc"),
                    std::string("atomic_fetch_compare_inc<") +
                        MapNames::getClNamespace() +
                        std::string("access::address_space::local_space>")),
                CALL_FACTORY_ENTRY("atomicInc",
                                   CALL(MapNames::getDpctNamespace() +
                                            "atomic_fetch_compare_inc",
                                        makeCallArgCreatorWithCall(0),
                                        makeCallArgCreatorWithCall(1)))))))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
    CONDITIONAL_FACTORY_ENTRY(
        GetUsingGenericSpace,
        CALL_FACTORY_ENTRY(
            "atomicDec",
            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                     MapNames::getDpctNamespace() + "atomic_fetch_compare_dec",
                     LITERAL(MapNames::getClNamespace() +
                             "access::address_space::generic_space")),
                 makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1))),
        CONDITIONAL_FACTORY_ENTRY(
            GetHasSharedAttr(0),
            CALL_FACTORY_ENTRY(
                "atomicDec",
                CALL(TEMPLATED_CALLEE_WITH_ARGS(
                         MapNames::getDpctNamespace() +
                             "atomic_fetch_compare_dec",
                         LITERAL(MapNames::getClNamespace() +
                                 "access::address_space::local_space")),
                     makeCallArgCreatorWithCall(0),
                     makeCallArgCreatorWithCall(1))),
            CONDITIONAL_FACTORY_ENTRY(
                ReportMemoryAttrDeduce(0),
                WARNING_FACTORY_ENTRY(
                    "atomicDec",
                    CALL_FACTORY_ENTRY("atomicDec",
                                       CALL(MapNames::getDpctNamespace() +
                                                "atomic_fetch_compare_dec",
                                            makeCallArgCreatorWithCall(0),
                                            makeCallArgCreatorWithCall(1))),
                    Diagnostics::SHARE_MEMORY_ATTR_DEDUCE,
                    makeCallArgCreatorWithCall(0),
                    std::string("atomic_fetch_compare_dec"),
                    std::string("atomic_fetch_compare_dec<") +
                        MapNames::getClNamespace() +
                        std::string("access::address_space::local_space>")),
                CALL_FACTORY_ENTRY("atomicDec",
                                   CALL(MapNames::getDpctNamespace() +
                                            "atomic_fetch_compare_dec",
                                        makeCallArgCreatorWithCall(0),
                                        makeCallArgCreatorWithCall(1)))))))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
    CONDITIONAL_FACTORY_ENTRY(
        GetUsingGenericSpace,
        CALL_FACTORY_ENTRY(
            "atomicAnd",
            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                     MapNames::getDpctNamespace() + "atomic_fetch_and",
                     LITERAL(MapNames::getClNamespace() +
                             "access::address_space::generic_space")),
                 makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1))),
        CONDITIONAL_FACTORY_ENTRY(
            GetHasSharedAttr(0),
            CALL_FACTORY_ENTRY(
                "atomicAnd",
                CALL(TEMPLATED_CALLEE_WITH_ARGS(
                         MapNames::getDpctNamespace() + "atomic_fetch_and",
                         LITERAL(MapNames::getClNamespace() +
                                 "access::address_space::local_space")),
                     makeCallArgCreatorWithCall(0),
                     makeCallArgCreatorWithCall(1))),
            CONDITIONAL_FACTORY_ENTRY(
                ReportMemoryAttrDeduce(0),
                WARNING_FACTORY_ENTRY(
                    "atomicAnd",
                    CALL_FACTORY_ENTRY("atomicAnd",
                                       CALL(MapNames::getDpctNamespace() +
                                                "atomic_fetch_and",
                                            makeCallArgCreatorWithCall(0),
                                            makeCallArgCreatorWithCall(1))),
                    Diagnostics::SHARE_MEMORY_ATTR_DEDUCE,
                    makeCallArgCreatorWithCall(0),
                    std::string("atomic_fetch_and"),
                    std::string("atomic_fetch_and<") +
                        MapNames::getClNamespace() +
                        std::string("access::address_space::local_space>")),
                CALL_FACTORY_ENTRY("atomicAnd",
                                   CALL(MapNames::getDpctNamespace() +
                                            "atomic_fetch_and",
                                        makeCallArgCreatorWithCall(0),
                                        makeCallArgCreatorWithCall(1)))))))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
    CONDITIONAL_FACTORY_ENTRY(
        GetUsingGenericSpace,
        CALL_FACTORY_ENTRY(
            "atomicOr",
            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                     MapNames::getDpctNamespace() + "atomic_fetch_or",
                     LITERAL(MapNames::getClNamespace() +
                             "access::address_space::generic_space")),
                 makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1))),
        CONDITIONAL_FACTORY_ENTRY(
            GetHasSharedAttr(0),
            CALL_FACTORY_ENTRY(
                "atomicOr",
                CALL(TEMPLATED_CALLEE_WITH_ARGS(
                         MapNames::getDpctNamespace() + "atomic_fetch_or",
                         LITERAL(MapNames::getClNamespace() +
                                 "access::address_space::local_space")),
                     makeCallArgCreatorWithCall(0),
                     makeCallArgCreatorWithCall(1))),
            CONDITIONAL_FACTORY_ENTRY(
                ReportMemoryAttrDeduce(0),
                WARNING_FACTORY_ENTRY(
                    "atomicOr",
                    CALL_FACTORY_ENTRY("atomicOr",
                                       CALL(MapNames::getDpctNamespace() +
                                                "atomic_fetch_or",
                                            makeCallArgCreatorWithCall(0),
                                            makeCallArgCreatorWithCall(1))),
                    Diagnostics::SHARE_MEMORY_ATTR_DEDUCE,
                    makeCallArgCreatorWithCall(0),
                    std::string("atomic_fetch_or"),
                    std::string("atomic_fetch_or<") +
                        MapNames::getClNamespace() +
                        std::string("access::address_space::local_space>")),
                CALL_FACTORY_ENTRY("atomicOr",
                                   CALL(MapNames::getDpctNamespace() +
                                            "atomic_fetch_or",
                                        makeCallArgCreatorWithCall(0),
                                        makeCallArgCreatorWithCall(1)))))))

SET_NEED_DPCT_DEVICE_EXT_FACTORY(
    CONDITIONAL_FACTORY_ENTRY(
        GetUsingGenericSpace,
        CALL_FACTORY_ENTRY(
            "atomicXor",
            CALL(TEMPLATED_CALLEE_WITH_ARGS(
                     MapNames::getDpctNamespace() + "atomic_fetch_xor",
                     LITERAL(MapNames::getClNamespace() +
                             "access::address_space::generic_space")),
                 makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1))),
        CONDITIONAL_FACTORY_ENTRY(
            GetHasSharedAttr(0),
            CALL_FACTORY_ENTRY(
                "atomicXor",
                CALL(TEMPLATED_CALLEE_WITH_ARGS(
                         MapNames::getDpctNamespace() + "atomic_fetch_xor",
                         LITERAL(MapNames::getClNamespace() +
                                 "access::address_space::local_space")),
                     makeCallArgCreatorWithCall(0),
                     makeCallArgCreatorWithCall(1))),
            CONDITIONAL_FACTORY_ENTRY(
                ReportMemoryAttrDeduce(0),
                WARNING_FACTORY_ENTRY(
                    "atomicXor",
                    CALL_FACTORY_ENTRY("atomicXor",
                                       CALL(MapNames::getDpctNamespace() +
                                                "atomic_fetch_xor",
                                            makeCallArgCreatorWithCall(0),
                                            makeCallArgCreatorWithCall(1))),
                    Diagnostics::SHARE_MEMORY_ATTR_DEDUCE,
                    makeCallArgCreatorWithCall(0),
                    std::string("atomic_fetch_xor"),
                    std::string("atomic_fetch_xor<") +
                        MapNames::getClNamespace() +
                        std::string("access::address_space::local_space>")),
                CALL_FACTORY_ENTRY("atomicXor",
                                   CALL(MapNames::getDpctNamespace() +
                                            "atomic_fetch_xor",
                                        makeCallArgCreatorWithCall(0),
                                        makeCallArgCreatorWithCall(1)))))))