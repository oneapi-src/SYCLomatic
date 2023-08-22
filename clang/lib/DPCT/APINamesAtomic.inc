//===--------------- APINamesAtomic.inc -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

CALL_FACTORY_ENTRY(
    "atomicAdd",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_fetch_add",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))

CALL_FACTORY_ENTRY(
    "atomicAdd_system",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_fetch_add",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space"),
        LITERAL(MapNames::getClNamespace() +
                "memory_order::relaxed"),
        LITERAL(MapNames::getClNamespace() +
                "memory_scope::system")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))


CALL_FACTORY_ENTRY(
    "atomicSub",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_fetch_sub",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))


CALL_FACTORY_ENTRY(
    "atomicSub_system",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_fetch_sub",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space"),
        LITERAL(MapNames::getClNamespace() +
                "memory_order::relaxed"),
        LITERAL(MapNames::getClNamespace() +
                "memory_scope::system")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))


CALL_FACTORY_ENTRY(
    "atomicExch",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_exchange",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))

CALL_FACTORY_ENTRY(
    "atomicExch_system",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_exchange",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space"),
        LITERAL(MapNames::getClNamespace() +
                "memory_order::relaxed"),
        LITERAL(MapNames::getClNamespace() +
                "memory_scope::system")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))


CALL_FACTORY_ENTRY(
    "atomicMax",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_fetch_max",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))


CALL_FACTORY_ENTRY(
    "atomicMax_system",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_fetch_max",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space"),
        LITERAL(MapNames::getClNamespace() +
                "memory_order::relaxed"),
        LITERAL(MapNames::getClNamespace() +
                "memory_scope::system")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))


CALL_FACTORY_ENTRY(
    "atomicMin",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_fetch_min",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))


CALL_FACTORY_ENTRY(
    "atomicMin_system",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_fetch_min",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space"),
        LITERAL(MapNames::getClNamespace() +
                "memory_order::relaxed"),
        LITERAL(MapNames::getClNamespace() +
                "memory_scope::system")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))

CALL_FACTORY_ENTRY(
    "atomicCAS",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() +
            "atomic_compare_exchange_strong",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1),
    makeCallArgCreatorWithCall(2)))

CALL_FACTORY_ENTRY(
    "atomicCAS_system",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() +
            "atomic_compare_exchange_strong",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space"),
        LITERAL(MapNames::getClNamespace() +
                "memory_order::relaxed"),
        LITERAL(MapNames::getClNamespace() +
                "memory_scope::system")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1),
    makeCallArgCreatorWithCall(2)))


CALL_FACTORY_ENTRY(
    "atomicInc",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_fetch_compare_inc",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))


CALL_FACTORY_ENTRY(
    "atomicInc_system",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_fetch_compare_inc",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space"),
        LITERAL(MapNames::getClNamespace() +
                "memory_order::relaxed"),
        LITERAL(MapNames::getClNamespace() +
                "memory_scope::system")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))

CALL_FACTORY_ENTRY(
    "atomicDec",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_fetch_compare_dec",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))

CALL_FACTORY_ENTRY(
    "atomicDec_system",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_fetch_compare_dec",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space"),
        LITERAL(MapNames::getClNamespace() +
                "memory_order::relaxed"),
        LITERAL(MapNames::getClNamespace() +
                "memory_scope::system")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))


CALL_FACTORY_ENTRY(
    "atomicAnd",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_fetch_and",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))


CALL_FACTORY_ENTRY(
    "atomicAnd_system",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_fetch_and",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space"),
        LITERAL(MapNames::getClNamespace() +
                "memory_order::relaxed"),
        LITERAL(MapNames::getClNamespace() +
                "memory_scope::system")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))

CALL_FACTORY_ENTRY(
    "atomicOr",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_fetch_or",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))


CALL_FACTORY_ENTRY(
    "atomicOr_system",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_fetch_or",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space"),
        LITERAL(MapNames::getClNamespace() +
                "memory_order::relaxed"),
        LITERAL(MapNames::getClNamespace() +
                "memory_scope::system")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))


CALL_FACTORY_ENTRY(
    "atomicXor",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_fetch_xor",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))

CALL_FACTORY_ENTRY(
    "atomicXor_system",
    CALL(TEMPLATED_CALLEE_WITH_ARGS(
        MapNames::getDpctNamespace() + "atomic_fetch_xor",
        LITERAL(MapNames::getClNamespace() +
                "access::address_space::generic_space"),
        LITERAL(MapNames::getClNamespace() +
                "memory_order::relaxed"),
        LITERAL(MapNames::getClNamespace() +
                "memory_scope::system")),
    makeCallArgCreatorWithCall(0), makeCallArgCreatorWithCall(1)))
