//===--------------- APINamesTemplateType.inc -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

TYPE_REWRITE_ENTRY(
    "cuda::atomic",
        TYPE_CONDITIONAL_FACTORY(
        CheckTemplateArgCount(2),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "atomic"),
                     TEMPLATE_ARG(0),
                     TEMPLATE_ARG(1),
                     STR(MapNames::getClNamespace() + "memory_order::relaxed")),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "atomic"),
                     TEMPLATE_ARG(0))))

TYPE_REWRITE_ENTRY(
    "cuda::std::atomic",
    TYPE_CONDITIONAL_FACTORY(
        CheckTemplateArgCount(2),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "atomic"),
                     TEMPLATE_ARG(0),
                     TEMPLATE_ARG(1),
                     STR(MapNames::getClNamespace() + "memory_order::relaxed")),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "atomic"),
                     TEMPLATE_ARG(0))))

TYPE_REWRITE_ENTRY(
    "thrust::not_equal_to",
    TYPE_FACTORY(STR("std::not_equal_to"), TEMPLATE_ARG(0)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::DplExtrasIterators_constant_iterator,
TYPE_REWRITE_ENTRY(
    "thrust::constant_iterator",
    TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "constant_iterator"),
                     TEMPLATE_ARG(0))))

TYPE_REWRITE_ENTRY(
    "cub::CountingInputIterator",
    HEADER_INSERTION_FACTORY(HeaderType::HT_DPL_Iterator,
                             TYPE_FACTORY(STR("oneapi::dpl::counting_iterator"),
                                          TEMPLATE_ARG(0))))

TYPE_REWRITE_ENTRY("cub::TransformInputIterator",
                   HEADER_INSERTION_FACTORY(
                       HeaderType::HT_DPL_Iterator,
                       TYPE_FACTORY(STR("oneapi::dpl::transform_iterator"),
                                    TEMPLATE_ARG(2), TEMPLATE_ARG(1))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Memory_usm_host_allocator_alias,
    TYPE_REWRITE_ENTRY("thrust::system::cuda::experimental::pinned_allocator",
                       TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                                        "deprecated::usm_host_allocator"),
                                    TEMPLATE_ARG(0))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::Memory_usm_host_allocator_alias,
    TYPE_REWRITE_ENTRY("thrust::cuda::experimental::pinned_allocator",
                       TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                                        "deprecated::usm_host_allocator"),
                                    TEMPLATE_ARG(0))))

// thrust::greater_equal
TYPE_REWRITE_ENTRY(
    "thrust::greater_equal",
    TYPE_FACTORY(STR("std::greater_equal"), TEMPLATE_ARG(0)))

// thrust::less_equal
TYPE_REWRITE_ENTRY(
    "thrust::less_equal",
    TYPE_FACTORY(STR("std::less_equal"), TEMPLATE_ARG(0)))

// logical_and
TYPE_REWRITE_ENTRY(
    "thrust::logical_and",
    TYPE_FACTORY(STR("std::logical_and"), TEMPLATE_ARG(0)))

// thrust::bit_and
TYPE_REWRITE_ENTRY(
    "thrust::bit_and",
    TYPE_FACTORY(STR("std::bit_and"), TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY(
    "thrust::bit_or",
    TYPE_FACTORY(STR("std::bit_or"), TEMPLATE_ARG(0)))

// thrust::bit_or
TYPE_REWRITE_ENTRY(
    "thrust::minimum",
    TYPE_FACTORY(STR("oneapi::dpl::minimum"), TEMPLATE_ARG(0)))

//thrust::bit_xor
TYPE_REWRITE_ENTRY(
    "thrust::bit_xor",
    TYPE_FACTORY(STR("std::bit_xor"), TEMPLATE_ARG(0)))
