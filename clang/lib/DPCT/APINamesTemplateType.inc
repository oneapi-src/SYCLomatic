//===--------------- APINamesTemplateType.inc -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off

TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("cuda::atomic", 2),
        TYPE_CONDITIONAL_FACTORY(
        CheckTemplateArgCount(2),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "atomic"),
                     TEMPLATE_ARG(0),
                     TEMPLATE_ARG(1),
                     STR(MapNames::getClNamespace() + "memory_order::relaxed")),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "atomic"),
                     TEMPLATE_ARG(0))))

TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("cuda::std::atomic"),
    TYPE_CONDITIONAL_FACTORY(
        CheckTemplateArgCount(2),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "atomic"),
                     TEMPLATE_ARG(0),
                     TEMPLATE_ARG(1),
                     STR(MapNames::getClNamespace() + "memory_order::relaxed")),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "atomic"),
                     TEMPLATE_ARG(0))))

TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("thrust::not_equal_to", 1),
    TYPE_FACTORY(STR("std::not_equal_to"), TEMPLATE_ARG(0)))

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("thrust::constant_iterator", 3),
    TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "constant_iterator"),
                     TEMPLATE_ARG(0))))

TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("cudaExtent"),
    TYPE_CONDITIONAL_FACTORY(
        CheckTypeNameAndInVarDecl("cudaExtent"),
        CTAD_TYPE_FACTORY(STR(MapNames::getClNamespace() + "range"), STR("3")),
        TYPE_FACTORY(STR(MapNames::getClNamespace() + "range"), STR("3"))))

// TYPE_REWRITE_ENTRY(
//     "cub::CountingInputIterator",
//     HEADER_INSERTION_FACTORY(HeaderType::HT_DPL_Iterator,
//                              TYPE_FACTORY(STR("oneapi::dpl::counting_iterator"),
//                                           TEMPLATE_ARG(0))))

// TYPE_REWRITE_ENTRY("cub::TransformInputIterator",
//                    HEADER_INSERTION_FACTORY(
//                        HeaderType::HT_DPL_Iterator,
//                        TYPE_FACTORY(STR("oneapi::dpl::transform_iterator"),
//                                     TEMPLATE_ARG(2), TEMPLATE_ARG(1))))

// TYPE_REWRITE_ENTRY("cub::ConstantInputIterator",
//                    HEADER_INSERTION_FACTORY(HeaderType::HT_DPCT_DPL_Utils,
//                    TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "constant_iterator"), TEMPLATE_ARG(0))))

// FEATURE_REQUEST_FACTORY(
//     HelperFeatureEnum::device_ext,
//     TYPE_REWRITE_ENTRY("cub::ArgIndexInputIterator",
//                        HEADER_INSERTION_FACTORY(
//                            HeaderType::HT_DPCT_DPL_Utils,
//                            TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "arg_index_input_iterator"),
//                                         TEMPLATE_ARG(0)))))

// TYPE_REWRITE_ENTRY(
//     "cub::DiscardOutputIterator",
//     HEADER_INSERTION_FACTORY(HeaderType::HT_DPL_Iterator,
//                              TYPE_FACTORY(STR("oneapi::dpl::discard_iterator"))))

// TYPE_REWRITE_ENTRY("cub::Sum",
//                    TYPE_FACTORY(STR(MapNames::getClNamespace() + "plus"),
//                                 STR("")))

// TYPE_REWRITE_ENTRY("cub::Min",
//                    TYPE_FACTORY(STR(MapNames::getClNamespace() + "minimum"),
//                                 STR("")))

// TYPE_REWRITE_ENTRY("cub::Max",
//                    TYPE_FACTORY(STR(MapNames::getClNamespace() + "maximum"),
//                                 STR("")))

// TYPE_REWRITE_ENTRY("cub::Equality", TYPE_FACTORY(STR("std::equal_to"), STR("")))

// TYPE_REWRITE_ENTRY(
//     "cub::NullType",
//     HEADER_INSERTION_FACTORY(HeaderType::HT_DPCT_DPL_Utils,
//                              TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "null_type"))))

// FEATURE_REQUEST_FACTORY(
//     HelperFeatureEnum::device_ext,
//     TYPE_REWRITE_ENTRY("cub::KeyValuePair",
//                        HEADER_INSERTION_FACTORY(
//                            HeaderType::HT_DPCT_DPL_Utils,
//                            TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "key_value_pair"),
//                                         TEMPLATE_ARG(0), TEMPLATE_ARG(1)))))
// TYPE_REWRITE_ENTRY("cub::ArgMax",
//                     HEADER_INSERTION_FACTORY(
//                            HeaderType::HT_DPCT_DPL_Utils,
//                    TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "argmax"))))
// TYPE_REWRITE_ENTRY("cub::ArgMin",
//                     HEADER_INSERTION_FACTORY(
//                            HeaderType::HT_DPCT_DPL_Utils,
//                    TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "argmin"))))
FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    TYPE_REWRITE_ENTRY(TypeMatchingDesc("thrust::system::cuda::experimental::pinned_allocator", 1),
                       TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                                        "deprecated::usm_host_allocator"),
                                    TEMPLATE_ARG(0))))

TYPE_REWRITE_ENTRY(TypeMatchingDesc("thrust::detail::enable_if", 2),
                   TYPE_FACTORY(STR("std::enable_if"),
                                TEMPLATE_ARG(0), TEMPLATE_ARG(1)))

TYPE_REWRITE_ENTRY(TypeMatchingDesc("thrust::detail::is_same", 2),
                   TYPE_FACTORY(STR("std::is_same"),
                                TEMPLATE_ARG(0), TEMPLATE_ARG(1)))

TYPE_REWRITE_ENTRY(TypeMatchingDesc("thrust::detail::integral_constant", 2),
                   TYPE_FACTORY(STR("std::integral_constant"),
                                TEMPLATE_ARG(0), TEMPLATE_ARG(1)))

TYPE_REWRITE_ENTRY(TypeMatchingDesc("thrust::detail::true_type"),
                   TYPE_FACTORY(STR("std::true_type")))

TYPE_REWRITE_ENTRY(TypeMatchingDesc("thrust::detail::false_type"),
                   TYPE_FACTORY(STR("std::false_type")))

TYPE_REWRITE_ENTRY(TypeMatchingDesc("thrust::system::detail::bad_alloc"),
                    TYPE_FACTORY(STR("std::bad_alloc")))

TYPE_REWRITE_ENTRY(TypeMatchingDesc("thrust::iterator_traits", 1),
                    TYPE_FACTORY(STR("oneapi::dpl::iterator_traits"),
                                 TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY(TypeMatchingDesc("thrust::detail::vector_base", 2),
                    TYPE_FACTORY(STR("dpct::device_vector"),
                                 TEMPLATE_ARG(0), TEMPLATE_ARG(1)))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    TYPE_REWRITE_ENTRY(TypeMatchingDesc("thrust::cuda::experimental::pinned_allocator", 1),
                       TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                                        "deprecated::usm_host_allocator"),
                                    TEMPLATE_ARG(0))))

// thrust::greater_equal
TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("thrust::greater_equal", 1),
    TYPE_FACTORY(STR("std::greater_equal"), TEMPLATE_ARG(0)))

// thrust::less_equal
TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("thrust::less_equal"),
    TYPE_FACTORY(STR("std::less_equal")))

// thrust::device_allocator
TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("thrust::device_allocator"),
    TYPE_FACTORY(STR("dpct::deprecated::usm_device_allocator")))

// logical_and
TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("thrust::logical_and"),
    TYPE_FACTORY(STR("std::logical_and")))

//  thrust::system_error(
TYPE_REWRITE_ENTRY(TypeMatchingDesc("thrust::system::system_error"),
                   TYPE_FACTORY(STR("std::system_error")))

// thrust::system::error_code
TYPE_REWRITE_ENTRY(TypeMatchingDesc("thrust::system::error_code"),
                   TYPE_FACTORY(STR("std::error_code")))

// thrust::system::error_condition
TYPE_REWRITE_ENTRY(TypeMatchingDesc("thrust::system::error_condition"),
                   TYPE_FACTORY(STR("std::error_condition")))

// thrust::bit_and
TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("thrust::bit_and", 1),
    TYPE_FACTORY(STR("std::bit_and"), TEMPLATE_ARG(0)))

// thrust::bit_or
TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("thrust::bit_or", 1),
    TYPE_FACTORY(STR("std::bit_or"), TEMPLATE_ARG(0)))

// thrust::minimum
TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("thrust::minimum", 1),
    TYPE_FACTORY(STR("oneapi::dpl::minimum"), TEMPLATE_ARG(0)))

// thrust::optional
TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("thrust::optional", 1),
    TYPE_FACTORY(STR("std::optional"), TEMPLATE_ARG(0)))

//thrust::bit_xor
TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("thrust::bit_xor", 1),
    TYPE_FACTORY(STR("std::bit_xor"), TEMPLATE_ARG(0)))

//thrust::modulus
TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("thrust::modulus", 1),
    TYPE_FACTORY(STR("std::modulus"), TEMPLATE_ARG(0)))

//thrust::reverse_iterator
TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("thrust::reverse_iterator", 1),
    TYPE_FACTORY(STR("oneapi::dpl::reverse_iterator"), TEMPLATE_ARG(0)))

// TYPE_REWRITE_ENTRY(
//     "cub::DoubleBuffer",
//     TYPE_CONDITIONAL_FACTORY(
//         CheckForPostfixDeclaratorType(0),
//         WARNING_FACTORY(Diagnostics::KNOWN_UNSUPPORTED_TYPE, TYPESTR),
//         HEADER_INSERTION_FACTORY(HeaderType::HT_DPCT_DPL_Utils,
//         TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "io_iterator_pair"), ADD_POINTER(TEMPLATE_ARG(0))))))

// TYPE_REWRITE_ENTRY(
//     "cudaPos",
//     TYPE_CONDITIONAL_FACTORY(
//         CheckTypeNameAndInVarDecl("cudaPos"),
//         CTAD_TYPE_FACTORY(STR(MapNames::getClNamespace() + "id"), STR("3")),
//         TYPE_FACTORY(STR(MapNames::getClNamespace() + "id"), STR("3"))))

TYPE_REWRITE_ENTRY(TypeMatchingDesc("thrust::identity", 1),
                   TYPE_FACTORY(STR("oneapi::dpl::identity")))

TYPE_REWRITE_ENTRY(TypeMatchingDesc("thrust::null_type"),
                   TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "null_type")))

TYPE_REWRITE_ENTRY(TypeMatchingDesc("thrust::device_system_tag"),
                   TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "device_sys_tag")))

TYPE_REWRITE_ENTRY(TypeMatchingDesc("thrust::pointer", 2),
                   TYPE_FACTORY(STR(MapNames::getDpctNamespace() + "tagged_pointer"),
                                TEMPLATE_ARG(0),
                                TEMPLATE_ARG(1)))

// TYPE_REWRITE_ENTRY("cooperative_groups::__v1::plus",
//                    TYPE_FACTORY(STR(MapNames::getClNamespace() + "plus"),
//                                 TEMPLATE_ARG(0)))

// TYPE_REWRITE_ENTRY("cooperative_groups::__v1::less",
//                    TYPE_FACTORY(STR(MapNames::getClNamespace() + "minimum"),
//                                 TEMPLATE_ARG(0)))

// TYPE_REWRITE_ENTRY("cooperative_groups::__v1::greater",
//                    TYPE_FACTORY(STR(MapNames::getClNamespace() + "maximum"),
//                                 TEMPLATE_ARG(0)))

// TYPE_REWRITE_ENTRY("cooperative_groups::__v1::bit_and",
//                    TYPE_FACTORY(STR(MapNames::getClNamespace() + "bit_and"),
//                                 TEMPLATE_ARG(0)))

// TYPE_REWRITE_ENTRY("cooperative_groups::__v1::bit_xor",
//                    TYPE_FACTORY(STR(MapNames::getClNamespace() + "bit_xor"),
//                                 TEMPLATE_ARG(0)))

// TYPE_REWRITE_ENTRY("cooperative_groups::__v1::bit_or",
//                    TYPE_FACTORY(STR(MapNames::getClNamespace() + "bit_or"),
//                                 TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY(TypeMatchingDesc("thrust::random::default_random_engine"),
                   TYPE_FACTORY(STR("oneapi::dpl::default_engine")))

TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("thrust::random::uniform_real_distribution", 1),
    TYPE_FACTORY(STR("oneapi::dpl::uniform_real_distribution"), TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("thrust::random::normal_distribution", 1),
    TYPE_FACTORY(STR("oneapi::dpl::normal_distribution"), TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("thrust::random::uniform_int_distribution", 1),
    TYPE_FACTORY(STR("oneapi::dpl::uniform_int_distribution"), TEMPLATE_ARG(0)))

// TYPE_REWRITE_ENTRY("nvcuda::wmma::matrix_a",
//                    TYPE_CONDITIONAL_FACTORY(
//                        checkEnableJointMatrixForType(),
//                        TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
//                                         "experimental::matrix::a")),
//                        WARNING_FACTORY(Diagnostics::KNOWN_UNSUPPORTED_TYPE,
//                                        TYPESTR)))

// TYPE_REWRITE_ENTRY("nvcuda::wmma::matrix_b",
//                    TYPE_CONDITIONAL_FACTORY(
//                        checkEnableJointMatrixForType(),
//                        TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
//                                         "experimental::matrix::b")),
//                        WARNING_FACTORY(Diagnostics::KNOWN_UNSUPPORTED_TYPE,
//                                        TYPESTR)))

// TYPE_REWRITE_ENTRY("nvcuda::wmma::row_major",
//                    TYPE_CONDITIONAL_FACTORY(
//                        checkEnableJointMatrixForType(),
//                        TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
//                                         "experimental::matrix::row_major")),
//                        WARNING_FACTORY(Diagnostics::KNOWN_UNSUPPORTED_TYPE,
//                                        TYPESTR)))

// TYPE_REWRITE_ENTRY("nvcuda::wmma::col_major",
//                    TYPE_CONDITIONAL_FACTORY(
//                        checkEnableJointMatrixForType(),
//                        TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
//                                         "experimental::matrix::col_major")),
//                        WARNING_FACTORY(Diagnostics::KNOWN_UNSUPPORTED_TYPE,
//                                        TYPESTR)))

// TYPE_REWRITE_ENTRY("nvcuda::wmma::accumulator",
//                    TYPE_CONDITIONAL_FACTORY(
//                        checkEnableJointMatrixForType(),
//                        TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
//                                         "experimental::matrix::accumulator")),
//                        WARNING_FACTORY(Diagnostics::KNOWN_UNSUPPORTED_TYPE,
//                                        TYPESTR)))

// TYPE_REWRITE_ENTRY(
//     "nvcuda::wmma::fragment",
//     TYPE_CONDITIONAL_FACTORY(
//         checkEnableJointMatrixForType(),
//         TYPE_CONDITIONAL_FACTORY(
//             CheckTemplateArgCount(5),
//             TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
//                              "experimental::matrix::joint_matrix"),
//                          TEMPLATE_ARG(0), TEMPLATE_ARG(1), TEMPLATE_ARG(2),
//                          TEMPLATE_ARG(3), TEMPLATE_ARG(4)),
//             TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
//                              "experimental::matrix::joint_matrix"),
//                          TEMPLATE_ARG(0), TEMPLATE_ARG(1), TEMPLATE_ARG(2),
//                          TEMPLATE_ARG(3), TEMPLATE_ARG(4), TEMPLATE_ARG(5))),
//         WARNING_FACTORY(Diagnostics::KNOWN_UNSUPPORTED_TYPE, TYPESTR)))

// TYPE_REWRITE_ENTRY(
//     "nvcuda::wmma::layout_t",
//     TYPE_CONDITIONAL_FACTORY(
//         checkEnableJointMatrixForType(),
//         TYPE_FACTORY(STR(MapNames::getClNamespace() +
//                          "ext::oneapi::experimental::matrix::layout")),
//         WARNING_FACTORY(Diagnostics::KNOWN_UNSUPPORTED_TYPE, TYPESTR)))

// // clang-format on

// TYPE_REWRITE_ENTRY("csrsv2Info_t",
//                    TYPE_FACTORY(STR("std::shared_ptr<" +
//                                     MapNames::getDpctNamespace() +
//                                     "sparse::optimize_info>")))
// TYPE_REWRITE_ENTRY("cusparseSolvePolicy_t", TYPE_FACTORY(STR("int")))
// TYPE_REWRITE_ENTRY("cusparseAction_t",
//                    TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
//                                     "sparse::conversion_scope")))

// TYPE_REWRITE_ENTRY(
//     "cooperative_groups::__v1::coalesced_group",
//     TYPE_CONDITIONAL_FACTORY(
//         isUseNonUniformGroupsForType(),
//         TYPE_FACTORY(STR(MapNames::getClNamespace() +
//                          "ext::oneapi::experimental::opportunistic_group")),
//         WARNING_FACTORY(Diagnostics::TRY_EXPERIMENTAL_FEATURE, TYPESTR,
//                         STR("--use-experimental-features=non-uniform-groups"))))

TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("cooperative_groups::__v1::thread_block_tile", 2),
    TYPE_CONDITIONAL_FACTORY(
        checkTemplateArgSpelling(0, "32"),
        TYPE_FACTORY(STR(MapNames::getClNamespace() + "sub_group")),
        TYPE_CONDITIONAL_FACTORY(
            isUseLogicalGroupsForType(),
            TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                             "experimental::logical_group")),
            WARNING_FACTORY(Diagnostics::TRY_EXPERIMENTAL_FEATURE, TYPESTR,
                            STR("--use-experimental-features=logical-group")))))

TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("cudaGraph_t"),
    TYPE_CONDITIONAL_FACTORY(
        checkEnableGraphForType(),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                         "experimental::command_graph_ptr")),
        WARNING_FACTORY(Diagnostics::TRY_EXPERIMENTAL_FEATURE, TYPESTR,
                        STR("--use-experimental-features=graph"))))

TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("cudaGraphExec_t"),
    TYPE_CONDITIONAL_FACTORY(
        checkEnableGraphForType(),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                         "experimental::command_graph_exec_ptr")),
        WARNING_FACTORY(Diagnostics::TRY_EXPERIMENTAL_FEATURE, TYPESTR,
                        STR("--use-experimental-features=graph"))))

TYPE_REWRITE_ENTRY(
    TypeMatchingDesc("cudaGraphNode_t"),
    TYPE_CONDITIONAL_FACTORY(
        checkEnableGraphForType(),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                         "experimental::node_ptr")),
        WARNING_FACTORY(Diagnostics::TRY_EXPERIMENTAL_FEATURE, TYPESTR,
                        STR("--use-experimental-features=graph"))))
