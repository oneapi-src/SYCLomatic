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

FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
TYPE_REWRITE_ENTRY(
    "thrust::constant_iterator",
    TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() + "constant_iterator"),
                     TEMPLATE_ARG(0))))

TYPE_REWRITE_ENTRY(
    "cudaExtent",
    TYPE_CONDITIONAL_FACTORY(
        CheckTypeNameAndInVarDecl("cudaExtent"),
        CTAD_TYPE_FACTORY(STR(MapNames::getClNamespace() + "range"), STR("3")),
        TYPE_FACTORY(STR(MapNames::getClNamespace() + "range"), STR("3"))))

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

TYPE_REWRITE_ENTRY("cub::ConstantInputIterator",
                   HEADER_INSERTION_FACTORY(HeaderType::HT_DPCT_DPL_Utils,
                   TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() + "constant_iterator"), TEMPLATE_ARG(0))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    TYPE_REWRITE_ENTRY("cub::ArgIndexInputIterator",
                       HEADER_INSERTION_FACTORY(
                           HeaderType::HT_DPCT_DPL_Utils,
                           TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() + "arg_index_input_iterator"),
                                        TEMPLATE_ARG(0)))))

TYPE_REWRITE_ENTRY(
    "cub::DiscardOutputIterator",
    HEADER_INSERTION_FACTORY(HeaderType::HT_DPL_Iterator,
                             TYPE_FACTORY(STR("oneapi::dpl::discard_iterator"))))

TYPE_REWRITE_ENTRY("cub::Sum",
                   TYPE_FACTORY(STR(MapNames::getClNamespace() + "plus"),
                                STR("")))

TYPE_REWRITE_ENTRY("cub::Min",
                   TYPE_FACTORY(STR(MapNames::getClNamespace() + "minimum"),
                                STR("")))

TYPE_REWRITE_ENTRY("cub::Max",
                   TYPE_FACTORY(STR(MapNames::getClNamespace() + "maximum"),
                                STR("")))

TYPE_REWRITE_ENTRY("cub::Equality", TYPE_FACTORY(STR("std::equal_to"), STR("")))

TYPE_REWRITE_ENTRY(
    "cub::NullType",
    HEADER_INSERTION_FACTORY(HeaderType::HT_DPCT_DPL_Utils,
                             TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() + "null_type"))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    TYPE_REWRITE_ENTRY("cub::KeyValuePair",
                       HEADER_INSERTION_FACTORY(
                           HeaderType::HT_DPCT_DPL_Utils,
                           TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() + "key_value_pair"),
                                        TEMPLATE_ARG(0), TEMPLATE_ARG(1)))))
TYPE_REWRITE_ENTRY("cub::ArgMax",
                    HEADER_INSERTION_FACTORY(
                           HeaderType::HT_DPCT_DPL_Utils,
                   TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() + "argmax"))))
TYPE_REWRITE_ENTRY("cub::ArgMin",
                    HEADER_INSERTION_FACTORY(
                           HeaderType::HT_DPCT_DPL_Utils,
                   TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() + "argmin"))))
// cub::BlockRadixSort
TYPE_REWRITE_ENTRY(
    "cub::BlockRadixSort",
    TYPE_CONDITIONAL_FACTORY(
        UseSYCLCompat(),
        WARNING_FACTORY(Diagnostics::UNSUPPORT_SYCLCOMPAT, TYPESTR),
        HEADER_INSERTION_FACTORY(
            HeaderType::HT_DPCT_GROUP_Utils,
            TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() +
                             "group::group_radix_sort"),
                         TEMPLATE_ARG(0), TEMPLATE_ARG(2)))))

// cub::BlockExchange
TYPE_REWRITE_ENTRY(
    "cub::BlockExchange",
    TYPE_CONDITIONAL_FACTORY(
        UseSYCLCompat(),
        WARNING_FACTORY(Diagnostics::UNSUPPORT_SYCLCOMPAT, TYPESTR),
        HEADER_INSERTION_FACTORY(
            HeaderType::HT_DPCT_GROUP_Utils,
            TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() +
                             "group::exchange"),
                         TEMPLATE_ARG(0), TEMPLATE_ARG(2)))))
// cub::BlockLoad
TYPE_REWRITE_ENTRY(
    "cub::BlockLoad",
    TYPE_CONDITIONAL_FACTORY(
        UseSYCLCompat(),
        WARNING_FACTORY(Diagnostics::UNSUPPORT_SYCLCOMPAT, TYPESTR),
        HEADER_INSERTION_FACTORY(
            HeaderType::HT_DPCT_GROUP_Utils,
            TYPE_CONDITIONAL_FACTORY(
                CheckTemplateArgCount(4),
                TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() +
                                 "group::group_load"),
                             TEMPLATE_ARG(0), TEMPLATE_ARG(2), TEMPLATE_ARG(3)),
                TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() +
                                 "group::group_load"),
                             TEMPLATE_ARG(0), TEMPLATE_ARG(2))))))
// cub::BlockStore
TYPE_REWRITE_ENTRY(
    "cub::BlockStore",
    TYPE_CONDITIONAL_FACTORY(
        UseSYCLCompat(),
        WARNING_FACTORY(Diagnostics::UNSUPPORT_SYCLCOMPAT, TYPESTR),
        HEADER_INSERTION_FACTORY(
            HeaderType::HT_DPCT_GROUP_Utils,
            TYPE_CONDITIONAL_FACTORY(
                CheckTemplateArgCount(4),
                TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() +
                                 "group::group_store"),
                             TEMPLATE_ARG(0), TEMPLATE_ARG(2), TEMPLATE_ARG(3)),
                TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() +
                                 "group::group_store"),
                             TEMPLATE_ARG(0), TEMPLATE_ARG(2))))))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    TYPE_REWRITE_ENTRY("thrust::system::cuda::experimental::pinned_allocator",
                       TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() +
                                        "deprecated::usm_host_allocator"),
                                    TEMPLATE_ARG(0))))

TYPE_REWRITE_ENTRY("thrust::detail::enable_if",
                    TYPE_FACTORY(STR("std::enable_if"),
                                TEMPLATE_ARG(0), TEMPLATE_ARG(1)))

TYPE_REWRITE_ENTRY("thrust::detail::is_same",
                    TYPE_FACTORY(STR("std::is_same"),
                                TEMPLATE_ARG(0), TEMPLATE_ARG(1)))

TYPE_REWRITE_ENTRY("thrust::detail::integral_constant",
                    TYPE_FACTORY(STR("std::integral_constant"),
                                TEMPLATE_ARG(0), TEMPLATE_ARG(1)))

TYPE_REWRITE_ENTRY("thrust::detail::true_type",
                    TYPE_FACTORY(STR("std::true_type")))

TYPE_REWRITE_ENTRY("thrust::detail::false_type",
                    TYPE_FACTORY(STR("std::false_type")))

TYPE_REWRITE_ENTRY("thrust::system::detail::bad_alloc",
                    TYPE_FACTORY(STR("std::bad_alloc")))

TYPE_REWRITE_ENTRY("thrust::iterator_traits",
                    TYPE_FACTORY(STR("oneapi::dpl::iterator_traits"),
                                 TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY("thrust::detail::vector_base",
                    TYPE_FACTORY(STR("dpct::device_vector"),
                                 TEMPLATE_ARG(0), TEMPLATE_ARG(1)))

FEATURE_REQUEST_FACTORY(
    HelperFeatureEnum::device_ext,
    TYPE_REWRITE_ENTRY("thrust::cuda::experimental::pinned_allocator",
                       TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() +
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

// thrust::device_allocator
TYPE_REWRITE_ENTRY(
    "thrust::device_allocator",
    TYPE_FACTORY(STR("dpct::deprecated::usm_device_allocator"), TEMPLATE_ARG(0)))

// logical_and
TYPE_REWRITE_ENTRY(
    "thrust::logical_and",
    TYPE_FACTORY(STR("std::logical_and"), TEMPLATE_ARG(0)))

//  thrust::system_error(
TYPE_REWRITE_ENTRY("thrust::system::system_error",
                   TYPE_FACTORY(STR("std::system_error")))

// thrust::system::error_code
TYPE_REWRITE_ENTRY("thrust::system::error_code", TYPE_FACTORY(STR("std::error_code")))

// thrust::system::error_condition
TYPE_REWRITE_ENTRY("thrust::system::error_condition", TYPE_FACTORY(STR("std::error_condition")))

// thrust::system::errc::errc_t
TYPE_REWRITE_ENTRY("thrust::system::errc::errc_t",
                   TYPE_FACTORY(STR("std::errc")))

// thrust::bit_and
TYPE_REWRITE_ENTRY(
    "thrust::bit_and",
    TYPE_FACTORY(STR("std::bit_and"), TEMPLATE_ARG(0)))

// thrust::bit_or
TYPE_REWRITE_ENTRY(
    "thrust::bit_or",
    TYPE_FACTORY(STR("std::bit_or"), TEMPLATE_ARG(0)))

// thrust::minimum
TYPE_REWRITE_ENTRY(
    "thrust::minimum",
    TYPE_FACTORY(STR("oneapi::dpl::minimum"), TEMPLATE_ARG(0)))

// thrust::optional
TYPE_REWRITE_ENTRY(
    "thrust::optional",
    TYPE_FACTORY(STR("std::optional"), TEMPLATE_ARG(0)))

//thrust::bit_xor
TYPE_REWRITE_ENTRY(
    "thrust::bit_xor",
    TYPE_FACTORY(STR("std::bit_xor"), TEMPLATE_ARG(0)))

//thrust::modulus
TYPE_REWRITE_ENTRY(
    "thrust::modulus",
    TYPE_FACTORY(STR("std::modulus"), TEMPLATE_ARG(0)))

//thrust::reverse_iterator
TYPE_REWRITE_ENTRY(
    "thrust::reverse_iterator",
    TYPE_FACTORY(STR("oneapi::dpl::reverse_iterator"), TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY(
    "cub::DoubleBuffer",
    TYPE_CONDITIONAL_FACTORY(
        CheckForPostfixDeclaratorType(0),
        WARNING_FACTORY(Diagnostics::KNOWN_UNSUPPORTED_TYPE, TYPESTR),
        HEADER_INSERTION_FACTORY(HeaderType::HT_DPCT_DPL_Utils,
        TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() + "io_iterator_pair"), ADD_POINTER(TEMPLATE_ARG(0))))))

TYPE_REWRITE_ENTRY(
    "cudaPos",
    TYPE_CONDITIONAL_FACTORY(
        CheckTypeNameAndInVarDecl("cudaPos"),
        CTAD_TYPE_FACTORY(STR(MapNames::getClNamespace() + "id"), STR("3")),
        TYPE_FACTORY(STR(MapNames::getClNamespace() + "id"), STR("3"))))

TYPE_REWRITE_ENTRY("thrust::identity",
                   TYPE_FACTORY(STR("oneapi::dpl::identity")))

TYPE_REWRITE_ENTRY("thrust::null_type", TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() + "null_type")))

TYPE_REWRITE_ENTRY("thrust::device_system_tag", TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() + "device_sys_tag")))

TYPE_REWRITE_ENTRY("thrust::pointer", TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() + "tagged_pointer"),
                   TEMPLATE_ARG(0),
                   TEMPLATE_ARG(1)))

TYPE_REWRITE_ENTRY("cooperative_groups::__v1::plus",
                   TYPE_FACTORY(STR(MapNames::getClNamespace() + "plus"),
                                TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY("cooperative_groups::__v1::less",
                   TYPE_FACTORY(STR(MapNames::getClNamespace() + "minimum"),
                                TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY("cooperative_groups::__v1::greater",
                   TYPE_FACTORY(STR(MapNames::getClNamespace() + "maximum"),
                                TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY("cooperative_groups::__v1::bit_and",
                   TYPE_FACTORY(STR(MapNames::getClNamespace() + "bit_and"),
                                TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY("cooperative_groups::__v1::bit_xor",
                   TYPE_FACTORY(STR(MapNames::getClNamespace() + "bit_xor"),
                                TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY("cooperative_groups::__v1::bit_or",
                   TYPE_FACTORY(STR(MapNames::getClNamespace() + "bit_or"),
                                TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY("thrust::random::default_random_engine",
                   TYPE_FACTORY(STR("oneapi::dpl::default_engine")))

TYPE_REWRITE_ENTRY("thrust::random::uniform_real_distribution",
                   TYPE_FACTORY(STR("oneapi::dpl::uniform_real_distribution"), TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY("thrust::random::normal_distribution",
                   TYPE_FACTORY(STR("oneapi::dpl::normal_distribution"), TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY("thrust::random::uniform_int_distribution",
                   TYPE_FACTORY(STR("oneapi::dpl::uniform_int_distribution"), TEMPLATE_ARG(0)))

TYPE_REWRITE_ENTRY("nvcuda::wmma::matrix_a",
                   TYPE_CONDITIONAL_FACTORY(
                       checkEnableJointMatrixForType(),
                       TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                                        "experimental::matrix::a")),
                       WARNING_FACTORY(Diagnostics::KNOWN_UNSUPPORTED_TYPE,
                                       TYPESTR)))

TYPE_REWRITE_ENTRY("nvcuda::wmma::matrix_b",
                   TYPE_CONDITIONAL_FACTORY(
                       checkEnableJointMatrixForType(),
                       TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                                        "experimental::matrix::b")),
                       WARNING_FACTORY(Diagnostics::KNOWN_UNSUPPORTED_TYPE,
                                       TYPESTR)))

TYPE_REWRITE_ENTRY("nvcuda::wmma::row_major",
                   TYPE_CONDITIONAL_FACTORY(
                       checkEnableJointMatrixForType(),
                       TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                                        "experimental::matrix::row_major")),
                       WARNING_FACTORY(Diagnostics::KNOWN_UNSUPPORTED_TYPE,
                                       TYPESTR)))

TYPE_REWRITE_ENTRY("nvcuda::wmma::col_major",
                   TYPE_CONDITIONAL_FACTORY(
                       checkEnableJointMatrixForType(),
                       TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                                        "experimental::matrix::col_major")),
                       WARNING_FACTORY(Diagnostics::KNOWN_UNSUPPORTED_TYPE,
                                       TYPESTR)))

TYPE_REWRITE_ENTRY("nvcuda::wmma::accumulator",
                   TYPE_CONDITIONAL_FACTORY(
                       checkEnableJointMatrixForType(),
                       TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                                        "experimental::matrix::accumulator")),
                       WARNING_FACTORY(Diagnostics::KNOWN_UNSUPPORTED_TYPE,
                                       TYPESTR)))

TYPE_REWRITE_ENTRY(
    "nvcuda::wmma::fragment",
    TYPE_CONDITIONAL_FACTORY(
        checkEnableJointMatrixForType(),
        TYPE_CONDITIONAL_FACTORY(
            CheckTemplateArgCount(5),
            EMIT_WARNING_FACTORY(
                TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                                 "experimental::matrix::joint_matrix"),
                             TEMPLATE_ARG(0), TEMPLATE_ARG(1), TEMPLATE_ARG(2),
                             TEMPLATE_ARG(3), TEMPLATE_ARG(4)),
                Diagnostics::JOINT_MATRIX_SHAPE),
            EMIT_WARNING_FACTORY(
                TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                                 "experimental::matrix::joint_matrix"),
                             TEMPLATE_ARG(0), TEMPLATE_ARG(1), TEMPLATE_ARG(2),
                             TEMPLATE_ARG(3), TEMPLATE_ARG(4), TEMPLATE_ARG(5)),
                Diagnostics::JOINT_MATRIX_SHAPE)),
        WARNING_FACTORY(Diagnostics::KNOWN_UNSUPPORTED_TYPE, TYPESTR)))

TYPE_REWRITE_ENTRY(
    "nvcuda::wmma::layout_t",
    TYPE_CONDITIONAL_FACTORY(
        checkEnableJointMatrixForType(),
        TYPE_FACTORY(STR(MapNames::getClNamespace() +
                         "ext::oneapi::experimental::matrix::layout")),
        WARNING_FACTORY(Diagnostics::KNOWN_UNSUPPORTED_TYPE, TYPESTR)))

// clang-format on

TYPE_REWRITE_ENTRY("csrsv2Info_t",
                   TYPE_FACTORY(STR("std::shared_ptr<" +
                                    MapNames::getLibraryHelperNamespace() +
                                    "sparse::optimize_info>")))
TYPE_REWRITE_ENTRY("cusparseSolvePolicy_t", TYPE_FACTORY(STR("int")))
TYPE_REWRITE_ENTRY("cusparseAction_t",
                   TYPE_FACTORY(STR(MapNames::getLibraryHelperNamespace() +
                                    "sparse::conversion_scope")))

TYPE_REWRITE_ENTRY(
    "cooperative_groups::__v1::coalesced_group",
    TYPE_CONDITIONAL_FACTORY(
        isUseNonUniformGroupsForType(),
        TYPE_FACTORY(STR(MapNames::getClNamespace() +
                         "ext::oneapi::experimental::opportunistic_group")),
        WARNING_FACTORY(Diagnostics::TRY_EXPERIMENTAL_FEATURE, TYPESTR,
                        STR("--use-experimental-features=non-uniform-groups"))))
TYPE_REWRITE_ENTRY(
    "cooperative_groups::__v1::grid_group",
    TYPE_CONDITIONAL_FACTORY(
        isUseRootGroupForType(),
        TYPE_FACTORY(STR(MapNames::getExpNamespace() + "root_group")),
        TYPE_CONDITIONAL_FACTORY(
            isUseNdRangeBarrier(), TYPE_FACTORY(STR("")),
            WARNING_FACTORY(Diagnostics::TRY_EXPERIMENTAL_FEATURE, TYPESTR,
                            STR("--use-experimental-features=root-group")))))

TYPE_REWRITE_ENTRY(
    "cooperative_groups::__v1::thread_block_tile",
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
    "cudaGraph_t",
    TYPE_CONDITIONAL_FACTORY(
        checkEnableGraphForType(),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                         "experimental::command_graph_ptr")),
        WARNING_FACTORY(Diagnostics::TRY_EXPERIMENTAL_FEATURE, TYPESTR,
                        STR("--use-experimental-features=graph"))))

TYPE_REWRITE_ENTRY(
    "cudaGraphExec_t",
    TYPE_CONDITIONAL_FACTORY(
        checkEnableGraphForType(),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                         "experimental::command_graph_exec_ptr")),
        WARNING_FACTORY(Diagnostics::TRY_EXPERIMENTAL_FEATURE, TYPESTR,
                        STR("--use-experimental-features=graph"))))

TYPE_REWRITE_ENTRY(
    "cudaGraphNode_t",
    TYPE_CONDITIONAL_FACTORY(
        checkEnableGraphForType(),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                         "experimental::node_ptr")),
        WARNING_FACTORY(Diagnostics::TRY_EXPERIMENTAL_FEATURE, TYPESTR,
                        STR("--use-experimental-features=graph"))))

// Graphics Interop Handle
TYPE_REWRITE_ENTRY(
    "cudaGraphicsResource",
    TYPE_CONDITIONAL_FACTORY(
        checkEnableBindlessImagesForType(),
        TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                         "experimental::external_mem_wrapper")),
        WARNING_FACTORY(Diagnostics::TRY_EXPERIMENTAL_FEATURE, TYPESTR,
                        STR("--use-experimental-features=bindless_images"))))

TYPE_REWRITE_ENTRY(
    "cudaGraphicsResource_t",
    TYPE_CONDITIONAL_FACTORY(
        [](const TypeLoc) { return DpctGlobalInfo::useSYCLCompat(); },
        WARNING_FACTORY(Diagnostics::UNSUPPORT_SYCLCOMPAT,
                        STR("cudaGraphicsResource_t")),
        TYPE_CONDITIONAL_FACTORY(
            checkEnableBindlessImagesForType(),
            TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                             "experimental::external_mem_wrapper_ptr")),
            WARNING_FACTORY(
                Diagnostics::TRY_EXPERIMENTAL_FEATURE, TYPESTR,
                STR("--use-experimental-features=bindless_images")))))

TYPE_REWRITE_ENTRY(
    "CUgraphicsResource",
    TYPE_CONDITIONAL_FACTORY(
        [](const TypeLoc) { return DpctGlobalInfo::useSYCLCompat(); },
        WARNING_FACTORY(Diagnostics::UNSUPPORT_SYCLCOMPAT,
                        STR("CUgraphicsResource")),
        TYPE_CONDITIONAL_FACTORY(
            checkEnableBindlessImagesForType(),
            TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                             "experimental::external_mem_wrapper_ptr")),
            WARNING_FACTORY(
                Diagnostics::TRY_EXPERIMENTAL_FEATURE, TYPESTR,
                STR("--use-experimental-features=bindless_images")))))

// External Resource Handle Descs
TYPE_REWRITE_ENTRY(
    "cudaExternalMemory_t",
    TYPE_CONDITIONAL_FACTORY(
        [](const TypeLoc) { return DpctGlobalInfo::useSYCLCompat(); },
        WARNING_FACTORY(Diagnostics::UNSUPPORT_SYCLCOMPAT,
                        STR("cudaExternalMemory_t")),
        TYPE_CONDITIONAL_FACTORY(
            checkEnableBindlessImagesForType(),
            TYPE_FACTORY(STR(MapNames::getClNamespace() +
                             "ext::oneapi::experimental::external_mem")),
            WARNING_FACTORY(
                Diagnostics::TRY_EXPERIMENTAL_FEATURE, TYPESTR,
                STR("--use-experimental-features=bindless_images")))))

TYPE_REWRITE_ENTRY(
    "cudaExternalMemoryHandleDesc",
    TYPE_CONDITIONAL_FACTORY(
        [](const TypeLoc) { return DpctGlobalInfo::useSYCLCompat(); },
        WARNING_FACTORY(Diagnostics::UNSUPPORT_SYCLCOMPAT,
                        STR("cudaExternalMemoryHandleDesc")),
        TYPE_CONDITIONAL_FACTORY(
            checkEnableBindlessImagesForType(),
            TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                             "experimental::external_mem_handle_desc")),
            WARNING_FACTORY(
                Diagnostics::TRY_EXPERIMENTAL_FEATURE, TYPESTR,
                STR("--use-experimental-features=bindless_images")))))

TYPE_REWRITE_ENTRY(
    "cudaExternalMemoryMipmappedArrayDesc",
    TYPE_CONDITIONAL_FACTORY(
        [](const TypeLoc) { return DpctGlobalInfo::useSYCLCompat(); },
        WARNING_FACTORY(Diagnostics::UNSUPPORT_SYCLCOMPAT,
                        STR("cudaExternalMemoryMipmappedArrayDesc")),
        TYPE_CONDITIONAL_FACTORY(
            checkEnableBindlessImagesForType(),
            TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                             "experimental::external_mem_img_desc")),
            WARNING_FACTORY(
                Diagnostics::TRY_EXPERIMENTAL_FEATURE, TYPESTR,
                STR("--use-experimental-features=bindless_images")))))

TYPE_REWRITE_ENTRY(
    "cudaExternalMemoryBufferDesc",
    TYPE_CONDITIONAL_FACTORY(
        [](const TypeLoc) { return DpctGlobalInfo::useSYCLCompat(); },
        WARNING_FACTORY(Diagnostics::UNSUPPORT_SYCLCOMPAT,
                        STR("cudaExternalMemoryBufferDesc")),
        TYPE_CONDITIONAL_FACTORY(
            checkEnableBindlessImagesForType(),
            TYPE_FACTORY(STR(MapNames::getDpctNamespace() +
                             "experimental::external_mem_buf_desc")),
            WARNING_FACTORY(
                Diagnostics::TRY_EXPERIMENTAL_FEATURE, TYPESTR,
                STR("--use-experimental-features=bindless_images")))))
