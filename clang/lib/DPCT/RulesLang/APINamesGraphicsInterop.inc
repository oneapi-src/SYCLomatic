//===--------------------- APINamesGraphicsInterop.inc --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off
CONDITIONAL_FACTORY_ENTRY(
    UseExtBindlessImages,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
        "cudaGraphicsD3D11RegisterResource", DEREF(0),
        NEW(MapNames::getDpctNamespace() + "experimental::external_mem_wrapper",
            ARG(1), ARG(2)))),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphicsD3D11RegisterResource", Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphicsD3D11RegisterResource"),
                            ARG("--use-experimental-features=bindless_images")))

CONDITIONAL_FACTORY_ENTRY(
    checkIsCallExprOnly(),
    WARNING_FACTORY_ENTRY(
        "cudaGraphicsResourceSetMapFlags",
        TOSTRING_FACTORY_ENTRY("cudaGraphicsResourceSetMapFlags", LITERAL("")),
        Diagnostics::FUNC_CALL_REMOVED,
        std::string("cudaGraphicsResourceSetMapFlags"),
        getRemovedAPIWarningMessage("cudaGraphicsResourceSetMapFlags")),
    WARNING_FACTORY_ENTRY(
        "cudaGraphicsResourceSetMapFlags",
        TOSTRING_FACTORY_ENTRY("cudaGraphicsResourceSetMapFlags", LITERAL("0")),
        Diagnostics::FUNC_CALL_REMOVED_0,
        std::string("cudaGraphicsResourceSetMapFlags"),
        getRemovedAPIWarningMessage("cudaGraphicsResourceSetMapFlags")))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtBindlessImages,
    CONDITIONAL_FACTORY_ENTRY(
        makeCheckNot(CheckArgIsDefaultCudaStream(2)),
        CALL_FACTORY_ENTRY("cudaGraphicsMapResources",
                           CALL(MapNames::getDpctNamespace() +
                                    "experimental::map_resources",
                                ARG(0), ARG(1), ARG(2))),
        CALL_FACTORY_ENTRY("cudaGraphicsMapResources",
                           CALL(MapNames::getDpctNamespace() +
                                    "experimental::map_resources",
                                ARG(0), ARG(1)))),
    UNSUPPORT_FACTORY_ENTRY(
        "cudaGraphicsMapResources", Diagnostics::TRY_EXPERIMENTAL_FEATURE,
        ARG("cudaGraphicsMapResources"),
        ARG("--use-experimental-features=bindless_images"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtBindlessImages,
    MEMBER_CALL_FACTORY_ENTRY(
        "cudaGraphicsResourceGetMappedPointer",
        ARG(2), true, "get_mapped_pointer", ARG(0), ARG(1)),
    UNSUPPORT_FACTORY_ENTRY(
        "cudaGraphicsResourceGetMappedPointer", Diagnostics::TRY_EXPERIMENTAL_FEATURE,
        ARG("cudaGraphicsResourceGetMappedPointer"),
        ARG("--use-experimental-features=bindless_images"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtBindlessImages,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
        "cudaGraphicsResourceGetMappedMipmappedArray", DEREF(0),
        MEMBER_CALL(ARG(1), true, "get_mapped_mipmapped_array"))),
    UNSUPPORT_FACTORY_ENTRY(
        "cudaGraphicsResourceGetMappedMipmappedArray", Diagnostics::TRY_EXPERIMENTAL_FEATURE,
        ARG("cudaGraphicsResourceGetMappedMipmappedArray"),
        ARG("--use-experimental-features=bindless_images"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtBindlessImages,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
        "cudaGraphicsSubResourceGetMappedArray", DEREF(0),
        MEMBER_CALL(ARG(1), true, "get_sub_resource_mapped_array", ARG(2), ARG(3)))),
    UNSUPPORT_FACTORY_ENTRY(
        "cudaGraphicsSubResourceGetMappedArray", Diagnostics::TRY_EXPERIMENTAL_FEATURE,
        ARG("cudaGraphicsSubResourceGetMappedArray"),
        ARG("--use-experimental-features=bindless_images"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtBindlessImages,
    CONDITIONAL_FACTORY_ENTRY(
        makeCheckNot(CheckArgIsDefaultCudaStream(2)),
        CALL_FACTORY_ENTRY("cudaGraphicsUnmapResources",
                           CALL(MapNames::getDpctNamespace() +
                                    "experimental::unmap_resources",
                                ARG(0), ARG(1), ARG(2))),
        CALL_FACTORY_ENTRY("cudaGraphicsUnmapResources",
                           CALL(MapNames::getDpctNamespace() +
                                    "experimental::unmap_resources",
                                ARG(0), ARG(1)))),
    UNSUPPORT_FACTORY_ENTRY(
        "cudaGraphicsUnmapResources", Diagnostics::TRY_EXPERIMENTAL_FEATURE,
        ARG("cudaGraphicsUnmapResources"),
        ARG("--use-experimental-features=bindless_images"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtBindlessImages,
    DELETER_FACTORY_ENTRY("cudaGraphicsUnregisterResource", ARG(0)),
    UNSUPPORT_FACTORY_ENTRY(
        "cudaGraphicsUnregisterResource", Diagnostics::TRY_EXPERIMENTAL_FEATURE,
        ARG("cudaGraphicsUnregisterResource"),
        ARG("--use-experimental-features=bindless_images"))))

// External Resource APIs
CONDITIONAL_FACTORY_ENTRY(
    UseExtBindlessImages,
    ASSIGNABLE_FACTORY(CALL_FACTORY_ENTRY("cudaImportExternalMemory",
                           CALL(MapNames::getDpctNamespace() +
                                    "experimental::import_external_memory",
                                ARG(0), ARG(1)))),
    UNSUPPORT_FACTORY_ENTRY("cudaImportExternalMemory", Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaImportExternalMemory"),
                            ARG("--use-experimental-features=bindless_images")))

CONDITIONAL_FACTORY_ENTRY(
    UseExtBindlessImages,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
        "cudaExternalMemoryGetMappedMipmappedArray", DEREF(0),
        NEW(MapNames::getDpctNamespace() + "experimental::image_mem_wrapper",
            ARG(1), ARG(2)))),
    UNSUPPORT_FACTORY_ENTRY("cudaExternalMemoryGetMappedMipmappedArray", Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaExternalMemoryGetMappedMipmappedArray"),
                            ARG("--use-experimental-features=bindless_images")))

CONDITIONAL_FACTORY_ENTRY(
    UseExtBindlessImages,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
        "cudaExternalMemoryGetMappedBuffer", DEREF(0),
        CALL(MapNames::getClNamespace() + "ext::oneapi::experimental::map_external_linear_memory",
            ARG(1), MEMBER_CALL(ARG(2), true, "get_res_size"), MEMBER_CALL(ARG(2), true, "get_mem_offset"), QUEUESTR))),
    UNSUPPORT_FACTORY_ENTRY("cudaExternalMemoryGetMappedBuffer", Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaExternalMemoryGetMappedBuffer"),
                            ARG("--use-experimental-features=bindless_images")))

CONDITIONAL_FACTORY_ENTRY(
    UseExtBindlessImages,
    CALL_FACTORY_ENTRY("cudaDestroyExternalMemory",
                        CALL(MapNames::getClNamespace() +
                                 "ext::oneapi::experimental::release_external_memory",
                            ARG(0), QUEUESTR)),
    UNSUPPORT_FACTORY_ENTRY("cudaDestroyExternalMemory", Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaDestroyExternalMemory"),
                            ARG("--use-experimental-features=bindless_images")))
// clang-format on
