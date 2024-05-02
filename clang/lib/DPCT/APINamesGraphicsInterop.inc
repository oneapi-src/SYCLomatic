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
    CONDITIONAL_FACTORY_ENTRY(
        checkIsCallExprOnly(),
        WARNING_FACTORY_ENTRY("cudaGraphicsResourceSetMapFlags",
            TOSTRING_FACTORY_ENTRY("cudaGraphicsResourceSetMapFlags", LITERAL("")),
            Diagnostics::FUNC_CALL_REMOVED,
            std::string("cudaGraphicsResourceSetMapFlags"), getRemovedAPIWarningMessage("cudaGraphicsResourceSetMapFlags")),
        WARNING_FACTORY_ENTRY("cudaGraphicsResourceSetMapFlags",
            TOSTRING_FACTORY_ENTRY("cudaGraphicsResourceSetMapFlags", LITERAL("0")),
            Diagnostics::FUNC_CALL_REMOVED_0,
            std::string("cudaGraphicsResourceSetMapFlags"), getRemovedAPIWarningMessage("cudaGraphicsResourceSetMapFlags"))),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphicsResourceSetMapFlags", Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphicsResourceSetMapFlags"),
                            ARG("--use-experimental-features=bindless_images")))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtBindlessImages,
    CALL_FACTORY_ENTRY("cudaGraphicsMapResources",
                       CALL(MapNames::getDpctNamespace() +
                                "experimental::map_resources",
                            ARG(0), ARG(1), QUEUESTR)),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphicsMapResources", Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphicsMapResources"),
                            ARG("--use-experimental-features=bindless_images"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtBindlessImages,
    CALL_FACTORY_ENTRY("cudaGraphicsUnmapResources",
                       CALL(MapNames::getDpctNamespace() +
                                "experimental::unmap_resources",
                            ARG(0), ARG(1), QUEUESTR)),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphicsUnmapResources", Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphicsUnmapResources"),
                            ARG("--use-experimental-features=bindless_images"))))

ASSIGNABLE_FACTORY(CONDITIONAL_FACTORY_ENTRY(
    UseExtBindlessImages,
    CALL_FACTORY_ENTRY("cudaGraphicsUnregisterResource",
                       CALL(MapNames::getDpctNamespace() +
                                "experimental::unregister_resource",
                            ARG(0))),
    UNSUPPORT_FACTORY_ENTRY("cudaGraphicsUnregisterResource", Diagnostics::TRY_EXPERIMENTAL_FEATURE,
                            ARG("cudaGraphicsUnregisterResource"),
                            ARG("--use-experimental-features=bindless_images"))))

/*
CONDITIONAL_FACTORY_ENTRY(
    UseExtBindlessImages,
    ASSIGNABLE_FACTORY(ASSIGN_FACTORY_ENTRY(
        "cudaMallocMipmappedArray", DEREF(makeCallArgCreatorWithCall(0)),
        NEW(MapNames::getDpctNamespace() + "experimental::image_mem_wrapper",
            DEREF(1), ARG(2),
            LITERAL(MapNames::getClNamespace() +
                    "ext::oneapi::experimental::image_type::mipmap"),
            ARG(3)))),
    REMOVE_API_FACTORY_ENTRY_WITH_MSG("cudaMallocMipmappedArray",
                                      MipmapNeedBindlessImage))
*/

// clang-format on
