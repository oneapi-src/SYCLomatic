//===--------------- APINamesStream.inc -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

ASSIGNABLE_FACTORY(
    FEATURE_REQUEST_FACTORY(HelperFeatureEnum::device_ext,
    MEMBER_CALL_FACTORY_ENTRY(
        "cuStreamDestroy_v2", CALL(MapNames::getDpctNamespace() + "get_current_device"),
        false, "destroy_queue", ARG(0))))

CONDITIONAL_FACTORY_ENTRY(
    checkIsCallExprOnly(),
    WARNING_FACTORY_ENTRY("cuStreamAttachMemAsync",
        TOSTRING_FACTORY_ENTRY("cuStreamAttachMemAsync", LITERAL("")),
        Diagnostics::FUNC_CALL_REMOVED,
        std::string("cuStreamAttachMemAsync"), getRemovedAPIWarningMessage("cuStreamAttachMemAsync")),
    WARNING_FACTORY_ENTRY("cuStreamAttachMemAsync",
        TOSTRING_FACTORY_ENTRY("cuStreamAttachMemAsync", LITERAL("0")),
        Diagnostics::FUNC_CALL_REMOVED_0,
        std::string("cuStreamAttachMemAsync"), getRemovedAPIWarningMessage("cuStreamAttachMemAsync")))

ASSIGNABLE_FACTORY(HEADER_INSERT_FACTORY(
    HeaderType::HT_Future,
    CALL_FACTORY_ENTRY("cuStreamAddCallback",
                       CALL("std::async",
                            LAMBDA(true, false,
                                   MEMBER_CALL(ARG(0), true, "wait"),
                                   CALL(ARG(1), ARG(0), ARG("0"), ARG(2)))))))
