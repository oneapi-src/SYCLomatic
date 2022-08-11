//===--------------- APINamesMisc.inc -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
ASSIGNABLE_FACTORY(
    FEATURE_REQUEST_FACTORY(
        HelperFeatureEnum::Device_pointer_attributes,
        CALL_FACTORY_ENTRY(
                "cudaPointerGetAttributes",
                CALL(MapNames::getDpctNamespace() + "get_pointer_attributes", DEREF(0), ARG(1))
            )
        )
)