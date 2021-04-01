//===---APINamesTexture.inc ---------------------------------------------===//
//
// Copyright (C) 2018 - 2021 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifdef _WIN32

ASSIGN_FACTORY_ENTRY("cuModuleLoad", DEREF(0),
                     CALL("LoadLibraryA",
                          ARG("PlaceHolder/*Fix the module file name manually*/")))

ASSIGN_FACTORY_ENTRY("cuModuleLoadData", DEREF(0),
                     CALL("LoadLibraryA",
                          ARG("PlaceHolder/*Fix the module file name manually*/")))

ASSIGN_FACTORY_ENTRY("cuModuleGetFunction", DEREF(0),
                     CALL("(dpct::kernel_functor)GetProcAddress",
                          ARG(1), EXTENDSTR(2, "_wrapper")))

CALL_FACTORY_ENTRY("cuModuleUnload", CALL("FreeLibrary", ARG(0)))

#else

ASSIGN_FACTORY_ENTRY("cuModuleLoad", DEREF(0),
                     CALL("dlopen",
                          ARG("PlaceHolder/*Fix the module file name manually*/"),
                          ARG("RTLD_LAZY")))

ASSIGN_FACTORY_ENTRY("cuModuleLoadData", DEREF(0),
                     CALL("dlopen",
                          ARG("PlaceHolder/*Fix the module file name manually*/"),
                          ARG("RTLD_LAZY")))

ASSIGN_FACTORY_ENTRY("cuModuleGetFunction", DEREF(0),
                     CALL("dlsym", ARG(1), EXTENDSTR(2, "_wrapper")))

CALL_FACTORY_ENTRY("cuModuleUnload", CALL("dlclose", ARG(0)))

#endif