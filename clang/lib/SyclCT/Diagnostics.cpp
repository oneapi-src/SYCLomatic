//===--- Diagnostics.cpp ---------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
#include "Diagnostics.h"
#include "clang/Basic/DiagnosticIDs.h"

namespace clang {
namespace syclct {

std::unordered_map<int, DiagnosticsMessage> DiagnosticIDTable;
std::unordered_map<int, DiagnosticsMessage> CommentIDTable;

#define DEF_NOTE(NAME, ID, MSG)                                                \
  DiagnosticsMessage eg_##NAME(DiagnosticIDTable, ID,                          \
                               clang::DiagnosticIDs::Note,                     \
                               "SYCLCT" #ID ": " MSG);

#define DEF_ERROR(NAME, ID, MSG)                                               \
  DiagnosticsMessage eg_##NAME(DiagnosticIDTable, ID,                          \
                               clang::DiagnosticIDs::Error,                    \
                               "SYCLCT" #ID ": " MSG);

#define DEF_WARNING(NAME, ID, MSG)                                             \
  DiagnosticsMessage wg_##NAME(DiagnosticIDTable, ID,                          \
                               clang::DiagnosticIDs::Warning,                  \
                               "SYCLCT" #ID ": " MSG);

#define DEF_COMMENT(NAME, ID, MSG)                                             \
  DiagnosticsMessage cg_##NAME(CommentIDTable, ID, clang::DiagnosticIDs::Note, \
                               "SYCLCT" #ID ": " MSG);

#include "Diagnostics.inc"

#undef DEF_NOTE
#undef DEF_ERROR
#undef DEF_WARNING
#undef DEF_COMMENT

} // namespace syclct
} // namespace clang