//===--------------- DiagnosticsCMakeScriptMigration.inc ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FORMAT: {NAME, ID, MESSAGE}
DEF_COMMENT(CMAKE_NOT_SUPPORT_WARNING, 3000, "Migration of syntax \"{0}\" is not supported. You may need to adjust the code.")
DEF_COMMENT(CMAKE_CONFIG_FILE_WARNING, 3001, "\"{0}\" is used to generate an output file by performing transformation on input file. You may need to update the name of the input and output file, as well as migrate the input file if the input file contains CUDA syntax.")
DEF_COMMENT(WARNING_FOR_SYNTAX_REMOVED, 3002, "\"{0}\" is removed because it is not supported for use alongside option \"-fsycl\" of SYCL compiler. You may need to check the build options.")
