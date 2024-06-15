//===--------------- DiagnosticsCMakeScriptMigration.inc ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FORMAT: {NAME, ID, MESSAGE}
DEF_COMMENT(CMAKE_NOT_SUPPORT_WARNING, 3000, "Migration of syntax \"{0}\" is not supported. You may need to adjust the code.")
DEF_COMMENT(CMAKE_CONFIG_FILE_WARNING, 3001, "\"{0}\" is used to copy an input file to an output file while performing transformations of the input file content. You may need to copy the input file from CUDA code project to the migrated code project to make sure the command configure_file run successfully or consider removing this command.")
