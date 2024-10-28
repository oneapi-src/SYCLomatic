//===------------- DiagnosticsPythonBuildScriptMigration.inc --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FORMAT: {NAME, ID, MESSAGE}
DEF_COMMENT(WARNING_FOR_SYNTAX_REMOVED, 4000,
            "\"{0}\" is removed because it is not supported. You may need to "
            "adjust the code.")
