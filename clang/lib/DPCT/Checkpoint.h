//===--------------- Checkpoint.h -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_CHECKPOINT_H
#define DPCT_CHECKPOINT_H

#include "SignalProcess.h"
#include <setjmp.h>

#if defined(__linux__)
#define JMP_BUF sigjmp_buf
#define SETJMP(x) sigsetjmp(x, 1)
#define LONGJMP siglongjmp

#else
#define JMP_BUF jmp_buf
#define SETJMP(x) _setjmp(x)
#define LONGJMP longjmp
#endif

extern bool EnableErrorRecover;

extern JMP_BUF CPFileEnter;
extern JMP_BUF CPFileASTMaterEnter;
extern JMP_BUF CPRepPostprocessEnter;
extern JMP_BUF CPFormatCodeEnter;
extern JMP_BUF CPApplyReps;

extern int CheckPointStage;
extern int CheckPointStageCore;

extern int FatalErrorCnt;
extern int FatalErrorASTCnt;
extern bool CurFileMeetErr;

// During the migration, there are several big steps:
//   step1: Parse input file and build the AST tree.
//   step2: Call the AST consumer, and do AST Match
//   step3: Post processing the replacement generated.
//   step4: Format the migrated code if necessary.
//   step5: Write out the replacement to generate the migration result.
// now step1, step2, step3, step4 will have checkpoint, if fatal error
// happen, it will try to skip the current file and do further migration.
enum {
  CHECKPOINT_UNKNOWN = 0, /*No checkpoint available*/
  CHECKPOINT_PROCESSING_FILE = 1,
  CHECKPOINT_PROCESSING_FILE_ASTMATCHER = 2,
  CHECKPOINT_PROCESSING_REPLACEMENT_POSTPROCESS = 3,
  CHECKPOINT_FORMATTING_CODE = 4,
  CHECKPOINT_WRITE_OUT = 5,
};

class AstCPStageMaintainer {
public:
  ~AstCPStageMaintainer() { CheckPointStage = CHECKPOINT_UNKNOWN; }
};

#define CHECKPOINT_ASTMATCHER_RUN_ENTRY()                                      \
  AstCPStageMaintainer ACPSM;                                                  \
  CHECKPOINT_ASTMATCHER_RUN_ENTRY_INTERNAL()

#define CHECKPOINT_ASTMATCHER_RUN_ENTRY_INTERNAL()                             \
  do {                                                                         \
    if (EnableErrorRecover) {                                                  \
      int SetJmpRet = SETJMP(CPFileASTMaterEnter);                             \
      CheckPointStage = CHECKPOINT_PROCESSING_FILE_ASTMATCHER;                 \
      if (SetJmpRet != 0) {                                                    \
        return;                                                                \
      }                                                                        \
    }                                                                          \
  } while (0);

#define CHECKPOINT_ReplacementPostProcess_ENTRY(Ret)                           \
  do {                                                                         \
    if (EnableErrorRecover) {                                                  \
      Ret = SETJMP(CPRepPostprocessEnter);                                     \
      CheckPointStage = CHECKPOINT_PROCESSING_REPLACEMENT_POSTPROCESS;         \
    }                                                                          \
  } while (0);
#define CHECKPOINT_ReplacementPostProcess_EXIT()                               \
  do {                                                                         \
    if (EnableErrorRecover)                                                    \
      CheckPointStage = CHECKPOINT_UNKNOWN;                                    \
  } while (0);

#define CHECKPOINT_FORMATTING_CODE_ENTRY(Ret)                                  \
  do {                                                                         \
    if (EnableErrorRecover) {                                                  \
      Ret = SETJMP(CPFormatCodeEnter);                                         \
      CheckPointStage = CHECKPOINT_FORMATTING_CODE;                            \
    }                                                                          \
  } while (0);
#define CHECKPOINT_FORMATTING_CODE_EXIT()                                      \
  do {                                                                         \
    if (EnableErrorRecover)                                                    \
      CheckPointStage = CHECKPOINT_UNKNOWN;                                    \
  } while (0);

#endif // DPCT_CHECKPOINT_H
