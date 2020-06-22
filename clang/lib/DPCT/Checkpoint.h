//===--- Checkpoint.h -------------------------------------*- C++ -*---===//
//
// Copyright (C) 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------------------------------------------------===//

#ifndef DPCT_CHECKPOINT_H
#define DPCT_CHECKPOINT_H

#include "SignalProcess.h"
#include <setjmp.h>

#if defined(__linux__)
#define JMP_BUF   sigjmp_buf
#define SETJMP(x)       sigsetjmp(x, 1)
#define LONGJMP      siglongjmp

#else
#define JMP_BUF   jmp_buf
#define SETJMP(x)       setjmp(x)
#define LONGJMP      longjmp
#endif

extern bool EnableErrorRecover;

extern JMP_BUF CPFileEnter;
extern JMP_BUF CPFileASTMaterEnter;
extern JMP_BUF CPRepPostprocessEnter;
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
//   step4: Write out the replacement to generate the migration result.
// now step1, step2, step3 will have checkpoint, if fatal error
// happen, it will try to skip the current file and do further migration.
enum {
 CHECKPOINT_UNKNOWN=0, /*No checkpoint available*/
 CHECKPOINT_PROCESSING_FILE=1,
 CHECKPOINT_PROCESSING_FILE_ASTMATCHER=2,
 CHECKPOINT_PROCESSING_REPLACEMENT_POSTPROCESS=3,
 CHECKPOINT_WRITE_OUT=4,
};

class AstCPStageMaintainer{
  public:
    ~AstCPStageMaintainer(){
      CheckPointStage=CHECKPOINT_PROCESSING_FILE;
    }
};

#define CHECKPOINT_ASTMATCHER_RUN_ENTRY() AstCPStageMaintainer ACPSM; \
	CHECKPOINT_ASTMATCHER_RUN_ENTRY_INTERNAL()

#define CHECKPOINT_ASTMATCHER_RUN_ENTRY_INTERNAL()  do{\
  if(EnableErrorRecover){\
    int SetJmpRet=SETJMP(CPFileASTMaterEnter);\
    CheckPointStage = CHECKPOINT_PROCESSING_FILE_ASTMATCHER;\
    if(SetJmpRet != 0) {\
      return;\
    }\
  }\
}while(0);
#define CHECKPOINT_ASTMATCHER_RUN_EXIT()  do{\
  if(EnableErrorRecover)\
    CheckPointStage = CHECKPOINT_PROCESSING_FILE;\
}while(0);

#define CHECKPOINT_ReplacementPostProcess_ENTRY(Ret)  do{\
  if(EnableErrorRecover){\
    Ret=SETJMP(CPRepPostprocessEnter);\
    CheckPointStage = CHECKPOINT_PROCESSING_REPLACEMENT_POSTPROCESS;\
   }\
}while(0);
#define CHECKPOINT_ReplacementPostProcess_EXIT()  do{\
  if(EnableErrorRecover)\
    CheckPointStage = CHECKPOINT_UNKNOWN;\
}while(0);



#endif // DPCT_CHECKPOINT_H
