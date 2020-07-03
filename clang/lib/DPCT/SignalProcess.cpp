//===--- SignalProcess.cpp ------------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2020 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#include "Checkpoint.h"
#include "SignalProcess.h"
#include "SaveNewFiles.h"

#include "clang/Basic/LangOptions.h"

extern void PrintReportOnFault(std::string &FaultMsg);
#if defined(__linux__) || defined(_WIN64)
#include <signal.h>
#endif

#if defined(__linux__) || defined(_WIN64)
static const std::string SigDescription(const int &Signo) {
  switch (Signo) {
  case SIGABRT:
    return "SIGABRT"; // Abnormal termination
  case SIGSEGV:
    return "SIGSEGV"; // Illegal storage access
  case SIGILL:
    return "SIGILL"; // Illegal instruction
  case SIGINT:
    return "SIGINT"; // CTRL+C signal
  case SIGTERM:
    return "SIGTERM"; // Termination request
  case SIGFPE:
    return "SIGFPE"; // Floating-point error
  default:
    return "UNKNOWN EXCEPTION";
  }
}
#endif
extern unsigned long CurFileSigErrCnt;
void recoverCheckpoint(int Signo){
  CurFileSigErrCnt++;
  if( EnableErrorRecover && Signo == SIGSEGV) {
      if(CheckPointStage==CHECKPOINT_PROCESSING_FILE) {
        std::string FaultMsg = "dpct error: segmentation fault."
                             " Intel(R) DPC++ Compatibility Tool trys to recover by"
                             "skipping current file.\n";
        PrintReportOnFault(FaultMsg);
        if(!CurFileMeetErr) {
          FatalErrorCnt++;
          CurFileMeetErr=true;
        }
        LONGJMP(CPFileEnter, 1);
      } else if(CheckPointStage==CHECKPOINT_PROCESSING_FILE_ASTMATCHER) {
        std::string FaultMsg = "dpct error: segmentation fault."
                               " Intel(R) DPC++ Compatibility Tool tries to "
                               "recover by skipping the migration rule causing error.\n";
        PrintReportOnFault(FaultMsg);
        if(!CurFileMeetErr) {
          FatalErrorCnt++;
          CurFileMeetErr=true;
        }
        LONGJMP(CPFileASTMaterEnter, 1);
      } else if(CheckPointStage==CHECKPOINT_PROCESSING_REPLACEMENT_POSTPROCESS) {
        std::string FaultMsg = "dpct error: segmentation fault."
                               " Intel(R) DPC++ Compatibility Tool tries to "
                               "recover and write the migration result.\n";
        PrintReportOnFault(FaultMsg);
        if(!CurFileMeetErr) {
          FatalErrorCnt++;
          CurFileMeetErr=true;
        }
        LONGJMP(CPRepPostprocessEnter, 1);
      } else if(CheckPointStageCore==CHECKPOINT_WRITE_OUT) {
          std::string FaultMsg = "dpct error: segmentation fault."
                                 " Intel(R) DPC++ Compatibility Tool tries to "
                                 "recover and write the migration result.\n";
          PrintReportOnFault(FaultMsg);
          if(!CurFileMeetErr) {
            FatalErrorCnt++;
            CurFileMeetErr=true;
          }
          LONGJMP(CPApplyReps, 1);
      }
  }
}

#if defined(_WIN64)
void FaultHandler(int Signo) {
  std::string FaultMsg = "\ndpct error:" + SigDescription(Signo) +
                         " Intel(R) DPC++ Compatibility Tool tries to write "
                         "analysis reports and terminates...\n";
  PrintReportOnFault(FaultMsg);
  exit(MigrationError);
}

static void SetSignalHandler(void (*Handler)(int)) {
  signal(SIGABRT, Handler);
  signal(SIGSEGV, Handler);
  signal(SIGILL, Handler);
  signal(SIGINT, Handler);
  signal(SIGTERM, Handler);
  signal(SIGFPE, Handler);
}

#include "llvm/Support/Windows/WindowsSupport.h"

static LONG CALLBACK ExceptionHandler(PEXCEPTION_POINTERS ExceptionInfo)
{
  switch (ExceptionInfo->ExceptionRecord->ExceptionCode)
  {
    case STATUS_ACCESS_VIOLATION:
        recoverCheckpoint(SIGSEGV);
    default:
       return EXCEPTION_CONTINUE_EXECUTION;
  }
  llvm_unreachable("Handled the crash, should have longjmp'ed out of here");
}

// Here set both exception handler and signal handler.
// The reason use exception handler: in windows there is no function have
//    siglongjmp/sigprocmask.
// The reason use singal hander: keep same output msg in both win and linux.
static void SetHandler(){
  ::AddVectoredExceptionHandler(1, ExceptionHandler);
  SetSignalHandler(FaultHandler);
}

#endif

#if defined(__linux__)
static void FaultHandler(int Signo, siginfo_t *Info, void *Extra) {
  recoverCheckpoint(Signo);
  std::string FaultMsg = "\ndpct error: meet signal:" + SigDescription(Signo) +
                         " Intel(R) DPC++ Compatibility Tool trys to write "
                         "analysis reports and terminates...\n";
  PrintReportOnFault(FaultMsg);
  exit(MigrationError);
}

static void SetHandler(void (*handler)(int, siginfo_t *, void *)) {
  struct sigaction action;
  action.sa_flags = SA_SIGINFO;
  action.sa_sigaction = handler;

  if (sigaction(SIGSEGV, &action, NULL) == -1) {
    llvm::errs() << "SIGSEGV: sigaction installation failure\n";
    exit(-1);
  }

  if (sigaction(SIGABRT, &action, NULL) == -1) {
    llvm::errs() << "SIGABRT: sigaction installation failure\n";
    exit(-1);
  }

  if (sigaction(SIGTERM, &action, NULL) == -1) {
    llvm::errs() << "SIGTERM: sigaction installation failure\n";
    exit(-1);
  }

  if (sigaction(SIGFPE, &action, NULL) == -1) {
    llvm::errs() << "SIGFPE: sigaction installation failure\n";
    exit(-1);
  }

  if (sigaction(SIGSTKFLT, &action, NULL) == -1) {
    llvm::errs() << "SIGSTKFLT: sigaction installation failure\n";
    exit(-1);
  }
  if (sigaction(SIGPIPE, &action, NULL) == -1) {
    llvm::errs() << "SIGPIPE: sigaction installation failure\n";
    exit(-1);
  }
}
#endif

void InstallSignalHandle(void) {
#if  defined(__linux__)
  SetHandler(FaultHandler);
#elif defined(_WIN64);
  SetHandler();
#endif
}
