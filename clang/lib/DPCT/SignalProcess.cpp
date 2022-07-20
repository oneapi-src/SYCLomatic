//===--------------- SignalProcess.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SignalProcess.h"
#include "Checkpoint.h"
#include "Error.h"
#include "Utility.h"

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
extern uint64_t CurFileSigErrCnt;
void recoverCheckpoint(int Signo) {
  CurFileSigErrCnt++;
  if (EnableErrorRecover && Signo == SIGSEGV) {
    if (CheckPointStage == CHECKPOINT_PROCESSING_FILE) {
      std::string FaultMsg = "Error: dpct internal error. Current file "
                             "skipped. Migration continues.\n";
      PrintReportOnFault(FaultMsg);
      if (!CurFileMeetErr) {
        FatalErrorCnt++;
        CurFileMeetErr = true;
      }
      LONGJMP(CPFileEnter, 1);
    } else if (CheckPointStage == CHECKPOINT_PROCESSING_FILE_ASTMATCHER) {
      std::string FaultMsg =
          "Error: dpct internal error. Migration rule causing the error "
          "skipped. Migration continues.\n";
      PrintReportOnFault(FaultMsg);
      if (!CurFileMeetErr) {
        FatalErrorCnt++;
        CurFileMeetErr = true;
      }
      LONGJMP(CPFileASTMaterEnter, 1);
    } else if (CheckPointStage ==
               CHECKPOINT_PROCESSING_REPLACEMENT_POSTPROCESS) {
      std::string FaultMsg = "Error: dpct internal error. dpct tries to "
                             "recover and write the migration result.\n";
      PrintReportOnFault(FaultMsg);
      if (!CurFileMeetErr) {
        FatalErrorCnt++;
        CurFileMeetErr = true;
      }
      LONGJMP(CPRepPostprocessEnter, 1);
    } else if (CheckPointStage == CHECKPOINT_FORMATTING_CODE) {
      std::string FaultMsg = "Error: dpct internal error. Formatting of the "
                             "code skipped. Migration continues.\n";
      PrintReportOnFault(FaultMsg);
      if (!CurFileMeetErr) {
        FatalErrorCnt++;
        CurFileMeetErr = true;
      }
      LONGJMP(CPFormatCodeEnter, 1);
    } else if (CheckPointStageCore == CHECKPOINT_WRITE_OUT) {
      std::string FaultMsg = "Error: dpct internal error. dpct tries to "
                             "recover and write the migration result.\n";
      PrintReportOnFault(FaultMsg);
      if (!CurFileMeetErr) {
        FatalErrorCnt++;
        CurFileMeetErr = true;
      }
      LONGJMP(CPApplyReps, 1);
    }
  }
}

#if defined(_WIN64)
void FaultHandler(int Signo) {
  std::string FaultMsg = "\nError:" + SigDescription(Signo) +
                         " dpct tries to write "
                         "analysis reports and terminates...\n";
  PrintReportOnFault(FaultMsg);
  dpctExit(MigrationError);
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

static LONG CALLBACK ExceptionHandler(PEXCEPTION_POINTERS ExceptionInfo) {
  switch (ExceptionInfo->ExceptionRecord->ExceptionCode) {
  case STATUS_ACCESS_VIOLATION:
    recoverCheckpoint(SIGSEGV);
  default:
    return EXCEPTION_CONTINUE_SEARCH;
  }
  llvm_unreachable("Handled the crash, should have longjmp'ed out of here");
}

// Here set both exception handler and signal handler.
// The reason uses exception handler: in windows there is no function have
//    siglongjmp/sigprocmask.
// The reason uses signal hander: keep same output msg in both win and linux.
static void SetHandler() {
  ::AddVectoredExceptionHandler(1, ExceptionHandler);
  SetSignalHandler(FaultHandler);
}

#endif

#if defined(__linux__)
static void FaultHandler(int Signo, siginfo_t *Info, void *Extra) {
  recoverCheckpoint(Signo);
  std::string FaultMsg = "\nError: meet signal:" + SigDescription(Signo) +
                         " dpct tries to write "
                         "analysis reports and terminates...\n";
  PrintReportOnFault(FaultMsg);
  dpctExit(MigrationError);
}

static void SetHandler(void (*handler)(int, siginfo_t *, void *)) {
  struct sigaction action;
  action.sa_flags = SA_SIGINFO;
  action.sa_sigaction = handler;

  if (sigaction(SIGSEGV, &action, NULL) == -1) {
    llvm::errs() << "SIGSEGV: sigaction installation failure\n";
    dpctExit(-1);
  }

  if (sigaction(SIGABRT, &action, NULL) == -1) {
    llvm::errs() << "SIGABRT: sigaction installation failure\n";
    dpctExit(-1);
  }

  if (sigaction(SIGTERM, &action, NULL) == -1) {
    llvm::errs() << "SIGTERM: sigaction installation failure\n";
    dpctExit(-1);
  }

  if (sigaction(SIGFPE, &action, NULL) == -1) {
    llvm::errs() << "SIGFPE: sigaction installation failure\n";
    dpctExit(-1);
  }

  if (sigaction(SIGSTKFLT, &action, NULL) == -1) {
    llvm::errs() << "SIGSTKFLT: sigaction installation failure\n";
    dpctExit(-1);
  }
  if (sigaction(SIGPIPE, &action, NULL) == -1) {
    llvm::errs() << "SIGPIPE: sigaction installation failure\n";
    dpctExit(-1);
  }
}
#endif

void InstallSignalHandle(void) {
#if defined(__linux__)
  SetHandler(FaultHandler);
#elif defined(_WIN64)
  SetHandler();
#endif
}
