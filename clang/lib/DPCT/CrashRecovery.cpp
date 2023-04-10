//===--------------- CrashRecovery.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CrashRecovery.h"

#include "Error.h"
#include "Utility.h"

#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"

#include <csetjmp>
#include <csignal>
#include <string>

void PrintReportOnFault(const std::string &FaultMsg);

namespace {

std::string getFaultMsg(const int Signo) {
  std::string Msg;
  llvm::raw_string_ostream OS(Msg);
  OS << "\nError: meet signal:";
#define SIGNAL(x)                                                              \
  case x:                                                                      \
    OS << #x;                                                                  \
    break
  switch (Signo) {
    SIGNAL(SIGABRT);
    SIGNAL(SIGSEGV);
    SIGNAL(SIGILL);
    SIGNAL(SIGINT);
    SIGNAL(SIGTERM);
    SIGNAL(SIGFPE);
  default:
    OS << "UNKNOWN EXCEPTION";
  }
#undef SIGNAL
  OS << " dpct tries to write analysis reports and terminates...\n";
  return OS.str();
}
} // namespace

namespace clang {
namespace tooling {
void setCrashGuardFunc(bool (*Func)(llvm::function_ref<void()>, std::string));
}
namespace dpct {

class CrashGuard {
  std::jmp_buf RecoverPointer;
  llvm::function_ref<void(void)> Application;
  std::string FaultMsg;
  CrashGuard *Before;

  static constexpr int Recover = -1;

  static CrashGuard *Current;

  void HandleCrash() {
    PrintReportOnFault(FaultMsg);
    std::longjmp(RecoverPointer, Recover);
  }

public:
  CrashGuard(llvm::function_ref<void(void)> App, std::string Fault)
      : Application(App), FaultMsg(std::move(Fault)) {
    std::signal(SIGSEGV, signalHandle);
    Before = Current;
    Current = this;
  }
  ~CrashGuard() { Current = Before; }

  bool run() {
    try {
      if (setjmp(RecoverPointer) == Recover)
        return false;
      Application();
      return true;
    } catch (...) {
      HandleCrash();
      return false;
    }
  }

  static void signalHandle(int Signal) {
    switch (Signal) {
    case SIGSEGV:
      if (Current)
        Current->HandleCrash();
      LLVM_FALLTHROUGH;
    default:
      break;
    }
    PrintReportOnFault(getFaultMsg(Signal));
    dpctExit(MigrationError);
  }
};

CrashGuard *CrashGuard::Current = nullptr;

bool runWithCrashGuard(llvm::function_ref<void()> Func, std::string FaultMsg) {
  CrashGuard guard(Func, std::move(FaultMsg));
  return guard.run();
}

void initCrashRecovery() {
  clang::tooling::setCrashGuardFunc(runWithCrashGuard);
#define SIGNAL(x) std::signal(x, CrashGuard::signalHandle)
  SIGNAL(SIGABRT);
  SIGNAL(SIGSEGV);
  SIGNAL(SIGILL);
  SIGNAL(SIGINT);
  SIGNAL(SIGTERM);
  SIGNAL(SIGFPE);
#undef SIGNAL
}

} // namespace dpct
} // namespace clang