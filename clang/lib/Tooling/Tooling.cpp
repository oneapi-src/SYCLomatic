//===- Tooling.cpp - Running clang standalone tools -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements functions to run clang tools standalone instead
//  of running them as a plugin.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Tooling.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LLVM.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#ifdef SYCLomatic_CUSTOMIZATION
#include <setjmp.h>
#endif // SYCLomatic_CUSTOMIZATION

#define DEBUG_TYPE "clang-tooling"

using namespace clang;
using namespace tooling;

#ifdef SYCLomatic_CUSTOMIZATION
namespace clang {
namespace tooling {
static PrintType MsgPrintHandle = nullptr;
static std::string SDKIncludePath = "";
static std::set<std::string> *FileSetInCompiationDBPtr = nullptr;
static std::vector<std::pair<std::string, std::vector<std::string>>>
    *CompileTargetsMapPtr = nullptr;
static StringRef InRoot;
static StringRef OutRoot;
static FileProcessType FileProcessHandle = nullptr;
static std::set<std::string> *ReProcessFilePtr = nullptr;
static std::function<unsigned int()> GetRunRoundPtr;
static std::set<std::string> *ModuleFiles = nullptr;
static std::function<bool(const std::string &, bool)> IsExcludePathPtr;
extern std::string VcxprojFilePath;

void SetPrintHandle(PrintType Handle) {
  MsgPrintHandle = Handle;
}

void SetFileSetInCompiationDB(std::set<std::string> &FileSetInCompiationDB) {
  FileSetInCompiationDBPtr = &FileSetInCompiationDB;
}

void SetCompileTargetsMap(
    std::vector<std::pair<std::string, std::vector<std::string>>>
        &CompileTargetsMap) {
  CompileTargetsMapPtr = &CompileTargetsMap;
}

void SetFileProcessHandle(StringRef In, StringRef Out, FileProcessType Handle) {
  FileProcessHandle = Handle;
  InRoot = In;
  OutRoot = Out;
}

void CollectFileFromDB(std::string FileName) {
  if (FileSetInCompiationDBPtr != nullptr) {
    (*FileSetInCompiationDBPtr).insert(FileName);
  }
}

void CollectCompileTarget(std::string Target, std::vector<std::string> Options) {
  if (CompileTargetsMapPtr != nullptr) {
    CompileTargetsMapPtr->push_back(std::make_pair(Target, Options));
  }
}

void DoPrintHandle(const std::string &Msg, bool IsPrintOnNormal) {
  if (MsgPrintHandle != nullptr) {
    (*MsgPrintHandle)(Msg, IsPrintOnNormal);
  }
}

void DoFileProcessHandle(std::vector<std::string> &FilesNotProcessed) {
  if (FileProcessHandle != nullptr) {
    (*FileProcessHandle)(InRoot, OutRoot, FilesNotProcessed);
  }
}

bool isFileProcessAllSet() {
  return FileProcessHandle != nullptr;
}

void SetReProcessFile(std::set<std::string> &ReProcessFile){
  ReProcessFilePtr = &ReProcessFile;
}

void SetGetRunRound(std::function<unsigned int()> Func){
  GetRunRoundPtr = Func;
}

unsigned int DoGetRunRound(){
  if(GetRunRoundPtr){
    return GetRunRoundPtr();
  }
  return 0;
}

std::set<std::string> GetReProcessFile(){
  if(ReProcessFilePtr){
    return *ReProcessFilePtr;
  }
  return std::set<std::string>();
}

void SetSDKIncludePath(const std::string &Path) { SDKIncludePath = Path; }

static llvm::raw_ostream *OSTerm = nullptr;
void SetDiagnosticOutput(llvm::raw_ostream &OStream) { OSTerm = &OStream; }
void SetModuleFiles(std::set<std::string> &MF) { ModuleFiles = &MF; }
llvm::raw_ostream &DiagnosticsOS() {
  if (OSTerm != nullptr) {
    return *OSTerm;
  } else {
    return llvm::errs();
  }
}

std::string ClangToolOutputMessage = "";

std::string getRealFilePath(std::string File, clang::FileManager *FM){
#ifdef _WIN64
  std::string RealFilePath;
  llvm::SmallString<512> FilePathAbs(File);
  llvm::sys::path::native(FilePathAbs);
  llvm::sys::path::remove_dots(FilePathAbs, true);
  RealFilePath = FilePathAbs.str().str();
  auto FE = FM->getFile(File);
  std::error_code EC = FE.getError();
  if(!(bool)EC && !FE.get()->tryGetRealPathName().empty()) {
    RealFilePath = FE.get()->tryGetRealPathName().str();
  }
  return RealFilePath;
#else
  return File;
#endif
}

void SetIsExcludePathHandler(std::function<bool(const std::string &, bool)> Func){
  IsExcludePathPtr = Func;
}

bool isExcludePath(const std::string &Path, bool IsRelative) {
  if(IsExcludePathPtr) {
    return IsExcludePathPtr(Path, IsRelative);
  } else {
    return false;
  }
}

enum {
  ProcessingFilesCrash = -1,
};
typedef bool (*CrashGuardFunc)(llvm::function_ref<void()>, std::string);
bool processFilesWithCrashGuardDefault(llvm::function_ref<void()> Func,
                                       std::string) {
  Func();
  return true;
}

CrashGuardFunc ProcessFilesWithCrashGuardPtr =
    processFilesWithCrashGuardDefault;
void setCrashGuardFunc(CrashGuardFunc Func) {
  ProcessFilesWithCrashGuardPtr = Func;
}
static int processFilesWithCrashGuard(ClangTool *Tool, llvm::StringRef File,
                                      bool &ProcessingFailed, bool &FileSkipped,
                                      int &StaticSymbol, ToolAction *Action) {
  int Ret;
  if (ProcessFilesWithCrashGuardPtr(
          [&]() {
            Ret = Tool->processFiles(File, ProcessingFailed, FileSkipped,
                                     StaticSymbol, Action);
          },
          "Error: dpct internal error. Current file \"" + File.str() +
              "\" skipped. Migration continues.\n"))
    return Ret;
  return ProcessingFilesCrash;
}
bool SpecifyLanguageInOption = false;
void emitDefaultLanguageWarningIfNecessary(const std::string &FileName,
                                           bool SpecifyLanguageInOption) {
  if (!SpecifyLanguageInOption &&
      llvm::sys::path::extension(FileName) != ".cu" &&
      llvm::sys::path::extension(FileName) != ".cuh") {
    llvm::outs() << "NOTE: " << FileName
                 << " is treated as a CUDA file by default. Use the "
                    "--extra-arg=-xc++ option to treat "
                 << FileName << " as a C++ file if needed."
                 << "\n";
  }
}
} // namespace tooling
} // namespace 
bool StopOnParseErrTooling=false;
std::string InRootTooling;

// filename, error#
//  error: high32:processed sig error, low32: parse error
std::map<std::string, uint64_t> ErrorCnt;
uint64_t CurFileSigErrCnt=0;
uint64_t CurFileParseErrCnt=0;

#endif // SYCLomatic_CUSTOMIZATION

ToolAction::~ToolAction() = default;

FrontendActionFactory::~FrontendActionFactory() = default;

// FIXME: This file contains structural duplication with other parts of the
// code that sets up a compiler to run tools on it, and we should refactor
// it to be based on the same framework.

/// Builds a clang driver initialized for running clang tools.
static driver::Driver *
newDriver(DiagnosticsEngine *Diagnostics, const char *BinaryName,
          IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS) {
  driver::Driver *CompilerDriver =
      new driver::Driver(BinaryName, llvm::sys::getDefaultTargetTriple(),
                         *Diagnostics, "clang LLVM compiler", std::move(VFS));
  CompilerDriver->setTitle("clang_based_tool");
  return CompilerDriver;
}

/// Decide whether extra compiler frontend commands can be ignored.
static bool ignoreExtraCC1Commands(const driver::Compilation *Compilation) {
  const driver::JobList &Jobs = Compilation->getJobs();
  const driver::ActionList &Actions = Compilation->getActions();

  bool OffloadCompilation = false;

  // Jobs and Actions look very different depending on whether the Clang tool
  // injected -fsyntax-only or not. Try to handle both cases here.

  for (const auto &Job : Jobs)
    if (StringRef(Job.getExecutable()) == "clang-offload-bundler")
      OffloadCompilation = true;

  if (Jobs.size() > 1) {
    for (auto *A : Actions){
      // On MacOSX real actions may end up being wrapped in BindArchAction
      if (isa<driver::BindArchAction>(A))
        A = *A->input_begin();
      if (isa<driver::OffloadAction>(A)) {
        // Offload compilation has 2 top-level actions, one (at the front) is
        // the original host compilation and the other is offload action
        // composed of at least one device compilation. For such case, general
        // tooling will consider host-compilation only. For tooling on device
        // compilation, device compilation only option, such as
        // `--cuda-device-only`, needs specifying.
        assert(Actions.size() > 1);
        assert(
            isa<driver::CompileJobAction>(Actions.front()) ||
            // On MacOSX real actions may end up being wrapped in
            // BindArchAction.
            (isa<driver::BindArchAction>(Actions.front()) &&
             isa<driver::CompileJobAction>(*Actions.front()->input_begin())));
        OffloadCompilation = true;
        break;
      }
    }
  }

  return OffloadCompilation;
}

namespace clang {
namespace tooling {

const llvm::opt::ArgStringList *
getCC1Arguments(DiagnosticsEngine *Diagnostics,
                driver::Compilation *Compilation) {
  const driver::JobList &Jobs = Compilation->getJobs();

  auto IsCC1Command = [](const driver::Command &Cmd) {
    return StringRef(Cmd.getCreator().getName()) == "clang";
  };

  auto IsSrcFile = [](const driver::InputInfo &II) {
    return isSrcFile(II.getType());
  };

  llvm::SmallVector<const driver::Command *, 1> CC1Jobs;
  for (const driver::Command &Job : Jobs)
    if (IsCC1Command(Job) && llvm::all_of(Job.getInputInfos(), IsSrcFile))
      CC1Jobs.push_back(&Job);

  // If there are no jobs for source files, try checking again for a single job
  // with any file type. This accepts a preprocessed file as input.
  if (CC1Jobs.empty())
    for (const driver::Command &Job : Jobs)
      if (IsCC1Command(Job))
        CC1Jobs.push_back(&Job);

  if (CC1Jobs.empty() ||
      (CC1Jobs.size() > 1 && !ignoreExtraCC1Commands(Compilation))) {
    SmallString<256> error_msg;
    llvm::raw_svector_ostream error_stream(error_msg);
    Jobs.Print(error_stream, "; ", true);
    Diagnostics->Report(diag::err_fe_expected_compiler_job)
        << error_stream.str();
    return nullptr;
  }

  return &CC1Jobs[0]->getArguments();
}

/// Returns a clang build invocation initialized from the CC1 flags.
CompilerInvocation *newInvocation(DiagnosticsEngine *Diagnostics,
                                  ArrayRef<const char *> CC1Args,
                                  const char *const BinaryName) {
  assert(!CC1Args.empty() && "Must at least contain the program name!");
  CompilerInvocation *Invocation = new CompilerInvocation;
  CompilerInvocation::CreateFromArgs(*Invocation, CC1Args, *Diagnostics,
                                     BinaryName);
  Invocation->getFrontendOpts().DisableFree = false;
  Invocation->getCodeGenOpts().DisableFree = false;
  return Invocation;
}

bool runToolOnCode(std::unique_ptr<FrontendAction> ToolAction,
                   const Twine &Code, const Twine &FileName,
                   std::shared_ptr<PCHContainerOperations> PCHContainerOps) {
  return runToolOnCodeWithArgs(std::move(ToolAction), Code,
                               std::vector<std::string>(), FileName,
                               "clang-tool", std::move(PCHContainerOps));
}

} // namespace tooling
} // namespace clang

static std::vector<std::string>
getSyntaxOnlyToolArgs(const Twine &ToolName,
                      const std::vector<std::string> &ExtraArgs,
                      StringRef FileName) {
  std::vector<std::string> Args;
  Args.push_back(ToolName.str());
  Args.push_back("-fsyntax-only");
  Args.insert(Args.end(), ExtraArgs.begin(), ExtraArgs.end());
  Args.push_back(FileName.str());
  return Args;
}

namespace clang {
namespace tooling {

bool runToolOnCodeWithArgs(
    std::unique_ptr<FrontendAction> ToolAction, const Twine &Code,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS,
    const std::vector<std::string> &Args, const Twine &FileName,
    const Twine &ToolName,
    std::shared_ptr<PCHContainerOperations> PCHContainerOps) {
  SmallString<16> FileNameStorage;
  StringRef FileNameRef = FileName.toNullTerminatedStringRef(FileNameStorage);

  llvm::IntrusiveRefCntPtr<FileManager> Files(
      new FileManager(FileSystemOptions(), VFS));
  ArgumentsAdjuster Adjuster = getClangStripDependencyFileAdjuster();
  ToolInvocation Invocation(
      getSyntaxOnlyToolArgs(ToolName, Adjuster(Args, FileNameRef), FileNameRef),
      std::move(ToolAction), Files.get(), std::move(PCHContainerOps));
  return Invocation.run();
}

bool runToolOnCodeWithArgs(
    std::unique_ptr<FrontendAction> ToolAction, const Twine &Code,
    const std::vector<std::string> &Args, const Twine &FileName,
    const Twine &ToolName,
    std::shared_ptr<PCHContainerOperations> PCHContainerOps,
    const FileContentMappings &VirtualMappedFiles) {
  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFileSystem(
      new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);
  OverlayFileSystem->pushOverlay(InMemoryFileSystem);

  SmallString<1024> CodeStorage;
  InMemoryFileSystem->addFile(FileName, 0,
                              llvm::MemoryBuffer::getMemBuffer(
                                  Code.toNullTerminatedStringRef(CodeStorage)));

  for (auto &FilenameWithContent : VirtualMappedFiles) {
    InMemoryFileSystem->addFile(
        FilenameWithContent.first, 0,
        llvm::MemoryBuffer::getMemBuffer(FilenameWithContent.second));
  }

  return runToolOnCodeWithArgs(std::move(ToolAction), Code, OverlayFileSystem,
                               Args, FileName, ToolName);
}

llvm::Expected<std::string> getAbsolutePath(llvm::vfs::FileSystem &FS,
                                            StringRef File) {
  StringRef RelativePath(File);
  // FIXME: Should '.\\' be accepted on Win32?
  if (RelativePath.startswith("./")) {
    RelativePath = RelativePath.substr(strlen("./"));
  }

  SmallString<1024> AbsolutePath = RelativePath;
  if (auto EC = FS.makeAbsolute(AbsolutePath))
    return llvm::errorCodeToError(EC);
  llvm::sys::path::native(AbsolutePath);
  return std::string(AbsolutePath.str());
}

std::string getAbsolutePath(StringRef File) {
  return llvm::cantFail(getAbsolutePath(*llvm::vfs::getRealFileSystem(), File));
}

void addTargetAndModeForProgramName(std::vector<std::string> &CommandLine,
                                    StringRef InvokedAs) {
  if (CommandLine.empty() || InvokedAs.empty())
    return;
  const auto &Table = driver::getDriverOptTable();
  // --target=X
  StringRef TargetOPT =
      Table.getOption(driver::options::OPT_target).getPrefixedName();
  // -target X
  StringRef TargetOPTLegacy =
      Table.getOption(driver::options::OPT_target_legacy_spelling)
          .getPrefixedName();
  // --driver-mode=X
  StringRef DriverModeOPT =
      Table.getOption(driver::options::OPT_driver_mode).getPrefixedName();
  auto TargetMode =
      driver::ToolChain::getTargetAndModeFromProgramName(InvokedAs);
  // No need to search for target args if we don't have a target/mode to insert.
  bool ShouldAddTarget = TargetMode.TargetIsValid;
  bool ShouldAddMode = TargetMode.DriverMode != nullptr;
  // Skip CommandLine[0].
  for (auto Token = ++CommandLine.begin(); Token != CommandLine.end();
       ++Token) {
    StringRef TokenRef(*Token);
    ShouldAddTarget = ShouldAddTarget && !TokenRef.startswith(TargetOPT) &&
                      !TokenRef.equals(TargetOPTLegacy);
    ShouldAddMode = ShouldAddMode && !TokenRef.startswith(DriverModeOPT);
  }
  if (ShouldAddMode) {
    CommandLine.insert(++CommandLine.begin(), TargetMode.DriverMode);
  }
  if (ShouldAddTarget) {
    CommandLine.insert(++CommandLine.begin(),
                       (TargetOPT + TargetMode.TargetPrefix).str());
  }
}

void addExpandedResponseFiles(std::vector<std::string> &CommandLine,
                              llvm::StringRef WorkingDir,
                              llvm::cl::TokenizerCallback Tokenizer,
                              llvm::vfs::FileSystem &FS) {
  bool SeenRSPFile = false;
  llvm::SmallVector<const char *, 20> Argv;
  Argv.reserve(CommandLine.size());
  for (auto &Arg : CommandLine) {
    Argv.push_back(Arg.c_str());
    if (!Arg.empty())
      SeenRSPFile |= Arg.front() == '@';
  }
  if (!SeenRSPFile)
    return;
  llvm::BumpPtrAllocator Alloc;
  llvm::cl::ExpansionContext ECtx(Alloc, Tokenizer);
  llvm::Error Err =
      ECtx.setVFS(&FS).setCurrentDir(WorkingDir).expandResponseFiles(Argv);
  if (Err)
    llvm::errs() << Err;
  // Don't assign directly, Argv aliases CommandLine.
  std::vector<std::string> ExpandedArgv(Argv.begin(), Argv.end());
  CommandLine = std::move(ExpandedArgv);
}

} // namespace tooling
} // namespace clang

namespace {

class SingleFrontendActionFactory : public FrontendActionFactory {
  std::unique_ptr<FrontendAction> Action;

public:
  SingleFrontendActionFactory(std::unique_ptr<FrontendAction> Action)
      : Action(std::move(Action)) {}

  std::unique_ptr<FrontendAction> create() override {
    return std::move(Action);
  }
};

} // namespace

ToolInvocation::ToolInvocation(
    std::vector<std::string> CommandLine, ToolAction *Action,
    FileManager *Files, std::shared_ptr<PCHContainerOperations> PCHContainerOps)
    : CommandLine(std::move(CommandLine)), Action(Action), OwnsAction(false),
      Files(Files), PCHContainerOps(std::move(PCHContainerOps)) {}

ToolInvocation::ToolInvocation(
    std::vector<std::string> CommandLine,
    std::unique_ptr<FrontendAction> FAction, FileManager *Files,
    std::shared_ptr<PCHContainerOperations> PCHContainerOps)
    : CommandLine(std::move(CommandLine)),
      Action(new SingleFrontendActionFactory(std::move(FAction))),
      OwnsAction(true), Files(Files),
      PCHContainerOps(std::move(PCHContainerOps)) {}

ToolInvocation::~ToolInvocation() {
  if (OwnsAction)
    delete Action;
#ifdef SYCLomatic_CUSTOMIZATION
  if(DiagnosticPrinter)
    delete DiagnosticPrinter;
#endif // SYCLomatic_CUSTOMIZATION
}

bool ToolInvocation::run() {
  llvm::opt::ArgStringList Argv;
  for (const std::string &Str : CommandLine)
    Argv.push_back(Str.c_str());
  const char *const BinaryName = Argv[0];
  IntrusiveRefCntPtr<DiagnosticOptions> ParsedDiagOpts;
  DiagnosticOptions *DiagOpts = this->DiagOpts;
  if (!DiagOpts) {
    ParsedDiagOpts = CreateAndPopulateDiagOpts(Argv);
    DiagOpts = &*ParsedDiagOpts;
  }
#ifdef SYCLomatic_CUSTOMIZATION
  DiagnosticPrinter =new TextDiagnosticPrinter(DiagnosticsOS(), &*DiagOpts);
  DiagConsumer = DiagnosticPrinter;
#endif // SYCLomatic_CUSTOMIZATION
  TextDiagnosticPrinter DiagnosticPrinter(llvm::errs(), DiagOpts);
  IntrusiveRefCntPtr<DiagnosticsEngine> Diagnostics =
      CompilerInstance::createDiagnostics(
          &*DiagOpts, DiagConsumer ? DiagConsumer : &DiagnosticPrinter, false);

  // Although `Diagnostics` are used only for command-line parsing, the custom
  // `DiagConsumer` might expect a `SourceManager` to be present.
  SourceManager SrcMgr(*Diagnostics, *Files);
  Diagnostics->setSourceManager(&SrcMgr);

  // We already have a cc1, just create an invocation.
  if (CommandLine.size() >= 2 && CommandLine[1] == "-cc1") {
    ArrayRef<const char *> CC1Args = ArrayRef(Argv).drop_front();
    std::unique_ptr<CompilerInvocation> Invocation(
        newInvocation(&*Diagnostics, CC1Args, BinaryName));
    if (Diagnostics->hasErrorOccurred())
      return false;
    return Action->runInvocation(std::move(Invocation), Files,
                                 std::move(PCHContainerOps), DiagConsumer);
  }

  const std::unique_ptr<driver::Driver> Driver(
      newDriver(&*Diagnostics, BinaryName, &Files->getVirtualFileSystem()));
  // The "input file not found" diagnostics from the driver are useful.
  // The driver is only aware of the VFS working directory, but some clients
  // change this at the FileManager level instead.
  // In this case the checks have false positives, so skip them.
  if (!Files->getFileSystemOpts().WorkingDir.empty())
    Driver->setCheckInputsExist(false);
  const std::unique_ptr<driver::Compilation> Compilation(
      Driver->BuildCompilation(llvm::ArrayRef(Argv)));
  if (!Compilation)
    return false;
  const llvm::opt::ArgStringList *const CC1Args = getCC1Arguments(
      &*Diagnostics, Compilation.get());
  if (!CC1Args)
    return false;
  std::unique_ptr<CompilerInvocation> Invocation(
      newInvocation(&*Diagnostics, *CC1Args, BinaryName));
  return runInvocation(BinaryName, Compilation.get(), std::move(Invocation),
                       std::move(PCHContainerOps));
}

bool ToolInvocation::runInvocation(
    const char *BinaryName, driver::Compilation *Compilation,
    std::shared_ptr<CompilerInvocation> Invocation,
    std::shared_ptr<PCHContainerOperations> PCHContainerOps) {
  // Show the invocation, with -v.
  if (Invocation->getHeaderSearchOpts().Verbose) {
    llvm::errs() << "clang Invocation:\n";
    Compilation->getJobs().Print(llvm::errs(), "\n", true);
    llvm::errs() << "\n";
  }

  return Action->runInvocation(std::move(Invocation), Files,
                               std::move(PCHContainerOps), DiagConsumer);
}

bool FrontendActionFactory::runInvocation(
    std::shared_ptr<CompilerInvocation> Invocation, FileManager *Files,
    std::shared_ptr<PCHContainerOperations> PCHContainerOps,
    DiagnosticConsumer *DiagConsumer) {
  // Create a compiler instance to handle the actual work.
  CompilerInstance Compiler(std::move(PCHContainerOps));
  Compiler.setInvocation(std::move(Invocation));
  Compiler.setFileManager(Files);

  // The FrontendAction can have lifetime requirements for Compiler or its
  // members, and we need to ensure it's deleted earlier than Compiler. So we
  // pass it to an std::unique_ptr declared after the Compiler variable.
  std::unique_ptr<FrontendAction> ScopedToolAction(create());

  // Create the compiler's actual diagnostics engine.
  Compiler.createDiagnostics(DiagConsumer, /*ShouldOwnClient=*/false);
  if (!Compiler.hasDiagnostics())
    return false;

  Compiler.createSourceManager(*Files);

  const bool Success = Compiler.ExecuteAction(*ScopedToolAction);
#ifdef SYCLomatic_CUSTOMIZATION
  CurFileParseErrCnt = DiagConsumer -> getNumErrors();
#endif // SYCLomatic_CUSTOMIZATION
  Files->clearStatCache();
  return Success;
}

ClangTool::ClangTool(const CompilationDatabase &Compilations,
                     ArrayRef<std::string> SourcePaths,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS,
                     IntrusiveRefCntPtr<FileManager> Files)
    : Compilations(Compilations), SourcePaths(SourcePaths),
      PCHContainerOps(std::move(PCHContainerOps)),
      OverlayFileSystem(new llvm::vfs::OverlayFileSystem(std::move(BaseFS))),
      InMemoryFileSystem(new llvm::vfs::InMemoryFileSystem),
      Files(Files ? Files
                  : new FileManager(FileSystemOptions(), OverlayFileSystem)) {
  OverlayFileSystem->pushOverlay(InMemoryFileSystem);
  appendArgumentsAdjuster(getClangStripOutputAdjuster());
  appendArgumentsAdjuster(getClangSyntaxOnlyAdjuster());
  appendArgumentsAdjuster(getClangStripDependencyFileAdjuster());
  if (Files)
    Files->setVirtualFileSystem(OverlayFileSystem);
}

ClangTool::~ClangTool() = default;

void ClangTool::mapVirtualFile(StringRef FilePath, StringRef Content) {
  MappedFileContents.push_back(std::make_pair(FilePath, Content));
}

void ClangTool::appendArgumentsAdjuster(ArgumentsAdjuster Adjuster) {
  ArgsAdjuster = combineAdjusters(std::move(ArgsAdjuster), std::move(Adjuster));
}

void ClangTool::clearArgumentsAdjusters() {
  ArgsAdjuster = nullptr;
}

static void injectResourceDir(CommandLineArguments &Args, const char *Argv0,
                              void *MainAddr) {
  // Allow users to override the resource dir.
  for (StringRef Arg : Args)
    if (Arg.startswith("-resource-dir"))
      return;

  // If there's no override in place add our resource dir.
  Args = getInsertArgumentAdjuster(
      ("-resource-dir=" + CompilerInvocation::GetResourcesPath(Argv0, MainAddr))
          .c_str())(Args, "");
}

#ifdef SYCLomatic_CUSTOMIZATION
// Try to parse and migrate \pFile, and return process result with
// \pProcessingFailed, \pFileSkipped , \pStaticSymbol and its return value.
// if return value is -1, means current input file \p File is not processed,
// if return value < -1, report return value to upper caller,
// other values are ignored.
int ClangTool::processFiles(llvm::StringRef File,bool &ProcessingFailed,
                     bool &FileSkipped, int &StaticSymbol, ToolAction *Action) {
    //clear error# counter
    CurFileParseErrCnt=0;
    CurFileSigErrCnt=0;
    // Currently implementations of CompilationDatabase::getCompileCommands can
    // change the state of the file system (e.g.  prepare generated headers), so
    // this method needs to run right before we invoke the tool, as the next
    // file may require a different (incompatible) state of the file system.
    //
    // FIXME: Make the compilation database interface more explicit about the
    // requirements to the order of invocation of its members.
    std::vector<CompileCommand> CompileCommandsForFile =
        Compilations.getCompileCommands(File);
    if (CompileCommandsForFile.empty()) {
      llvm::errs() << "Skipping " << File
                   << ". Compile command for this file not found in "
                      "compile_commands.json.\n";
      FileSkipped = true;
      return -1;
    }
    for (CompileCommand &CompileCommand : CompileCommandsForFile) {
      // FIXME: chdir is thread hostile; on the other hand, creating the same
      // behavior as chdir is complex: chdir resolves the path once, thus
      // guaranteeing that all subsequent relative path operations work
      // on the same path the original chdir resulted in. This makes a
      // difference for example on network filesystems, where symlinks might be
      // switched during runtime of the tool. Fixing this depends on having a
      // file system abstraction that allows openat() style interactions.
      if (OverlayFileSystem->setCurrentWorkingDirectory(
              CompileCommand.Directory))
      {
        ClangToolOutputMessage = CompileCommand.Directory;
        return -29 /*MigrationErrorCannotAccessDirInDatabase*/;
      }

      // Now fill the in-memory VFS with the relative file mappings so it will
      // have the correct relative paths. We never remove mappings but that
      // should be fine.
      if (SeenWorkingDirectories.insert(CompileCommand.Directory).second)
        for (const auto &MappedFile : MappedFileContents)
          if (!llvm::sys::path::is_absolute(MappedFile.first))
            InMemoryFileSystem->addFile(
                MappedFile.first, 0,
                llvm::MemoryBuffer::getMemBuffer(MappedFile.second));

      std::vector<std::string> CommandLine = CompileCommand.CommandLine;

      /// TODO: When supporting Driver API migration, dpct needs to migrate the
      ///       source file(s) which is built by --cubin/--ptx option as module
      ///       file(s).

      // To remove --cubin/--ptx option from command line is to
      // avoid parsing error msgs like: "error: unknown argument: '-ptx'" or
      // "error: unknown argument: '-cubin'".
      bool IsModuleFile = false;
      for (size_t Index = 0; Index < CommandLine.size(); Index++) {
        if (CommandLine[Index] == "-ptx" || CommandLine[Index] == "--ptx" ||
            CommandLine[Index] == "-cubin" || CommandLine[Index] == "--cubin") {
          CommandLine.erase(CommandLine.begin() + Index--);
          IsModuleFile = true;
        }
      }
      if(IsModuleFile)
        ModuleFiles->insert(getRealFilePath(File.str(), Files.get()));

      std::string Filename = CompileCommand.Filename;
      if(!llvm::sys::path::is_absolute(Filename)) {
          // To convert the relative path to absolute path.
          llvm::SmallString<128> AbsPath(Filename);
          llvm::sys::fs::make_absolute(AbsPath);
          llvm::sys::path::remove_dots(AbsPath, /*remove_dot_dot=*/true);
          Filename = std::string(AbsPath.str());
      }

      std::vector<std::string> Options;
      // The first place is used to store the directory where the compile command runs
      Options.push_back(CompileCommand.Directory.c_str());
      for (size_t Index = 0; Index < CommandLine.size(); Index++)
        Options.push_back(CommandLine[Index]);

      // Skip parsing Linker command
      if (CompileCommand.Filename == "LinkerEntry") {
        static size_t Idx = 0;
        auto NewKey = CompileCommand.Filename + std::to_string(Idx++);
        CollectCompileTarget(NewKey, Options);
        continue;
      }
      CollectCompileTarget(CompileCommand.Filename.c_str(), Options);

      if (!llvm::sys::path::has_filename(CompileCommand.Filename)) {
        std::string CommandField = "";
        for (size_t Index = 0; Index < CommandLine.size(); Index++) {
          CommandField = CommandField + CommandLine[Index] + " ";
        }
        if (!CommandField.empty())
          CommandField.pop_back();

        ClangToolOutputMessage = std::string(35, ' ') + CompilationDatabaseDir +
                                 "/compile_commands.json:\n" +
                                 std::string(35, ' ') +
                                 "  -  \"command\" field: \"" + CommandField +
                                 "\", \"file\" field: \"\"";
        return -30 /*MigrationErrorInconsistentFileInDatabase*/;
      }

      StringRef BaseNameRef = llvm::sys::path::filename(Filename);
      std::string BaseNameStr = BaseNameRef.str();
      std::string ItemNameStr = "";
      bool Matched = false;
      for (size_t Index = 0; Index < CommandLine.size(); Index++) {
        if (!llvm::sys::path::has_filename(CommandLine[Index]))
          return -1;
        StringRef ItemNameRef = llvm::sys::path::filename(CommandLine[Index]);
        ItemNameStr = ItemNameRef.str();
        if (ItemNameStr == BaseNameStr) {
          Matched = true;
          // Try to convert the path of input source file into absolute path, as
          // relative path has the potential risk to change the working
          // directory of in-memory VFS, which may result in an unexpected
          // behavior.
          CommandLine[Index] = Filename;
          break;
        }
      }

      if (!Matched) {
        ClangToolOutputMessage = std::string(35, ' ') + CompilationDatabaseDir +
                                 "/compile_commands.json:\n" +
                                 std::string(35, ' ') +
                                 "  -  \"command\" field: \"" + ItemNameStr +
                                 "\", \"file\" field: \"" +
                                 BaseNameStr + "\"";
        return -30 /*MigrationErrorInconsistentFileInDatabase*/;
      }

      for (size_t index = 0; index < SDKIncludePath.size(); index++) {
        if (SDKIncludePath[index] == '\\') {
          SDKIncludePath[index] = '/';
        }
      }
      ArgumentsAdjuster CudaArgsAdjuster{ArgsAdjuster};
#ifdef _WIN32
      // In Microsoft Visual Studio Project, CUDA file in <None> is not part of
      // the build project, So if "*.cu" files is found in <None> node, just
      // skip it and give a warning message.
      if ((!CommandLine.empty() && CommandLine[0] == "None") &&
          llvm::sys::path::extension(File) == ".cu") {
        const std::string Msg =
            "warning: " + File.str() +
            " was found in <None> node in " + VcxprojFilePath + " and skipped; to "
            "migrate specify CUDA* Item Type for this file in project and try "
            "again.\n";
        DoPrintHandle(Msg, false);
        return -1;
      }

      if ((!CommandLine.empty() && CommandLine[0] == "CudaCompile") ||
          (!CommandLine.empty() && CommandLine[0] == "CustomBuild" &&
           llvm::sys::path::extension(File)==".cu")) {
        emitDefaultLanguageWarningIfNecessary(File.str(),
                                              SpecifyLanguageInOption);
        CudaArgsAdjuster = combineAdjusters(
            std::move(CudaArgsAdjuster),
            getInsertArgumentAdjuster("cuda", ArgumentInsertPosition::BEGIN));
        CudaArgsAdjuster = combineAdjusters(
            std::move(CudaArgsAdjuster),
            getInsertArgumentAdjuster("-x", ArgumentInsertPosition::BEGIN));
      }
#else
      if (!CommandLine.empty() && CommandLine[0].size() >= 4 &&
          CommandLine[0].substr(CommandLine[0].size() - 4) == "nvcc") {
        emitDefaultLanguageWarningIfNecessary(File.str(),
                                              SpecifyLanguageInOption);
        CudaArgsAdjuster = combineAdjusters(
            std::move(CudaArgsAdjuster),
            getInsertArgumentAdjuster("cuda", ArgumentInsertPosition::BEGIN));
        CudaArgsAdjuster = combineAdjusters(
            std::move(CudaArgsAdjuster),
            getInsertArgumentAdjuster("-x", ArgumentInsertPosition::BEGIN));
      }
#endif
      CommandLine = getInsertArgumentAdjuster(
          (std::string("-I") + SDKIncludePath).c_str(),
          ArgumentInsertPosition::END)(CommandLine, "");
      if (CudaArgsAdjuster)
        CommandLine = CudaArgsAdjuster(CommandLine, CompileCommand.Filename);

      assert(!CommandLine.empty());

      // Add the resource dir based on the binary of this tool. argv[0] in the
      // compilation database may refer to a different compiler and we want to
      // pick up the very same standard library that compiler is using. The
      // builtin headers in the resource dir need to match the exact clang
      // version the tool is using.
      // FIXME: On linux, GetMainExecutable is independent of the value of the
      // first argument, thus allowing ClangTool and runToolOnCode to just
      // pass in made-up names here. Make sure this works on other platforms.
      injectResourceDir(CommandLine, "clang_tool", &StaticSymbol);

      // FIXME: We need a callback mechanism for the tool writer to output a
      // customized message for each file.
      LLVM_DEBUG({ llvm::dbgs() << "Processing: " << File << ".\n"; });
      ToolInvocation Invocation(std::move(CommandLine), Action, Files.get(),
                                PCHContainerOps);
      Invocation.setDiagnosticConsumer(DiagConsumer);

      if (!Invocation.run()) {
        // FIXME: Diagnostics should be used instead.
        if (PrintErrorMessage && StopOnParseErrTooling) {
          std::string ErrMsg="Did not process 1 file(s) in -in-root folder \""
                   + InRootTooling + "\":\n"
                   "    " + File.str() + ": " + std::to_string(CurFileParseErrCnt)
                   + " parsing error(s)\n";
          llvm::errs() << ErrMsg;
        }
        ProcessingFailed = true;
        if(StopOnParseErrTooling)
            break;
      }
    }
    //collect the errror counter info.
    ErrorCnt[File.str()] =(CurFileSigErrCnt<<32) | CurFileParseErrCnt;
    return 0;
}
#endif // SYCLomatic_CUSTOMIZATION

int ClangTool::run(ToolAction *Action) {
  // Exists solely for the purpose of lookup of the resource path.
  // This just needs to be some symbol in the binary.
  static int StaticSymbol;

  // First insert all absolute paths into the in-memory VFS. These are global
  // for all compile commands.
  if (SeenWorkingDirectories.insert("/").second)
    for (const auto &MappedFile : MappedFileContents)
      if (llvm::sys::path::is_absolute(MappedFile.first))
        InMemoryFileSystem->addFile(
            MappedFile.first, 0,
            llvm::MemoryBuffer::getMemBuffer(MappedFile.second));

  bool ProcessingFailed = false;
  bool FileSkipped = false;
  // Compute all absolute paths before we run any actions, as those will change
  // the working directory.
  std::vector<std::string> AbsolutePaths;
#ifdef SYCLomatic_CUSTOMIZATION
  if(DoGetRunRound() == 0) {
#endif // SYCLomatic_CUSTOMIZATION
  AbsolutePaths.reserve(SourcePaths.size());
  for (auto SourcePath : SourcePaths) {
    auto AbsPath = getAbsolutePath(*OverlayFileSystem, SourcePath);
    if (!AbsPath) {
      llvm::errs() << "Skipping " << SourcePath
                   << ". Error while getting an absolute path: "
                   << llvm::toString(AbsPath.takeError()) << "\n";
      continue;
    }
#if defined(_WIN32)
    std::transform(AbsPath->begin(), AbsPath->end(), AbsPath->begin(),
                   [](unsigned char c) { return std::tolower(c); });
#endif
    AbsolutePaths.push_back(std::move(*AbsPath));
  }
#ifdef SYCLomatic_CUSTOMIZATION
  // If target source file names do not exist in the command line, dpct will
  // migrate all relevant files it detects in the compilation database.
  if (SourcePaths.size() == 0) {
    std::vector<std::string> SourcePaths = Compilations.getAllFiles();
    for (auto SourcePath : SourcePaths) {
#if defined(_WIN32)
      std::transform(SourcePath.begin(), SourcePath.end(), SourcePath.begin(),
                     [](unsigned char c) { return std::tolower(c); });
#endif
      AbsolutePaths.push_back(SourcePath);
      CollectFileFromDB(SourcePath);
    }
  } else {
    if (isFileProcessAllSet()) {
      const std::string Msg =
          "Warning: --process-all option was ignored, since input files were "
          "provided in command line.\n";
      DoPrintHandle(Msg, false);
    }
  }
  } else {
    for (auto File : GetReProcessFile()) {
#if defined(_WIN32)
      std::transform(File.begin(), File.end(), File.begin(),
                     [](unsigned char c) { return std::tolower(c); });
#endif
      AbsolutePaths.push_back(File);
    }
  }
#endif // SYCLomatic_CUSTOMIZATION
  // Remember the working directory in case we need to restore it.
  std::string InitialWorkingDir;
  if (auto CWD = OverlayFileSystem->getCurrentWorkingDirectory()) {
    InitialWorkingDir = std::move(*CWD);
  } else {
    llvm::errs() << "Could not get working directory: "
                 << CWD.getError().message() << "\n";
  }

  for (llvm::StringRef File : AbsolutePaths) {

#ifndef SYCLomatic_CUSTOMIZATION
    // Currently implementations of CompilationDatabase::getCompileCommands can
    // change the state of the file system (e.g.  prepare generated headers), so
    // this method needs to run right before we invoke the tool, as the next
    // file may require a different (incompatible) state of the file system.
    //
    // FIXME: Make the compilation database interface more explicit about the
    // requirements to the order of invocation of its members.
    std::vector<CompileCommand> CompileCommandsForFile =
        Compilations.getCompileCommands(File);
    if (CompileCommandsForFile.empty()) {
      llvm::errs() << "Skipping " << File << ". Compile command not found.\n";
      FileSkipped = true;
      continue;
    }
    for (CompileCommand &CompileCommand : CompileCommandsForFile) {
      // FIXME: chdir is thread hostile; on the other hand, creating the same
      // behavior as chdir is complex: chdir resolves the path once, thus
      // guaranteeing that all subsequent relative path operations work
      // on the same path the original chdir resulted in. This makes a
      // difference for example on network filesystems, where symlinks might be
      // switched during runtime of the tool. Fixing this depends on having a
      // file system abstraction that allows openat() style interactions.
      if (OverlayFileSystem->setCurrentWorkingDirectory(
              CompileCommand.Directory))
        llvm::report_fatal_error("Cannot chdir into \"" +
                                 Twine(CompileCommand.Directory) + "\"!");

      // Now fill the in-memory VFS with the relative file mappings so it will
      // have the correct relative paths. We never remove mappings but that
      // should be fine.
      if (SeenWorkingDirectories.insert(CompileCommand.Directory).second)
        for (const auto &MappedFile : MappedFileContents)
          if (!llvm::sys::path::is_absolute(MappedFile.first))
            InMemoryFileSystem->addFile(
                MappedFile.first, 0,
                llvm::MemoryBuffer::getMemBuffer(MappedFile.second));

      std::vector<std::string> CommandLine = CompileCommand.CommandLine;

      if (ArgsAdjuster)
        CommandLine = ArgsAdjuster(CommandLine, CompileCommand.Filename);

      assert(!CommandLine.empty());

      // Add the resource dir based on the binary of this tool. argv[0] in the
      // compilation database may refer to a different compiler and we want to
      // pick up the very same standard library that compiler is using. The
      // builtin headers in the resource dir need to match the exact clang
      // version the tool is using.
      // FIXME: On linux, GetMainExecutable is independent of the value of the
      // first argument, thus allowing ClangTool and runToolOnCode to just
      // pass in made-up names here. Make sure this works on other platforms.
      injectResourceDir(CommandLine, "clang_tool", &StaticSymbol);

      // FIXME: We need a callback mechanism for the tool writer to output a
      // customized message for each file.
      LLVM_DEBUG({ llvm::dbgs() << "Processing: " << File << ".\n"; });
      ToolInvocation Invocation(std::move(CommandLine), Action, Files.get(),
                                PCHContainerOps);
      Invocation.setDiagnosticConsumer(DiagConsumer);

      if (!Invocation.run()) {
        // FIXME: Diagnostics should be used instead.
        if (PrintErrorMessage)
          llvm::errs() << "Error while processing " << File << ".\n";
        ProcessingFailed = true;
      }
    }
#else
    if(isExcludePath(File.str(), true)) {
      continue;
    }
    int Ret = processFiles(File, ProcessingFailed, FileSkipped, StaticSymbol,
                            Action);
    if (Ret == -1)
      continue;
    else if (Ret < -1)
      return Ret;
#endif // SYCLomatic_CUSTOMIZATION
  }

#ifdef SYCLomatic_CUSTOMIZATION
  // if input file(s) is not specified in command line, and the process-all
  // option is given in the comomand line, dpct tries to migrate or copy all
  // files from -in-root to the output directory.
  if(SourcePaths.size() == 0 && DoGetRunRound() == 0) {
    std::vector<std::string> FilesNotProcessed;

    // To traverse all the files in the directory specified by
    // -in-root, collecting *.cu files not processed by the first loop of
    // calling processFiles() into FilesNotProcessed, and copies the rest
    // files to the output directory.
    DoFileProcessHandle(FilesNotProcessed);
    for (auto &Entry : FilesNotProcessed) {
      auto File = llvm::StringRef(Entry);

      int Ret = processFilesWithCrashGuard(this, File, ProcessingFailed,
                                           FileSkipped, StaticSymbol, Action);
      if (Ret == ProcessingFilesCrash)
        continue;
      else if (Ret < 0)
        return Ret;
    }
  }

  // exit point for the file processing.
#endif // SYCLomatic_CUSTOMIZATION
  if (!InitialWorkingDir.empty()) {
    if (auto EC =
            OverlayFileSystem->setCurrentWorkingDirectory(InitialWorkingDir))
      llvm::errs() << "Error when trying to restore working dir: "
                   << EC.message() << "\n";
  }
  return ProcessingFailed ? 1 : (FileSkipped ? 2 : 0);
}

namespace {

class ASTBuilderAction : public ToolAction {
  std::vector<std::unique_ptr<ASTUnit>> &ASTs;

public:
  ASTBuilderAction(std::vector<std::unique_ptr<ASTUnit>> &ASTs) : ASTs(ASTs) {}

  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *Files,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override {
    std::unique_ptr<ASTUnit> AST = ASTUnit::LoadFromCompilerInvocation(
        Invocation, std::move(PCHContainerOps),
        CompilerInstance::createDiagnostics(&Invocation->getDiagnosticOpts(),
                                            DiagConsumer,
                                            /*ShouldOwnClient=*/false),
        Files);
    if (!AST)
      return false;

    ASTs.push_back(std::move(AST));
    return true;
  }
};

} // namespace

int ClangTool::buildASTs(std::vector<std::unique_ptr<ASTUnit>> &ASTs) {
  ASTBuilderAction Action(ASTs);
  return run(&Action);
}

void ClangTool::setPrintErrorMessage(bool PrintErrorMessage) {
  this->PrintErrorMessage = PrintErrorMessage;
}

namespace clang {
namespace tooling {

std::unique_ptr<ASTUnit>
buildASTFromCode(StringRef Code, StringRef FileName,
                 std::shared_ptr<PCHContainerOperations> PCHContainerOps) {
  return buildASTFromCodeWithArgs(Code, std::vector<std::string>(), FileName,
                                  "clang-tool", std::move(PCHContainerOps));
}

std::unique_ptr<ASTUnit> buildASTFromCodeWithArgs(
    StringRef Code, const std::vector<std::string> &Args, StringRef FileName,
    StringRef ToolName, std::shared_ptr<PCHContainerOperations> PCHContainerOps,
    ArgumentsAdjuster Adjuster, const FileContentMappings &VirtualMappedFiles,
    DiagnosticConsumer *DiagConsumer) {
  std::vector<std::unique_ptr<ASTUnit>> ASTs;
  ASTBuilderAction Action(ASTs);
  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFileSystem(
      new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem(
      new llvm::vfs::InMemoryFileSystem);
  OverlayFileSystem->pushOverlay(InMemoryFileSystem);
  llvm::IntrusiveRefCntPtr<FileManager> Files(
      new FileManager(FileSystemOptions(), OverlayFileSystem));

  ToolInvocation Invocation(
      getSyntaxOnlyToolArgs(ToolName, Adjuster(Args, FileName), FileName),
      &Action, Files.get(), std::move(PCHContainerOps));
  Invocation.setDiagnosticConsumer(DiagConsumer);

  InMemoryFileSystem->addFile(FileName, 0,
                              llvm::MemoryBuffer::getMemBufferCopy(Code));
  for (auto &FilenameWithContent : VirtualMappedFiles) {
    InMemoryFileSystem->addFile(
        FilenameWithContent.first, 0,
        llvm::MemoryBuffer::getMemBuffer(FilenameWithContent.second));
  }

  if (!Invocation.run())
    return nullptr;

  assert(ASTs.size() == 1);
  return std::move(ASTs[0]);
}

} // namespace tooling
} // namespace clang
