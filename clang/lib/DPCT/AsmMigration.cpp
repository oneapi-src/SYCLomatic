//===------------------------ AsmMigration.cpp
//-----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===//

#include "AsmMigration.h"
#include "AnalysisInfo.h"
#include "Asm/AsmParser.h"
#include "CallExprRewriter.h"
#include "MigrationRuleManager.h"

using namespace clang;
using namespace clang::dpct;
using llvm::StringMap;

namespace {

class AsmStmtConsumer {
  llvm::raw_ostream &OS;
  StringMap<std::string> SymbolAlias;

public:
  AsmStmtConsumer(llvm::raw_ostream &OS) : OS(OS) {}

  void AddSymbolAlias(StringRef Origin, StringRef Alias) {
    SymbolAlias[Origin] = Alias;
  }

  bool HasSymbolAlias(StringRef Origin) const {
    return SymbolAlias.find(Origin) != SymbolAlias.end();
  }

  StringRef GetSymbolAlias(StringRef Origin) { return SymbolAlias[Origin]; }

  void HandleStmt(const AsmStatement *Stmt);
  void EmitInst(const AsmStatement *Inst);
  void EmitMov(const AsmStatement *Mov);
};

void AsmStmtConsumer::HandleStmt(const AsmStatement *Stmt) {
  if (!Stmt)
    return;

  switch (Stmt->Kind) {
  case AsmStatement::SK_Block:
    OS << "{\n\t";
    for (const auto *S : Stmt->Block) {
      HandleStmt(S);
    }
    OS << "}\n";
    break;
  case AsmStatement::SK_Inst:
    if (Stmt->Pred) {
      /// TODO: Generate predicate guard expr
    }
    if (Stmt->Body) {
      return EmitInst(Stmt->Body);
    }
    break;
  case AsmStatement::SK_Variable:
    if (HasSymbolAlias(Stmt->Variable->Name))
      OS << GetSymbolAlias(Stmt->Variable->Name);
    else
      OS << Stmt->Variable->Name;
    break;
  case AsmStatement::SK_Integer:
    OS << Stmt->i64 << "ll";
    break;
  case AsmStatement::SK_Unsigned:
    OS << Stmt->u64 << "ull";
    break;
  case AsmStatement::SK_Float:
    OS << Stmt->f32 << "f";
    break;
  case AsmStatement::SK_Double:
    OS << Stmt->f64;
    break;
  default:
    break;
  }
}

void AsmStmtConsumer::EmitInst(const AsmStatement *Inst) {
  if (!Inst)
    return;
  switch (Inst->InstructionAttr.Opcode) {
  case ptx::InstKind::Mov:
    EmitMov(Inst);
    break;
  default:
    break;
  }
  OS << ";\n";
}

void AsmStmtConsumer::EmitMov(const AsmStatement *Inst) {
  if (Inst->Operands.size() != 2)
    return;
  HandleStmt(Inst->Operands[0]);
  OS << " = ";
  HandleStmt(Inst->Operands[1]);
}

} // namespace

void AsmRule::registerMatcher(ast_matchers::MatchFinder &MF) {
  using namespace clang::ast_matchers;
  MF.addMatcher(
      asmStmt(hasAncestor(functionDecl(
                  anyOf(hasAttr(attr::CUDADevice), hasAttr(attr::CUDAGlobal)))))
          .bind("asm"),
      this);
}

bool canAsmLop3ExprFast(std::ostringstream &OS, const std::string &a,
                        const std::string &b, const std::string &c,
                        const std::uint8_t imm) {
#define EMPTY4 "", "", "", ""
#define EMPTY16 EMPTY4, EMPTY4, EMPTY4, EMPTY4
  static const std::string FastMap[256] = {
      /*0x00*/ "0",
      // clang-format off
      EMPTY16, EMPTY4, EMPTY4, "",
      /*0x1a*/ "(@a & @b | @c) ^ @a",
      "", "", "",
      /*0x1e*/ "@a ^ (@b | @c)",
      EMPTY4, EMPTY4, EMPTY4, "", "",
      /*0x2d*/ "~@a ^ (~@b & @c)",
      EMPTY16, "", "",
      /*0x40*/ "@a & @b & ~@c",
      EMPTY16, EMPTY16, EMPTY16, EMPTY4, "", "", "",
      /*0x78*/ "@a ^ (@b & @c)",
      EMPTY4, "", "", "",
      /*0x80*/ "@a & @b & @c",
      EMPTY16, EMPTY4, "",
      /*0x96*/ "@a ^ @b ^ @c",
      EMPTY16, EMPTY4, EMPTY4, EMPTY4, "",
      /*0xb4*/ "@a ^ (@b & ~@c)",
      "", "", "",
      /*0xb8*/ "(@a ^ (@b & (@c ^ @a)))",
      EMPTY16, EMPTY4, EMPTY4, "",
      /*0xd2*/ "@a ^ (~@b & @c)",
      EMPTY16, EMPTY4, "",
      /*0xe8*/ "((@a & (@b | @c)) | (@b & @c))",
      "",
      /*0xea*/ "(@a & @b) | @c",
      EMPTY16, "", "", "",
      // clang-format on
      /*0xfe*/ "@a | @b | @c",
      /*0xff*/ "1"};
#undef EMPTY16
#undef EMPTY4
  const StringRef Expr = FastMap[imm];
  if (Expr.empty()) {
    return false;
  }
  OS << " ";
  const StringRef ReplaceMap[3] = {a, b, c};
  std::string::size_type Pre = 0;
  auto Pos = Expr.find('@');
  while (Pos != std::string::npos) {
    OS << Expr.substr(Pre, Pos - Pre).str();
    OS << ReplaceMap[Expr[Pos + 1] - 'a'].str();
    Pre = Pos + 2;
    Pos = Expr.find('@', Pre);
  }
  OS << Expr.substr(Pre).str();
  return true;
}

std::string getAsmLop3Expr(const llvm::SmallVector<std::string, 5> &Operands) {
  if (Operands.size() != 5) {
    return "";
  }
  const auto &d = Operands[0];
  const auto &a = Operands[1];
  const auto &b = Operands[2];
  const auto &c = Operands[3];
  auto imm = std::stoi(Operands[4], 0, 16);
  assert(imm >= 0 && imm <= UINT8_MAX);
  std::ostringstream OS;
  OS << d << " =";
  if (canAsmLop3ExprFast(OS, a, b, c, imm)) {
    return OS.str();
  }
  if (imm & 0x01)
    OS << " (~" << a << " & ~" << b << " & ~" << c << ") |";
  if (imm & 0x02)
    OS << " (~" << a << " & ~" << b << " & " << c << ") |";
  if (imm & 0x04)
    OS << " (~" << a << " & " << b << " & ~" << c << ") |";
  if (imm & 0x08)
    OS << " (~" << a << " & " << b << " & " << c << ") |";
  if (imm & 0x10)
    OS << " (" << a << " & ~" << b << " & ~" << c << ") |";
  if (imm & 0x20)
    OS << " (" << a << " & ~" << b << " & " << c << ") |";
  if (imm & 0x40)
    OS << " (" << a << " & " << b << " & ~" << c << ") |";
  if (imm & 0x80)
    OS << " (" << a << " & " << b << " & " << c << ") |";
  auto ret = OS.str();
  return ret.replace(ret.length() - 2, 2, "");
}

void AsmRule::runRule(const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto *AS = getNodeAsType<AsmStmt>(Result, "asm")) {
    if (const auto *Asm = dyn_cast<GCCAsmStmt>(AS)) {
      const auto &C = DpctGlobalInfo::getContext();
      std::string S = Asm->generateAsmString(C);

      AsmContext Context;
      std::string Replacement;
      llvm::raw_string_ostream OS(Replacement);
      AsmStmtConsumer Consumer(OS);
      llvm::SourceMgr Mgr;
      Mgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(S),
                             llvm::SMLoc());
      PtxParser Parser(Context, Mgr);
      unsigned OperandIdx = 0;
      std::string AsmString;
      for (unsigned I = 0; I < Asm->getNumInputs(); ++I) {
        ExprAnalysis EA;
        EA.analyze(Asm->getInputExpr(I));
        std::string Placeholder = "$" + std::to_string(OperandIdx++);
        Parser.AddBuiltinSymbol(Placeholder, nullptr);
        Consumer.AddSymbolAlias(Placeholder, EA.getReplacedString());
      }
      for (unsigned I = 0; I < Asm->getNumOutputs(); ++I) {
        ExprAnalysis EA;
        EA.analyze(Asm->getOutputExpr(I));
        std::string Placeholder = "$" + std::to_string(OperandIdx++);
        Parser.AddBuiltinSymbol(Placeholder, nullptr);
        Consumer.AddSymbolAlias(Placeholder, EA.getReplacedString());
      }
      auto Inst = Parser.ParseStatement();
      if (Inst.isInvalid())
        return;
      Consumer.HandleStmt(Inst.get());
      OS.flush();
      emplaceTransformation(new ReplaceStmt(AS, Replacement));
      return;
    }
    auto AsmString = AS->generateAsmString(*Result.Context);
    auto TemplateString = StringRef(AsmString).substr(0, AsmString.find(';'));
    auto CurrIndex = TemplateString.find(' ');
    auto OpCode = TemplateString.substr(0, CurrIndex);
    if (OpCode == "lop3.b32") {
      // ASM instruction pattern: lop3.b32 d, a, b, c, immLut;
      llvm::SmallVector<std::string, 4> Args;
      for (const auto *const it : AS->children()) {
        ExprAnalysis EA;
        EA.analyze(cast<Expr>(it));
        if (isa<IntegerLiteral>(it) || isa<DeclRefExpr>(it) ||
            isa<ImplicitCastExpr>(it)) {
          Args.push_back(EA.getReplacedString());
        } else {
          Args.push_back("(" + EA.getReplacedString() + ")");
        }
      }
      llvm::SmallVector<std::string, 5> Operands;
      auto PreIndex = CurrIndex;
      CurrIndex = TemplateString.find(",", PreIndex);
      // Clang will generate the ASM instruction into a string like this:
      // Cuda code: asm("lop3.b32 %0, %1*%1, %1, 3, 0x1A;" : "=r"(b) : "r"(a));
      // TemplateString: "lop3.b32 $0, $1*$1,$ 1, 3, 0x1A"
      while (PreIndex != StringRef::npos) {
        auto TempStr =
            TemplateString.substr(PreIndex + 1, CurrIndex - PreIndex - 1).str();
        // Replace all args, example: the "$1*$1" will be replace by "(a*a)".
        if (TempStr.find('$') != TempStr.length() - 2 && Operands.size() != 4) {
          // When the operands only contain a register, or is the last imm, not
          // need add the paren.
          TempStr = "(" + TempStr + ")";
        }
        auto ArgIndex = TempStr.find('$');
        while (ArgIndex != std::string::npos) {
          // The PTX Instructions has mostly 4 parameters, so just use the char
          // after '$'.
          auto ArgNo = TempStr[ArgIndex + 1] - '0';
          TempStr.replace(ArgIndex, 2, Args[ArgNo]);
          ArgIndex = TempStr.find('$');
        }
        Operands.push_back(std::move(TempStr));
        PreIndex = CurrIndex;
        CurrIndex = TemplateString.find(",", PreIndex + 1);
      }
      auto Replacement = getAsmLop3Expr(Operands);
      if (!Replacement.empty()) {
        return emplaceTransformation(new ReplaceStmt(AS, Replacement));
      }
    }
    report(AS->getAsmLoc(), Diagnostics::DEVICE_ASM, true);
  }
  return;
}
