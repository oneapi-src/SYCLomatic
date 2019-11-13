//===--- TextModification.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 - 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef DPCT_TEXT_MODIFICATION_H
#define DPCT_TEXT_MODIFICATION_H

#include "MapNames.h"
#include "Utility.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Tooling.h"

#include <string>

namespace clang {
namespace dpct {

class KernelCallExpr;
class TextModification;
using TransformSetTy = std::vector<std::unique_ptr<TextModification>>;
enum InsertPosition {
  InsertPositionAlwaysLeft = 0,
  InsertPositionLeft,
  InsertPositionRight,
};

/// Extend Replacemnt to contain more meta info of Replament inserted by
/// AST Rule. Further Analysis Pass like Merge Pass can happen based
/// on these meta info of Replament.
///  eg. Replament happen at same position may be merged to avoid conflict.
class ExtReplacement : public tooling::Replacement {
public:
  /// Creates an invalid (not applicable) replacement.
  ExtReplacement() : Replacement(){};

  /// Creates a replacement of the range [Offset, Offset+Length) in
  /// FilePath with ReplacementText.
  ///
  /// \param FilePath A source file accessible via a SourceManager.
  /// \param Offset The byte offset of the start of the range in the file.
  /// \param Length The length of the range in bytes.
  ExtReplacement(StringRef FilePath, unsigned Offset, unsigned Length,
                 StringRef ReplacementText, const TextModification *_TM)
      : Replacement(FilePath, Offset, Length, ReplacementText), TM(_TM) {}

  /// Creates a Replacement of the range [Start, Start+Length) with
  /// ReplacementText.
  ExtReplacement(const SourceManager &Sources, SourceLocation Start,
                 unsigned Length, StringRef ReplacementText,
                 const TextModification *_TM)
      : Replacement(Sources, Start, Length, ReplacementText), TM(_TM) {}

  /// Creates a Replacement of the given range with ReplacementText.
  ExtReplacement(const SourceManager &Sources, const CharSourceRange &Range,
                 StringRef ReplacementText, const TextModification *_TM,
                 const LangOptions &LangOpts = LangOptions())
      : Replacement(Sources, Range, ReplacementText, LangOpts), TM(_TM) {}

  /// Creates a Replacement of the node with ReplacementText.
  template <typename Node>
  ExtReplacement(const SourceManager &Sources, const Node &NodeToReplace,
                 StringRef ReplacementText, const TextModification *_TM,
                 const LangOptions &LangOpts = LangOptions())
      : Replacement(Sources, NodeToReplace, ReplacementText, LangOpts),
        TM(_TM) {}
  void setInsertPosition(InsertPosition IP) { InsertPos = IP; }
  unsigned int getInsertPosition() const { return InsertPos; }

  const TextModification *getParentTM() const { return TM; }

  inline void setPairID(unsigned Pair) { PairID = Pair; }
  inline unsigned getPairID() { return PairID; }

  bool equal(std::shared_ptr<ExtReplacement> RHS) {
    return getLength() == RHS->getLength() &&
           getReplacementText().equals(RHS->getReplacementText());
  }

private:
  InsertPosition InsertPos = InsertPositionLeft;
  unsigned BeginLine = 0, EndLine = 0;
  const TextModification *TM;
  unsigned PairID = 0;
};

enum class TextModificationID : int {
#define TRANSFORMATION(TYPE) TYPE,
#include "Transformations.inc"
#undef TRANSFORMATION
};

using TMID = TextModificationID;

/// Base class for compatibility tool-related source code modifications.
class TextModification {
public:
  // getReplacement() method will be called according to the grouping:
  // Modifications belonging to G1 will have getReplacement() called
  // before modifications belonging to G2, and G2s before G3s
  enum Group { Any = 0, G1 = 1, G2 = 2, G3 = 3 };

  static const std::unordered_map<int, std::string> TMNameMap;

public:
  TextModification(TMID _TMID) : ID(_TMID), Key(Any), ParentRuleID(0) {}
  TextModification(TMID _TMID, Group _Key)
      : ID(_TMID), Key(_Key), ParentRuleID(0) {}
  virtual ~TextModification() {}
  /// Generate actual Replacement from this TextModification object.
  virtual std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const = 0;
  virtual void print(llvm::raw_ostream &OS, ASTContext &Context,
                     const bool PrintDetail = true) const = 0;
  bool operator<(const TextModification &TM) const { return Key < TM.Key; }
  static bool Compare(const std::unique_ptr<TextModification> &L,
                      const std::unique_ptr<TextModification> &R) {
    return L->Key < R->Key;
  }

  TMID getID() const { return ID; }
  const std::string &getName() const;

  void setParentRuleID(const char *RuleID) { ParentRuleID = RuleID; }
  const char *getParentRuleID() const { return ParentRuleID; }
  inline void setPairID(unsigned Pair) { PairID = Pair; }

private:
  const TMID ID;
  Group Key;
  const char *ParentRuleID;
  unsigned PairID = 0;
};

/// Insert string in given position.
class InsertText : public TextModification {
  SourceLocation Begin;
  std::string T;
  unsigned PairID;

public:
  InsertText(SourceLocation Loc, std::string &&S, unsigned PairID = 0)
      : TextModification(TMID::InsertText), Begin(Loc), T(S), PairID(PairID) {}
  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// For macros and typedefs source location is unreliable (begin and end of the
/// source range point to the same character. Replacing by token is a simple
/// workaround.
class ReplaceToken : public TextModification {
  SourceLocation Begin;
  SourceLocation End;
  std::string T;

public:
  ReplaceToken(SourceLocation Loc, std::string &&S)
      : TextModification(TMID::ReplaceToken), Begin(Loc), End(Loc), T(S) {}
  ReplaceToken(SourceLocation BLoc, SourceLocation ELoc, std::string &&S)
      : TextModification(TMID::ReplaceToken), Begin(BLoc), End(ELoc), T(S) {}
  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// Replace a statement (w/o semicolon) with a specified string.
class ReplaceStmt : public TextModification {
  const Stmt *TheStmt;
  // If ReplaceStmt replaces calls to compatibility APIs
  bool IsReplaceCompatibilityAPI;
  std::string OrigAPIName;
  // If the callexpr need to migrate is a macro, IsProcessMacro should
  // be true and the migration will be done correctly.
  bool IsProcessMacro;
  std::string ReplacementString;

  // When replacing the Stmt with empty string, an option to clean up
  // redundant trailing semicolons and spaces in the same line
  bool IsCleanup = true;

public:
  template <class... Args>
  ReplaceStmt(const Stmt *E, Args &&... S)
      : TextModification(TMID::ReplaceStmt), TheStmt(E),
        IsReplaceCompatibilityAPI(false), IsProcessMacro(false),
        ReplacementString(std::forward<Args>(S)...) {}

  template <class... Args>
  ReplaceStmt(const Stmt *E, bool IsReplaceCompatibilityAPI,
              std::string OrigAPIName, Args &&... S)
      : TextModification(TMID::ReplaceStmt), TheStmt(E),
        IsReplaceCompatibilityAPI(IsReplaceCompatibilityAPI),
        OrigAPIName(OrigAPIName), IsProcessMacro(false),
        ReplacementString(std::forward<Args>(S)...) {}

  template <class... Args>
  ReplaceStmt(const Stmt *E, bool IsReplaceCompatibilityAPI,
              std::string OrigAPIName, bool IsNeedProcessMacro, Args &&... S)
      : TextModification(TMID::ReplaceStmt), TheStmt(E),
        IsReplaceCompatibilityAPI(IsReplaceCompatibilityAPI),
        OrigAPIName(OrigAPIName), IsProcessMacro(IsNeedProcessMacro),
        ReplacementString(std::forward<Args>(S)...) {}

  template <class... Args>
  ReplaceStmt(const CUDAKernelCallExpr *E, Args &&... S)
      : ReplaceStmt((const Stmt *)E, std::forward<Args>(S)...) {
    // Don't clean up for CUDAKernelCallExpr to avoid overlapping problems
    IsCleanup = false;
    IsProcessMacro = false;
  }

  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;

  std::shared_ptr<ExtReplacement>
  removeStmtWithCleanups(const SourceManager &SM) const;
};

/// Replace the call name of function calls
class ReplaceCalleeName : public TextModification {
  const CallExpr *C;
  std::string ReplStr;
  std::string OrigAPIName;

public:
  ReplaceCalleeName(const CallExpr *C, std::string &&S,
                    const std::string &OrigAPIName)
      : TextModification(TMID::ReplaceCalleeName), C(C), ReplStr(S),
        OrigAPIName(OrigAPIName) {}

  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;

private:
  llvm::StringRef getCalleeName(const ASTContext &Context) const {
    const auto &SM = Context.getSourceManager();
    const char *Start = SM.getCharacterData(C->getBeginLoc());
    const char *End = SM.getCharacterData(C->getEndLoc());
    // Eg. xx::sqrt(double a)
    llvm::StringRef SourceCode(Start, End - Start + 1);
    size_t NameEnd = SourceCode.find('(');
    assert(NameEnd != llvm::StringRef::npos);
    return SourceCode.substr(0, NameEnd);
  }
};

/// Replace C-style cast with constructor call for a given type.
class ReplaceCCast : public TextModification {
  const CStyleCastExpr *Cast;
  std::string TypeName;

public:
  ReplaceCCast(const CStyleCastExpr *Cast, std::string &&TypeName)
      : TextModification(TMID::ReplaceCCast), Cast(Cast), TypeName(TypeName) {}
  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// Remove an attribute from a declaration.
class RemoveAttr : public TextModification {
  const Attr *TheAttr;

public:
  RemoveAttr(const Attr *A) : TextModification(TMID::RemoveAttr), TheAttr(A) {}
  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// Replace type in variable declaration.
class ReplaceTypeInDecl : public TextModification {
  TypeLoc TL;
  const DeclaratorDecl *DD; // DD points to a VarDecl or a FieldDecl
  std::string T;

public:
  ReplaceTypeInDecl(const DeclaratorDecl *DD, std::string &&T)
      : TextModification(TMID::ReplaceTypeInDecl), DD(DD), T(T) {
    assert(dyn_cast<VarDecl>(DD) || dyn_cast<FieldDecl>(DD));
    if (DD->getType()->isArrayType())
      TL = DD->getTypeSourceInfo()
               ->getTypeLoc()
               .getAs<ArrayTypeLoc>()
               .getElementLoc();
    else
      TL = DD->getTypeSourceInfo()->getTypeLoc();
  }
  ReplaceTypeInDecl(const DeclaratorDecl *DD, const TemplateArgumentLoc &TAL,
                    std::string &&T)
      : TextModification(TMID::ReplaceTypeInDecl), DD(DD), T(T) {
    assert(dyn_cast<VarDecl>(DD) || dyn_cast<FieldDecl>(DD));
    TL = TAL.getTypeSourceInfo()->getTypeLoc();
  }
  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// Replace type in variable declaration.
class ReplaceVarDecl : public TextModification {
  const VarDecl *D;
  CharSourceRange SR;
  std::string T;
  std::string Indent;
  std::string NL;

public:
  static ReplaceVarDecl *getVarDeclReplacement(const VarDecl *VD,
                                               std::string &&Text);

  ReplaceVarDecl(const VarDecl *D, std::string &&T);
  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;

private:
  void addVarDecl(const VarDecl *D, std::string &&Text);
  static std::map<unsigned, ReplaceVarDecl *> ReplaceMap;
};

/// Replace return type in function declaration.
class ReplaceReturnType : public TextModification {
  const FunctionDecl *FD;
  std::string T;

public:
  ReplaceReturnType(const FunctionDecl *FD, std::string &&T)
      : TextModification(TMID::ReplaceReturnType), FD(FD), T(T) {}
  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// Rename field in expression.
class RenameFieldInMemberExpr : public TextModification {
  const MemberExpr *ME;
  std::string T;
  unsigned PositionOfDot;

public:
  RenameFieldInMemberExpr(const MemberExpr *ME, std::string &&T,
                          unsigned PositionOfDot = 0)
      : TextModification(TMID::RenameFieldInMemberExpr, G1), ME(ME), T(T),
        PositionOfDot(PositionOfDot) {}

  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// Insert a string after a statement
class InsertAfterStmt : public TextModification {
  const Stmt *S;
  std::string T;
  unsigned PairID;
  bool DoMacroExpansion;

public:
  InsertAfterStmt(const Stmt *S, std::string &&T, unsigned PairID = 0,
                  bool DoMacroExpansion = false)
      : TextModification(TMID::InsertAfterStmt), S(S), T(T), PairID(PairID),
        DoMacroExpansion(DoMacroExpansion) {}

  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// Insert '/*  */' C style multi line comments
class InsertComment : public TextModification {
  // The comment will be inserted at this position
  SourceLocation SL;
  std::string Text;

public:
  InsertComment(SourceLocation SL, std::string Text)
      : TextModification(TMID::InsertComment), SL(SL), Text(Text) {}

  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// Replace including diretives
class ReplaceInclude : public TextModification {
  CharSourceRange Range;
  std::string T;

public:
  ReplaceInclude(CharSourceRange Range, std::string &&T)
      : TextModification(TMID::ReplaceInclude), Range(Range), T(T) {}

  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// Replace Dim3 constructors
class ReplaceDim3Ctor : public TextModification {
  bool isDecl;
  const CXXConstructExpr *Ctor;
  const CXXConstructExpr *FinalCtor;
  CharSourceRange CSR;
  mutable std::string ReplacementString;

  void setRange();
  const Stmt *getReplaceStmt(const Stmt *S) const;
  std::string getSyclRangeCtor(const CXXConstructExpr *Ctor) const;
  std::string getReplaceString() const;

public:
  ReplaceDim3Ctor(const CXXConstructExpr *_Ctor, bool _isDecl = false)
      : TextModification(TMID::ReplaceDim3Ctor, G2), isDecl(_isDecl),
        Ctor(_Ctor), FinalCtor(nullptr) {
    setRange();
  }
  ReplaceDim3Ctor(const CXXConstructExpr *_Ctor,
                  const CXXConstructExpr *_FinalCtor)
      : TextModification(TMID::ReplaceDim3Ctor, G2), isDecl(false), Ctor(_Ctor),
        FinalCtor(_FinalCtor) {
    setRange();
  }
  static const CXXConstructExpr *getConstructExpr(const Expr *E);
  ReplaceInclude *getEmpty();
  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

class InsertBeforeStmt : public TextModification {
  const Stmt *S;
  std::string T;
  unsigned PairID;
  bool DoMacroExpansion;

public:
  InsertBeforeStmt(const Stmt *S, std::string &&T, unsigned PairID = 0,
                   bool DoMacroExpansion = false)
      : TextModification(TMID::InsertBeforeStmt), S(S), T(T), PairID(PairID),
        DoMacroExpansion(DoMacroExpansion) {}

  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// Remove an argument in a function call
class RemoveArg : public TextModification {
  const CallExpr *CE;
  const unsigned N;

public:
  RemoveArg(const CallExpr *CE, const unsigned N)
      : TextModification(TMID::RemoveArg), CE(CE), N(N) {}

  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// Insert before constructor's initializer lists
class InsertBeforeCtrInitList : public TextModification {
  const CXXConstructorDecl *CDecl;
  std::string T;

  SourceLocation getInsertLoc() const;

public:
  InsertBeforeCtrInitList(const CXXConstructorDecl *S, std::string &&T)
      : TextModification(TMID::InsertBeforeCtrInitList), CDecl(S), T(T) {}

  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// Insert class names
class InsertClassName : public TextModification {
  const CXXRecordDecl *CD;
  static unsigned Count;

public:
  InsertClassName(const CXXRecordDecl *C)
      : TextModification(TMID::InsertClassName), CD(C) {}

  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// Replace raw texts
class ReplaceText : public TextModification {
  SourceLocation BeginLoc;
  unsigned Len;
  std::string T;
  // If ReplaceText replaces calls to compatibility APIs
  bool IsReplaceCompatibilityAPI;
  std::string OrigAPIName;

public:
  ReplaceText(const SourceLocation &Begin, unsigned Len, std::string &&S)
      : TextModification(TMID::ReplaceText), BeginLoc(Begin), Len(Len),
        T(std::move(S)), IsReplaceCompatibilityAPI(false), OrigAPIName("") {}
  ReplaceText(const SourceLocation &Begin, unsigned Len, std::string &&S,
              bool IsReplaceCompatibilityAPI, std::string OrigAPIName)
      : TextModification(TMID::ReplaceText), BeginLoc(Begin), Len(Len),
        T(std::move(S)), IsReplaceCompatibilityAPI(IsReplaceCompatibilityAPI),
        OrigAPIName(OrigAPIName) {}

  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

} // namespace dpct
} // namespace clang

#endif // DPCT_TEXT_MODIFICATION_H
