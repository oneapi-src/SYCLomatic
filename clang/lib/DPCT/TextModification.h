//===--------------- TextModification.h -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
using TransformSetTy = std::vector<std::shared_ptr<TextModification>>;

class ReplaceInclude;
using IncludeMapSetTy =
    std::map<std::string, std::vector<std::unique_ptr<ReplaceInclude>>>;

enum InsertPosition {
  IP_AlwaysLeft = 0,
  IP_Left,
  IP_Right,
};

/// Extend Replacement to contain more meta info of Replacement inserted by
/// AST Rule. Further Analysis Pass like Merge Pass can happen based
/// on this meta info of Replacement.
///  eg. Replacement happen at same position may be merged to avoid conflict.
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

  /// merge the constant info to LHS
  /// the information precedence: HostDevice > Device > Host > empty
  void mergeConstantInfo(std::shared_ptr<ExtReplacement> RHS) {
    auto setConstantInfoUsingRHSInfo = [&]() {
      setConstantFlag(RHS->getConstantFlag());
      setConstantOffset(RHS->getConstantOffset());
      setInitStr(RHS->getInitStr());
      setNewHostVarName(RHS->getNewHostVarName());
    };

    // LHS has highest precedence or RHS has lowest precedence, use LHS directly
    if (getConstantFlag() == dpct::ConstantFlagType::HostDevice ||
        RHS->getConstantFlag() == dpct::ConstantFlagType::Default) {
      return;
    }

    // LHS has lowest precedence or RHS has highest precedence, use RHS directly
    if (RHS->getConstantFlag() == dpct::ConstantFlagType::HostDevice ||
        getConstantFlag() == dpct::ConstantFlagType::Default) {
      setConstantInfoUsingRHSInfo();
      return;
    }

    // Code at here means it is either "Device" or "Host", use device
    if (getConstantFlag() == dpct::ConstantFlagType::Device ||
        RHS->getConstantFlag() == dpct::ConstantFlagType::Host) {
      return;
    }
    if (RHS->getConstantFlag() == dpct::ConstantFlagType::Host ||
        getConstantFlag() == dpct::ConstantFlagType::Device) {
      setConstantInfoUsingRHSInfo();
      return;
    }
  }

  inline bool IsSYCLHeaderNeeded() { return SYCLHeaderNeeded; }
  inline void setSYCLHeaderNeeded(bool Val) { SYCLHeaderNeeded = Val; }

private:
  InsertPosition InsertPos = IP_Left;
  const TextModification *TM;
  unsigned PairID = 0;
  bool SYCLHeaderNeeded = true;
};

enum class TextModificationID : int {
#define TRANSFORMATION(TYPE) TYPE,
#include "Transformations.inc"
#undef TRANSFORMATION
};

using TMID = TextModificationID;

/// Base class for tool-related source code modifications.
class TextModification {
public:
  // getReplacement() method will be called according to the grouping:
  // Modifications belonging to G1 will have getReplacement() called
  // before modifications belonging to G2, and G2s before G3s
  enum Group { Any = 0, G1 = 1, G2 = 2, G3 = 3 };

  static const std::unordered_map<int, std::string> TMNameMap;
  InsertPosition InsertPos = InsertPosition::IP_Left;

public:
  TextModification(TMID _TMID) : ID(_TMID), Key(Any) {}
  TextModification(TMID _TMID, Group _Key)
      : ID(_TMID), Key(_Key) {}
  virtual ~TextModification() {}
  /// Generate actual Replacement from this TextModification object.
  virtual std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const = 0;
  virtual void print(llvm::raw_ostream &OS, ASTContext &Context,
                     const bool PrintDetail = true) const = 0;
  bool operator<(const TextModification &TM) const { return Key < TM.Key; }
  static bool Compare(const std::shared_ptr<TextModification> &L,
                      const std::shared_ptr<TextModification> &R) {
    return L->Key < R->Key;
  }

  TMID getID() const { return ID; }
  const std::string &getName() const;

  void setParentRuleName(StringRef Name) { ParentRuleName = Name; }
  StringRef getParentRuleName() const { return ParentRuleName; }
  inline void setPairID(unsigned Pair) { PairID = Pair; }
  dpct::ConstantFlagType getConstantFlag() const { return ConstantFlag; }
  void setConstantFlag(dpct::ConstantFlagType F) { ConstantFlag = F; }
  unsigned int getLineBeginOffset() const { return LineBeginOffset; }
  void setLineBeginOffset(unsigned int O) { LineBeginOffset = O; }
  unsigned int getConstantOffset() const { return ConstantOffset; }
  void setConstantOffset(unsigned int O) { ConstantOffset = O; }
  std::string getInitStr() const { return InitStr; }
  void setInitStr(std::string S) { InitStr = S; }
  std::string getNewHostVarName() const { return NewHostVarName; }
  void setNewHostVarName(std::string N) { NewHostVarName = N; }
  void setIgnoreTM(bool Flag = true) { IgnoreTM = Flag; }
  bool isIgnoreTM() const { return IgnoreTM; }
  bool getNotFormatFlag() const { return NotFormatFlag; }
  void setInsertPosition(InsertPosition IP) { InsertPos = IP; }

  // BlockLevelFormatFlag is used to decide which replacement need to be format
  // second time.
  // Each replacement should be processed by the first formatting (except the
  // NotFormatFlag is ture). But because each line's indent is kept as same as
  // the original code's indent, if the migration is one line => multi lines,
  // the indent may be strange. So we do a second time format for these
  // replacement, using unified IndentWidth to do the format. Currently, the
  // replacement need second format are: kernel calls, warnings and some library
  // APIs migration.
  void setBlockLevelFormatFlag(bool Flag = true) {
    BlockLevelFormatFlag = Flag;
  }
  bool getBlockLevelFormatFlag() const { return BlockLevelFormatFlag; }

private:
  const TMID ID;
  Group Key;
  StringRef ParentRuleName;
  unsigned PairID = 0;
  bool BlockLevelFormatFlag = false;
  // below members are used for process __constant__ macro used in host and
  // device
  unsigned int LineBeginOffset = 0;
  bool IgnoreTM = false;
  dpct::ConstantFlagType ConstantFlag = dpct::ConstantFlagType::Default;
  unsigned int ConstantOffset = 0;
  std::string InitStr = "";
  std::string NewHostVarName = "";

protected:
  bool NotFormatFlag = false;
};

/// Insert string in given position.
class InsertText : public TextModification {
  SourceLocation Begin;
  std::string T;
  unsigned PairID;

public:
  InsertText(SourceLocation Loc, const std::string &S, unsigned PairID = 0)
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
  // If the callexpr need to migrate is a macro, IsProcessMacro should
  // be true and the migration will be done correctly.
  bool IsProcessMacro;
  std::string ReplacementString;

  // When replacing the Stmt with empty string, an option to clean up
  // redundant trailing semicolons and spaces in the same line
  bool IsCleanup = true;

  // Since getReplacement() is a const function and IsMacroRemoved is assigned
  // in it, IsMacroRemoved is declared as "mutable".
  // In getReplacement(), if the callexpr spelling is in macro define and it is
  // outermost, this callexpr will be removed in macro definition and this
  // flag will be true. Then in removeStmtWithCleanups() function, the call of
  // this macro will be removed also.
  mutable bool IsMacroRemoved = false;

  static bool inCompoundStmt(const Stmt *E);

public:
  template <class... Args>
  ReplaceStmt(const Stmt *E, Args &&...S)
      : TextModification(TMID::ReplaceStmt), TheStmt(E), IsProcessMacro(false),
        ReplacementString(std::forward<Args>(S)...),
        IsCleanup(inCompoundStmt(E)) {}

  template <class... Args>
  ReplaceStmt(const Stmt *E, bool IsNeedProcessMacro, Args &&...S)
      : TextModification(TMID::ReplaceStmt), TheStmt(E),
        IsProcessMacro(IsNeedProcessMacro),
        ReplacementString(std::forward<Args>(S)...),
        IsCleanup(inCompoundStmt(E)) {}

  template <class... Args>
  ReplaceStmt(const Stmt *E, bool IsNeedProcessMacro, bool IsNeedCleanup,
              Args &&...S)
      : TextModification(TMID::ReplaceStmt), TheStmt(E),
        IsProcessMacro(IsNeedProcessMacro),
        ReplacementString(std::forward<Args>(S)...), IsCleanup(IsNeedCleanup) {}

  template <class... Args>
  ReplaceStmt(const CUDAKernelCallExpr *E, Args &&...S)
      : ReplaceStmt((const Stmt *)E, std::forward<Args>(S)...) {
    // Don't clean up for CUDAKernelCallExpr to avoid overlapping problems
    IsCleanup = false;
    IsProcessMacro = true;
  }

  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;

  std::shared_ptr<ExtReplacement>
  removeStmtWithCleanups(const SourceManager &SM) const;
};

class ReplaceDecl : public TextModification {
  const Decl *TheDecl;
  std::string ReplacementString;

public:
  template <class... Args>
  ReplaceDecl(const Decl *E, Args &&...S)
      : TextModification(TMID::ReplaceDecl), TheDecl(E),
        ReplacementString(std::forward<Args>(S)...) {}

  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// Replace the call name of function calls
class ReplaceCalleeName : public TextModification {
  const CallExpr *C;
  std::string ReplStr;

public:
  ReplaceCalleeName(const CallExpr *C, std::string &&S)
      : TextModification(TMID::ReplaceCalleeName), C(C), ReplStr(S) {}

  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
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

/// Replace type in variable declaration.
class ReplaceTypeInDecl : public TextModification {
  TypeLoc TL;
  const DeclaratorDecl *DD = nullptr; // DD points to a VarDecl or a FieldDecl
  SourceLocation SL; // The input source location used to record MigrationInfo
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
  ReplaceTypeInDecl(const SourceLocation SL, const TemplateArgumentLoc &TAL,
                    std::string &&T)
      : TextModification(TMID::ReplaceTypeInDecl), SL(SL), T(T) {
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

/// Insert a string after a statement
class InsertAfterDecl : public TextModification {
  const Decl *D;
  std::string T;

public:
  InsertAfterDecl(const Decl *D, std::string &&T)
      : TextModification(TMID::InsertAfterStmt), D(D), T(T) {}

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
  bool UseTextBegin = false;

public:
  InsertComment(SourceLocation SL, std::string Text, bool UseTextBegin = false)
      : TextModification(TMID::InsertComment), SL(SL), Text(Text),
        UseTextBegin(UseTextBegin) {}

  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// Replace including directives
class ReplaceInclude : public TextModification {
  CharSourceRange Range;
  std::string T;
  bool RemoveTrailingSpaces;

public:
  ReplaceInclude(CharSourceRange Range, std::string T,
                 bool RemoveTrailingSpaces = false)
      : TextModification(TMID::ReplaceInclude), Range(Range), T(std::move(T)),
        RemoveTrailingSpaces(RemoveTrailingSpaces) {}

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

public:
  ReplaceText(const SourceLocation &Begin, unsigned Len, std::string &&S)
      : TextModification(TMID::ReplaceText), BeginLoc(Begin), Len(Len),
        T(std::move(S)) {
    this->NotFormatFlag = false;
  }
  ReplaceText(const SourceLocation &Begin, const SourceLocation &End,
              std::string &&S)
      : TextModification(TMID::ReplaceText), BeginLoc(Begin),
        Len(End.getRawEncoding() - Begin.getRawEncoding()), T(std::move(S)) {
    this->NotFormatFlag = false;
  }
  ReplaceText(const SourceLocation &Begin, unsigned Len, std::string &&S,
              bool NotFormatFlag)
      : TextModification(TMID::ReplaceText), BeginLoc(Begin), Len(Len),
        T(std::move(S)) {
    this->NotFormatFlag = NotFormatFlag;
  }

  std::shared_ptr<ExtReplacement>
  getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

} // namespace dpct
} // namespace clang

#endif // DPCT_TEXT_MODIFICATION_H
