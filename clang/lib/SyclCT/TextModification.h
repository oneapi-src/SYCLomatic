//===--- TextModification.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2018 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//

#ifndef SYCLCT_TEXT_MODIFICATION_H
#define SYCLCT_TEXT_MODIFICATION_H

#include "MapNames.h"
#include "Utility.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Tooling.h"

#include <string>

namespace clang {
namespace syclct {

class KernelCallExpr;
class TextModification;
using TransformSetTy = std::vector<std::unique_ptr<TextModification>>;
enum InsertPosition {
  InsertPositionLeft = 0,
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
                 const TextModification *_TM, bool IsComments = false)
      : Replacement(Sources, Start, Length, ReplacementText), TM(_TM),
        IsComments(IsComments) {}

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
  void setInsertPosition(int IP) { InsertPosition = IP; }
  unsigned int getInsertPosition() const { return InsertPosition; }
  bool getMerged() const { return Merged; }
  bool isComments() const { return IsComments; }

  /// return true if two code repl has same length(not zero) and same
  /// replacement text. else return false.
  /// NOTE: length == 0  code relacement is an insert operation.
  ///       length != 0  code relacement is a real code replace operation.
  bool isEqualExtRepl(unsigned int Length, std::string &ReplacementText) const {
    if (Length == this->getLength() && this->getLength() != 0 &&
        ReplacementText == this->getReplacementText().str()) {
      return true;
    } else {
      return false;
    }
  }
  void setMerged(bool M) { Merged = M; }
  const TextModification *getParentTM() const { return TM; }

private:
  unsigned int InsertPosition = InsertPositionLeft;
  bool Merged = false;
  const TextModification *TM;
  bool IsComments = false;
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

public:
  TextModification(TMID _TMID) : ID(_TMID), Key(Any), ParentRuleID(0) {}
  TextModification(TMID _TMID, Group _Key)
      : ID(_TMID), Key(_Key), ParentRuleID(0) {}
  virtual ~TextModification() {}
  /// Generate actual Replacement from this TextModification object.
  virtual ExtReplacement getReplacement(const ASTContext &Context) const = 0;
  virtual void print(llvm::raw_ostream &OS, ASTContext &Context,
                     const bool PrintDetail = true) const = 0;
  bool operator<(const TextModification &TM) const { return Key < TM.Key; }
  static bool Compare(const std::unique_ptr<TextModification> &L,
                      const std::unique_ptr<TextModification> &R) {
    return L->Key < R->Key;
  }

  TMID getID() const { return ID; }
  const std::string getName() const;

  void setParentRuleID(const char *RuleID) { ParentRuleID = RuleID; }
  const char *getParentRuleID() const { return ParentRuleID; }

private:
  const TMID ID;
  Group Key;
  const char *ParentRuleID;
};

///  Insert string in given position.
class InsertText : public TextModification {
  SourceLocation Begin;
  std::string T;

public:
  InsertText(SourceLocation Loc, std::string &&S)
      : TextModification(TMID::InsertText), Begin(Loc), T(S) {}
  ExtReplacement getReplacement(const ASTContext &Context) const override;
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
  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// Replace a statement (w/o semicolon) with a specified string.
class ReplaceStmt : public TextModification {
  const Stmt *TheStmt;
  // If ReplaceStmt replaces calls to compatibility APIs
  bool IsReplaceCompatibilityAPI;
  std::string OrigAPIName;
  std::string ReplacementString;

public:
  template <class... Args>
  ReplaceStmt(const Stmt *E, Args &&... S)
      : TextModification(TMID::ReplaceStmt), TheStmt(E),
        IsReplaceCompatibilityAPI(false),
        ReplacementString(std::forward<Args>(S)...) {}

  template <class... Args>
  ReplaceStmt(const Stmt *E, bool IsReplaceCompatibilityAPI,
              std::string OrigAPIName, Args &&... S)
      : TextModification(TMID::ReplaceStmt), TheStmt(E),
        IsReplaceCompatibilityAPI(IsReplaceCompatibilityAPI),
        OrigAPIName(OrigAPIName), ReplacementString(std::forward<Args>(S)...) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

class ReplaceCalleeName : public TextModification {
  const CallExpr *C;
  std::string ReplStr;
  std::string OrigAPIName;

public:
  ReplaceCalleeName(const CallExpr *C, std::string &&S,
                    const std::string &OrigAPIName)
      : TextModification(TMID::ReplaceCalleeName), C(C), ReplStr(S),
        OrigAPIName(OrigAPIName) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
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
  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// Remove an attribute from a declaration.
class RemoveAttr : public TextModification {
  const Attr *TheAttr;

public:
  RemoveAttr(const Attr *A) : TextModification(TMID::RemoveAttr), TheAttr(A) {}
  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

// Replace type in var. declaration.
class ReplaceTypeInDecl : public TextModification {
  TypeLoc TL;
  const VarDecl *D;
  const FieldDecl *FD;
  std::string T;

public:
  ReplaceTypeInDecl(const VarDecl *D, std::string &&T)
      : TextModification(TMID::ReplaceTypeInDecl), D(D), FD(nullptr), T(T) {
    if (D->getType()->isArrayType())
      TL = D->getTypeSourceInfo()
               ->getTypeLoc()
               .getAs<ArrayTypeLoc>()
               .getElementLoc();
    else
      TL = D->getTypeSourceInfo()->getTypeLoc();
  }
  ReplaceTypeInDecl(const FieldDecl *FD, std::string &&T)
      : TextModification(TMID::ReplaceTypeInDecl), D(nullptr), FD(FD), T(T) {
    if (FD->getType()->isArrayType())
      TL = FD->getTypeSourceInfo()
               ->getTypeLoc()
               .getAs<ArrayTypeLoc>()
               .getElementLoc();
    else
      TL = FD->getTypeSourceInfo()->getTypeLoc();
  }
  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

// Replace type in var. declaration.
class InsertNameSpaceInVarDecl : public TextModification {
  const VarDecl *D;
  std::string T;
  unsigned int InsertPosition;

public:
  InsertNameSpaceInVarDecl(const VarDecl *D, std::string &&T,
                           unsigned int InsertPosition = 0)
      : TextModification(TMID::InsertNameSpaceInVarDecl), D(D), T(T),
        InsertPosition(InsertPosition) {}
  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

// Replace type in var. declaration.
class InsertNameSpaceInCastExpr : public TextModification {
  const CStyleCastExpr *D;
  std::string T;

public:
  InsertNameSpaceInCastExpr(const CStyleCastExpr *D, std::string &&T)
      : TextModification(TMID::InsertNameSpaceInCastExpr), D(D), T(T) {}
  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

// Replace type in var. declaration.
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
  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;

private:
  void addVarDecl(const VarDecl *D, std::string &&Text);
  static std::map<unsigned, ReplaceVarDecl *> ReplaceMap;
};

// Replace return type in function declaration.
class ReplaceReturnType : public TextModification {
  const FunctionDecl *FD;
  std::string T;

public:
  ReplaceReturnType(const FunctionDecl *FD, std::string &&T)
      : TextModification(TMID::ReplaceReturnType), FD(FD), T(T) {}
  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

// Rename field in expression.
class RenameFieldInMemberExpr : public TextModification {
  const MemberExpr *ME;
  std::string T;
  unsigned PositionOfDot;

public:
  RenameFieldInMemberExpr(const MemberExpr *ME, std::string &&T,
                          unsigned PositionOfDot = 0)
      : TextModification(TMID::RenameFieldInMemberExpr, G1), ME(ME), T(T),
        PositionOfDot(PositionOfDot) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

class InsertAfterStmt : public TextModification {
  const Stmt *S;
  std::string T;

public:
  InsertAfterStmt(const Stmt *S, std::string &&T)
      : TextModification(TMID::InsertAfterStmt), S(S), T(T) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

// Insert '/*  */' C style multi line comment
class InsertComment : public TextModification {
  // The comment will be inserted at this position
  SourceLocation SL;
  std::string Text;

public:
  InsertComment(SourceLocation SL, std::string Text)
      : TextModification(TMID::InsertComment), SL(SL), Text(Text) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// Replace CallExpr with another call.
// TODO: return values are not handled.
// TODO: we probably need more genric class, which would take the list of
// strings and expressions and compose them to a single srting, also doing look
// up for already modified expressions and use their new spelling when needed.
class ReplaceCallExpr : public TextModification {
  // Call to replace.
  const CallExpr *C;
  // New function name.
  std::string Name;
  // New function params.
  std::vector<const Expr *> Args;
  // New function type.
  std::vector<std::string> Types;

public:
  ReplaceCallExpr(const CallExpr *Call, std::string &&NewName,
                  std::vector<const Expr *> &&NewArgs)
      : TextModification(TMID::ReplaceCallExpr), C(Call), Name(NewName),
        Args(NewArgs) {}

  ReplaceCallExpr(const CallExpr *Call, std::string &&NewName,
                  std::vector<const Expr *> &&NewArgs,
                  std::vector<std::string> NewTypes)
      : TextModification(TMID::ReplaceCallExpr), C(Call), Name(NewName),
        Args(NewArgs), Types(NewTypes) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

class InsertArgument : public TextModification {
  const FunctionDecl *FD;
  // Argument string without comma.
  std::string ArgName;
  bool Lazy = false;

public:
  InsertArgument(const FunctionDecl *FD, std::string &&ArgName)
      : TextModification(TMID::InsertArgument), FD(FD), ArgName(ArgName) {}
  InsertArgument(const FunctionDecl *FD, std::string &&ArgName, bool Lazy)
      : TextModification(TMID::InsertArgument), FD(FD), ArgName(ArgName),
        Lazy(Lazy) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

class InsertCallArgument : public TextModification {
  const CallExpr *CE;
  // Argument string without comma;
  std::string Arg;

public:
  InsertCallArgument(const CallExpr *CE, std::string &&Arg)
      : TextModification(TMID::InsertCallArgument), CE(CE), Arg(Arg) {}
  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

class ReplaceInclude : public TextModification {
  CharSourceRange Range;
  std::string T;

public:
  ReplaceInclude(CharSourceRange Range, std::string &&T)
      : TextModification(TMID::ReplaceInclude), Range(Range), T(T) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

class ReplaceDim3Ctor : public TextModification {
  bool isDecl;
  const CXXConstructExpr *Ctor;
  const CXXConstructExpr *FinalCtor;
  StmtStringMap *SSM;
  CharSourceRange CSR;
  mutable std::string ReplacementString;

  void setRange();
  const Stmt *getReplaceStmt(const Stmt *S) const;
  std::string getSyclRangeCtor(const CXXConstructExpr *Ctor,
                               const ASTContext &Context) const;
  std::string getParamsString(const CXXConstructExpr *Ctor,
                              const ASTContext &Context) const;
  std::string getReplaceString(const ASTContext &Context) const;

public:
  ReplaceDim3Ctor(const CXXConstructExpr *_Ctor, StmtStringMap *_SSM,
                  bool _isDecl = false)
      : TextModification(TMID::ReplaceDim3Ctor, G2), isDecl(_isDecl),
        Ctor(_Ctor), FinalCtor(nullptr), SSM(_SSM) {
    setRange();
  }
  ReplaceDim3Ctor(const CXXConstructExpr *_Ctor, StmtStringMap *_SSM,
                  const CXXConstructExpr *_FinalCtor)
      : TextModification(TMID::ReplaceDim3Ctor, G2), isDecl(false), Ctor(_Ctor),
        FinalCtor(_FinalCtor), SSM(_SSM) {
    setRange();
  }
  static const CXXConstructExpr *getConstructExpr(const Expr *E);
  ReplaceInclude *getEmpty();
  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

class ReplaceKernelCallExpr : public TextModification {
  std::shared_ptr<KernelCallExpr> Kernel;
  StmtStringMap *SSM;

public:
  ReplaceKernelCallExpr(std::shared_ptr<KernelCallExpr> Kernel);
  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

/// A class that filters out Replacements that modify text inside a deleted code
/// block.
class ReplacementFilter {
  struct Interval {
    size_t Offset;
    size_t Length;
    bool operator<(const Interval &Other) const {
      return Offset < Other.Offset;
    }
  };

  using IntervalSet = std::vector<Interval>;

  const std::vector<ExtReplacement> &ReplSet;
  std::map<std::string, IntervalSet> FileMap;

private:
  bool containsInterval(const IntervalSet &IS, const Interval &I) const;
  bool isDeletedReplacement(const ExtReplacement &R) const;
  size_t findFirstNotDeletedReplacement(size_t Start) const;

  class iterator {
    const ReplacementFilter &RF;
    size_t Idx;

  public:
    iterator(const ReplacementFilter &RF, size_t Idx) : RF(RF), Idx(Idx) {}
    const ExtReplacement &operator*() const { return RF.ReplSet[Idx]; }
    iterator &operator++() {
      Idx = RF.findFirstNotDeletedReplacement(Idx + 1);
      return *this;
    }
    bool operator==(const iterator &Other) const {
      assert(&RF == &Other.RF && "Mismatching iterators");
      return Idx == Other.Idx;
    }
    bool operator!=(const iterator &Other) const { return !operator==(Other); }
  };

public:
  ReplacementFilter(const std::vector<ExtReplacement> &RS);

  iterator begin() {
    return iterator(*this, findFirstNotDeletedReplacement(0));
  }
  iterator end() { return iterator(*this, -1); }
};

class InsertBeforeStmt : public TextModification {
  const Stmt *S;
  std::string T;

public:
  InsertBeforeStmt(const Stmt *S, std::string &&T)
      : TextModification(TMID::InsertBeforeStmt), S(S), T(T) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

class RemoveArg : public TextModification {
  const CallExpr *CE;
  const unsigned N;

public:
  RemoveArg(const CallExpr *CE, const unsigned N)
      : TextModification(TMID::RemoveArg), CE(CE), N(N) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

class InsertBeforeCtrInitList : public TextModification {
  const CXXConstructorDecl *CDecl;
  std::string T;

  SourceLocation getInsertLoc() const;

public:
  InsertBeforeCtrInitList(const CXXConstructorDecl *S, std::string &&T)
      : TextModification(TMID::InsertBeforeCtrInitList), CDecl(S), T(T) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

class InsertClassName : public TextModification {
  const CXXRecordDecl *CD;
  static unsigned Count;

public:
  InsertClassName(const CXXRecordDecl *C)
      : TextModification(TMID::InsertClassName), CD(C) {}

  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

class ReplaceText : public TextModification {
  SourceLocation BeginLoc;
  unsigned Len;
  StringRef T;

public:
  ReplaceText(const SourceLocation &Begin, unsigned Len, std::string &&S)
      : TextModification(TMID::ReplaceText), BeginLoc(Begin), Len(Len), T(S) {}
  ExtReplacement getReplacement(const ASTContext &Context) const override;
  void print(llvm::raw_ostream &OS, ASTContext &Context,
             const bool PrintDetail = true) const override;
};

} // namespace syclct
} // namespace clang

#endif // SYCLCT_TEXT_MODIFICATION_H
