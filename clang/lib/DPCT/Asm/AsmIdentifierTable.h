//===------------------ AsmIdentifierTable.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DPCT_ASM_IDENTIFIER_TABLE_H
#define CLANG_DPCT_ASM_IDENTIFIER_TABLE_H

#include "AsmTokenKinds.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Allocator.h"

namespace clang::dpct {

class InlineAsmIdentifierInfo {
  friend class InlineAsmIdentifierTable;

  // Front-end token ID or tok::identifier.
  unsigned TokenID;

  llvm::StringMapEntry<InlineAsmIdentifierInfo *> *Entry = nullptr;

  InlineAsmIdentifierInfo() : TokenID(asmtok::identifier) {}

public:
  InlineAsmIdentifierInfo(const InlineAsmIdentifierInfo &) = delete;
  InlineAsmIdentifierInfo &operator=(const InlineAsmIdentifierInfo &) = delete;
  InlineAsmIdentifierInfo(InlineAsmIdentifierInfo &&) = delete;
  InlineAsmIdentifierInfo &operator=(InlineAsmIdentifierInfo &&) = delete;

  /// Return true if this is the identifier for the specified string.
  ///
  /// This is intended to be used for string literals only: II->isStr("foo").
  template <std::size_t StrLen> bool isStr(const char (&Str)[StrLen]) const {
    return getLength() == StrLen - 1 &&
           memcmp(getNameStart(), Str, StrLen - 1) == 0;
  }

  /// Return true if this is the identifier for the specified StringRef.
  bool isStr(llvm::StringRef Str) const {
    llvm::StringRef ThisStr(getNameStart(), getLength());
    return ThisStr == Str;
  }

  /// Return the beginning of the actual null-terminated string for this
  /// identifier.
  const char *getNameStart() const { return Entry->getKeyData(); }

  /// Efficiently return the length of this identifier info.
  unsigned getLength() const { return Entry->getKeyLength(); }

  /// Return the actual identifier string.
  StringRef getName() const { return StringRef(getNameStart(), getLength()); }

  /// If this is a source-language token (e.g. 'for'), this API
  /// can be used to cause the lexer to map identifiers to source-language
  /// tokens.
  asmtok::TokenKind getTokenID() const {
    return static_cast<asmtok::TokenKind>(TokenID);
  }

  /// Return true if this token is a keyword in asm.
  bool isInstruction() const;

  /// Return true if this token is a builtin type in asm.
  bool isBuiltinType() const;

  /// Provide less than operator for lexicographical sorting.
  bool operator<(const InlineAsmIdentifierInfo &RHS) const {
    return getName() < RHS.getName();
  }
};

class InlineAsmIdentifierInfoLookup {
public:
  virtual ~InlineAsmIdentifierInfoLookup();
  virtual InlineAsmIdentifierInfo *get(StringRef Name) = 0;
};

class InlineAsmIdentifierTable {
  using HashTableTy =
      llvm::StringMap<InlineAsmIdentifierInfo *, llvm::BumpPtrAllocator>;
  HashTableTy HashTable;

  InlineAsmIdentifierInfoLookup *ExternalLookup;

public:
  /// Create the identifier table.
  explicit InlineAsmIdentifierTable(
      InlineAsmIdentifierInfoLookup *ExternalLookup = nullptr);

  /// Set the external identifier lookup mechanism.
  void setExternalIdentifierLookup(InlineAsmIdentifierInfoLookup *IILookup) {
    ExternalLookup = IILookup;
  }

  /// Retrieve the external identifier lookup object, if any.
  InlineAsmIdentifierInfoLookup *getExternalIdentifierLookup() const {
    return ExternalLookup;
  }

  llvm::BumpPtrAllocator &getAllocator() { return HashTable.getAllocator(); }

  /// Return the identifier token info for the specified named
  /// identifier.
  InlineAsmIdentifierInfo &get(StringRef Name) {
    auto &Entry = *HashTable.try_emplace(Name, nullptr).first;

    InlineAsmIdentifierInfo *&II = Entry.second;
    if (II)
      return *II;

    // No entry; if we have an external lookup, look there first.
    if (ExternalLookup) {
      II = ExternalLookup->get(Name);
      if (II)
        return *II;
    }

    // Lookups failed, make a new IdentifierInfo.
    void *Mem = getAllocator().Allocate<InlineAsmIdentifierInfo>();
    II = new (Mem) InlineAsmIdentifierInfo();

    // Make sure getName() knows how to find the IdentifierInfo
    // contents.
    II->Entry = &Entry;

    return *II;
  }

  InlineAsmIdentifierInfo &get(StringRef Name, asmtok::TokenKind TokenCode) {
    InlineAsmIdentifierInfo &II = get(Name);
    II.TokenID = TokenCode;
    assert(II.TokenID == (unsigned)TokenCode && "TokenCode too large");
    return II;
  }

  using iterator = HashTableTy::const_iterator;
  using const_iterator = HashTableTy::const_iterator;

  iterator begin() const { return HashTable.begin(); }
  iterator end() const { return HashTable.end(); }
  unsigned size() const { return HashTable.size(); }

  iterator find(StringRef Name) const { return HashTable.find(Name); }

  bool contains(StringRef Name) const { return HashTable.contains(Name); }

  /// Populate the identifier table with info about the asm keywords.
  void AddKeywords();
};

} // namespace clang::dpct

namespace llvm {

// Provide PointerLikeTypeTraits for IdentifierInfo pointers, which
// are not guaranteed to be 8-byte aligned.
template <>
struct PointerLikeTypeTraits<clang::dpct::InlineAsmIdentifierInfo *> {
  static void *getAsVoidPointer(clang::dpct::InlineAsmIdentifierInfo *P) {
    return P;
  }

  static clang::dpct::InlineAsmIdentifierInfo *getFromVoidPointer(void *P) {
    return static_cast<clang::dpct::InlineAsmIdentifierInfo *>(P);
  }

  static constexpr int NumLowBitsAvailable = 1;
};

template <>
struct PointerLikeTypeTraits<const clang::dpct::InlineAsmIdentifierInfo *> {
  static const void *
  getAsVoidPointer(const clang::dpct::InlineAsmIdentifierInfo *P) {
    return P;
  }

  static const clang::dpct::InlineAsmIdentifierInfo *
  getFromVoidPointer(const void *P) {
    return static_cast<const clang::dpct::InlineAsmIdentifierInfo *>(P);
  }

  static constexpr int NumLowBitsAvailable = 1;
};

} // namespace llvm

#endif // CLANG_DPCT_ASM_IDENTIFIER_TABLE_H
