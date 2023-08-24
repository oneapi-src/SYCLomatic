//===--------------- Rules.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Rules.h"
#include "ASTTraversal.h"
#include "CallExprRewriter.h"
#include "Error.h"
#include "MapNames.h"
#include "MigrationRuleManager.h"
#include "Utility.h"
#include "llvm/Support/YAMLTraits.h"
#include "NCCLAPIMigration.h"

#include <optional>
#include <variant>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

std::vector<std::string> MetaRuleObject::RuleFiles;
std::vector<std::shared_ptr<MetaRuleObject>> MetaRules;

template <class Functor>
void reisterMigrationRule(const std::string &Name, Functor F) {
  class UserDefinedRuleFactory : public clang::dpct::MigrationRuleFactoryBase {
    Functor F;

  public:
    UserDefinedRuleFactory(Functor Func) : F(std::move(Func)) {}
    std::unique_ptr<clang::dpct::MigrationRule>
    createMigrationRule() const override {
      return F();
    }
  };
  clang::dpct::MigrationRuleManager::registerRule(
      clang::dpct::PassKind::PK_Migration, Name,
      std::make_shared<UserDefinedRuleFactory>(std::move(F)));
}

void registerMacroRule(MetaRuleObject &R) {
  auto It = MapNames::MacroRuleMap.find(R.In);
  if (It != MapNames::MacroRuleMap.end()) {
    if (It->second.Priority > R.Priority) {
      It->second.Id = R.RuleId;
      It->second.Priority = R.Priority;
      It->second.In = R.In;
      It->second.Out = R.Out;
      It->second.HelperFeature =
          clang::dpct::HelperFeatureEnum::none;
      It->second.Includes = R.Includes;
    }
  } else {
    MapNames::MacroRuleMap.emplace(
        R.In,
        MacroMigrationRule(R.RuleId, R.Priority, R.In, R.Out,
                           clang::dpct::HelperFeatureEnum::none,
                           R.Includes));
  }
}

void registerAPIRule(MetaRuleObject &R) {
  using namespace clang::dpct;
  // register rule
  reisterMigrationRule(R.RuleId, [=] {
    return std::make_unique<clang::dpct::UserDefinedAPIRule>(
        R.In, R.RuleAttributes.HasExplicitTemplateArgs);
  });
  // create and register rewriter
  // RewriterMap contains entries like {"FunctionName", RewriterFactory}
  // The priority of all the default rules are RulePriority::Fallback and
  // recorded in the RewriterFactory.
  // Search for existing rules of the same function name
  // if there is existing rule,
  //   compare the priority and decide whether to add the rule into the
  //   RewriterMap
  // if there is no existing rule,
  //   add the new rule to the RewriterMap
  auto Factory = createUserDefinedRewriterFactory(R.In, R);
  auto &Entry = (*CallExprRewriterFactoryBase::RewriterMap)[R.In];
  if (!Entry) {
    Entry = Factory;
  } else if (R.RuleAttributes.HasExplicitTemplateArgs) {
    Entry = std::make_shared<ConditionalRewriterFactory>(
        UserDefinedRewriterFactory::hasExplicitTemplateArgs, Factory, Entry);
  } else if (Entry->Priority > R.Priority) {
    Entry = Factory;
  }
}

void registerHeaderRule(MetaRuleObject &R) {
  auto It = MapNames::HeaderRuleMap.find(R.In);
  if (It != MapNames::HeaderRuleMap.end()) {
    if (It->second.Priority > R.Priority) {
      It->second = R;
    }
  } else {
    MapNames::HeaderRuleMap.emplace(R.In, R);
  }
}

void registerTypeRule(MetaRuleObject &R) {
  auto It = MapNames::TypeNamesMap.find(R.In);
  if (It != MapNames::TypeNamesMap.end()) {
    if (It->second->Priority > R.Priority) {
      It->second->NewName = R.Out;
      It->second->Priority = R.Priority;
      It->second->RequestFeature =
          clang::dpct::HelperFeatureEnum::none;
      It->second->Includes.insert(It->second->Includes.end(),
                                  R.Includes.begin(), R.Includes.end());
    }
  } else {
    reisterMigrationRule(R.RuleId, [=] {
      return std::make_unique<clang::dpct::UserDefinedTypeRule>(R.In);
    });
    auto RulePtr = std::make_shared<TypeNameRule>(
        R.Out, clang::dpct::HelperFeatureEnum::none, R.Priority);
    RulePtr->Includes.insert(RulePtr->Includes.end(), R.Includes.begin(),
                             R.Includes.end());
    MapNames::TypeNamesMap.emplace(R.In, RulePtr);
  }
}

void registerClassRule(MetaRuleObject &R) {
  // register class name migration rule
  registerTypeRule(R);
  // register all field rules
  for (auto ItField = R.Fields.begin(); ItField != R.Fields.end(); ItField++) {
    std::string BaseAndFieldName = R.In + "." + (*ItField)->In;
    auto ItFieldRule = MapNames::ClassFieldMap.find(BaseAndFieldName);
    if (ItFieldRule != MapNames::ClassFieldMap.end()) {
      if (ItFieldRule->second->Priority > R.Priority) {
        if((*ItField)->OutGetter != ""){
          ItFieldRule->second->SetterName = (*ItField)->OutSetter;
          ItFieldRule->second->GetterName = (*ItField)->OutGetter;
          ItFieldRule->second->NewName = "";
        } else {
          ItFieldRule->second->SetterName = "";
          ItFieldRule->second->GetterName = "";
          ItFieldRule->second->NewName = (*ItField)->Out;
        }
        ItFieldRule->second->Priority = R.Priority;
        ItFieldRule->second->RequestFeature =
            clang::dpct::HelperFeatureEnum::none;
        ItFieldRule->second->Includes.clear();
        ItFieldRule->second->Includes.insert(
            ItFieldRule->second->Includes.end(), R.Includes.begin(),
            R.Includes.end());
      }
    } else {
      reisterMigrationRule(BaseAndFieldName, [=] {
        return std::make_unique<clang::dpct::UserDefinedClassFieldRule>(
            R.In, (*ItField)->In);
      });
      std::shared_ptr<ClassFieldRule> RulePtr;
      if ((*ItField)->OutGetter != "") {
        RulePtr = std::make_shared<ClassFieldRule>(
            (*ItField)->OutSetter, (*ItField)->OutGetter,
            clang::dpct::HelperFeatureEnum::none, R.Priority);
      } else {
        RulePtr = std::make_shared<ClassFieldRule>(
            (*ItField)->Out, clang::dpct::HelperFeatureEnum::none,
            R.Priority);
      }
      RulePtr->Includes.insert(RulePtr->Includes.end(), R.Includes.begin(),
                               R.Includes.end());
      MapNames::ClassFieldMap.emplace(BaseAndFieldName, RulePtr);
    }
  }
  // register all method rules
  for (auto ItMethod = R.Methods.begin(); ItMethod != R.Methods.end();
       ItMethod++) {
    std::string BaseAndMethodName = R.In + "." + (*ItMethod)->In;
    reisterMigrationRule(BaseAndMethodName, [=] {
      return std::make_unique<clang::dpct::UserDefinedClassMethodRule>(R.In,
                                                             (*ItMethod)->In);
    });

    auto ItMethodRule =
        clang::dpct::CallExprRewriterFactoryBase::MethodRewriterMap->find(
            BaseAndMethodName);
    if (ItMethodRule ==
        clang::dpct::CallExprRewriterFactoryBase::MethodRewriterMap->end()) {
      clang::dpct::CallExprRewriterFactoryBase::MethodRewriterMap->emplace(
          BaseAndMethodName,
          clang::dpct::createUserDefinedMethodRewriterFactory(R.In, R,
                                                              *ItMethod));
    } else if (ItMethodRule->second->Priority > R.Priority) {
      (*clang::dpct::CallExprRewriterFactoryBase::MethodRewriterMap)
          [BaseAndMethodName] =
              clang::dpct::createUserDefinedMethodRewriterFactory(R.In, R,
                                                                  *ItMethod);
    }
  }
}

void registerEnumRule(MetaRuleObject &R) {
  auto It = clang::dpct::EnumConstantRule::EnumNamesMap.find(R.In);
  if (It != clang::dpct::EnumConstantRule::EnumNamesMap.end()) {
    if (It->second->Priority > R.Priority) {
      It->second->Priority = R.Priority;
      It->second->NewName = R.Out;
      It->second->RequestFeature =
          clang::dpct::HelperFeatureEnum::none;
      It->second->Includes.insert(It->second->Includes.end(),
                                 R.Includes.begin(), R.Includes.end());
    }
  } else {
    if(R.EnumName == ""){
      return;
    }
    reisterMigrationRule(R.RuleId, [=] {
      return std::make_unique<clang::dpct::UserDefinedEnumRule>(R.EnumName);
    });
    auto RulePtr = std::make_shared<EnumNameRule>(
        R.Out, clang::dpct::HelperFeatureEnum::none, R.Priority);
    RulePtr->Includes.insert(RulePtr->Includes.end(), R.Includes.begin(),
                             R.Includes.end());
    clang::dpct::EnumConstantRule::EnumNamesMap.emplace(
        R.EnumName + "::" + R.In, RulePtr);
  }
}

void deregisterAPIRule(MetaRuleObject &R) {
  using namespace clang::dpct;
  CallExprRewriterFactoryBase::RewriterMap->erase(R.In);
}

void registerPatternRewriterRule(MetaRuleObject &R) {
  MapNames::PatternRewriters.emplace_back(
      MetaRuleObject::PatternRewriter(R.In, R.Out, R.Subrules));
}

void importRules(llvm::cl::list<std::string> &RuleFiles) {
  for (auto &RuleFile : RuleFiles) {
    makeCanonical(RuleFile);
    // open the yaml file
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
        llvm::MemoryBuffer::getFile(RuleFile);
    if (!Buffer) {
      llvm::errs() << "Error: failed to read " << RuleFile << ": "
                   << Buffer.getError().message() << "\n";
      clang::dpct::ShowStatus(MigrationErrorInvalidRuleFilePath);
      dpctExit(MigrationErrorInvalidRuleFilePath);
    }

    // load rules
    std::vector<std::shared_ptr<MetaRuleObject>> CurrentRules;
    llvm::yaml::Input YAMLIn(Buffer.get()->getBuffer());
    YAMLIn >> CurrentRules;
    // store the rules, also prevent MetaRuleObjects from being destructed
    MetaRules.insert(MetaRules.end(), CurrentRules.begin(), CurrentRules.end());

    if (YAMLIn.error()) {
      // yaml parsing fail
      clang::dpct::ShowStatus(MigrationErrorCannotParseRuleFile);
      dpctExit(MigrationErrorCannotParseRuleFile);
    }

    MetaRuleObject::setRuleFiles(RuleFile);

    // Register Rules
    for (std::shared_ptr<MetaRuleObject> &r : CurrentRules) {
      r->RuleFile = RuleFile;
      switch (r->Kind) {
      case (RuleKind::Macro):
        registerMacroRule(*r);
        break;
      case (RuleKind::API):
        registerAPIRule(*r);
        break;
      case (RuleKind::Header):
        registerHeaderRule(*r);
        break;
      case (RuleKind::TypeRule):
        registerTypeRule(*r);
        break;
      case (RuleKind::Class):
        registerClassRule(*r);
        break;
      case (RuleKind::Enum):
        registerEnumRule(*r);
        break;
      case (RuleKind::DisableAPIMigration):
        deregisterAPIRule(*r);
        break;
      case (RuleKind::PatternRewriter):
        registerPatternRewriterRule(*r);
        break;
      default:
        break;
      }
    }
  }
}

// /RuleOutputString is the string specified in rule's "Out" session
void OutputBuilder::parse(std::string &RuleOutputString) {
  size_t i, StrStartIdx = 0;
  for (i = 0; i < RuleOutputString.length(); i++) {
    switch (RuleOutputString[i]) {
    case '\\': {
      // save previous string
      auto StringBuilder = std::make_shared<OutputBuilder>();
      StringBuilder->Kind = Kind::String;
      StringBuilder->Str =
          RuleOutputString.substr(StrStartIdx, i - StrStartIdx);
      SubBuilders.push_back(StringBuilder);
      // skip "/" and set the begin of the next string
      i++;
      StrStartIdx = i;
      break;
    }
    case '$': {
      // save previous string
      auto StringBuilder = std::make_shared<OutputBuilder>();
      StringBuilder->Kind = Kind::String;
      StringBuilder->Str =
          RuleOutputString.substr(StrStartIdx, i - StrStartIdx);
      SubBuilders.push_back(StringBuilder);
      SubBuilders.push_back(consumeKeyword(RuleOutputString, i));
      StrStartIdx = i;
    } break;
    default:
      break;
    }
  }
  if (i > StrStartIdx) {
    auto StringBuilder = std::make_shared<OutputBuilder>();
    StringBuilder->Kind = Kind::String;
    StringBuilder->Str = RuleOutputString.substr(StrStartIdx, i - StrStartIdx);
    SubBuilders.push_back(StringBuilder);
  }
}

// /OutStr is the string specified in rule's "Out" session
void OutputBuilder::ignoreWhitespaces(std::string &OutStr, size_t &Idx) {
  for (; Idx < OutStr.size(); Idx++) {
    if (OutStr[Idx] != ' ' && OutStr[Idx] != '\r' && OutStr[Idx] != '\n' &&
        OutStr[Idx] != '\t') {
      return;
    }
  }
}

// /OutStr is the string specified in rule's "Out" session
void OutputBuilder::consumeRParen(std::string &OutStr, size_t &Idx,
                                  std::string &&Keyword) {
  ignoreWhitespaces(OutStr, Idx);
  if (Idx >= OutStr.size() || OutStr[Idx] != ')') {
    llvm::errs() << RuleFile << ":Error: in rule " << RuleName
                 << ", ')' is expected after " << Keyword << "\n";
    clang::dpct::ShowStatus(MigrationErrorCannotParseRuleFile);
    dpctExit(MigrationErrorCannotParseRuleFile);
  }
  Idx++;
}

// /OutStr is the string specified in rule's "Out" session
void OutputBuilder::consumeLParen(std::string &OutStr, size_t &Idx,
                                  std::string &&Keyword) {
  ignoreWhitespaces(OutStr, Idx);
  if (Idx >= OutStr.size() || OutStr[Idx] != '(') {
    llvm::errs() << RuleFile << ":Error: in rule " << RuleName
      << ", '(' is expected after " << Keyword << "\n";
    clang::dpct::ShowStatus(MigrationErrorCannotParseRuleFile);
    dpctExit(MigrationErrorCannotParseRuleFile);
  }
  Idx++;
}

// /OutStr is the string specified in rule's "Out" session
int OutputBuilder::consumeArgIndex(std::string &OutStr, size_t &Idx,
                                   std::string &&Keyword) {
  ignoreWhitespaces(OutStr, Idx);

  if (Idx >= OutStr.size() || OutStr[Idx] != '$') {
    llvm::errs() << RuleFile << ":Error: in rule " << RuleName
                 << ", a positive integer is expected after "
                 << Keyword << "\n";
    clang::dpct::ShowStatus(MigrationErrorCannotParseRuleFile);
    dpctExit(MigrationErrorCannotParseRuleFile);
  }

  // consume $
  Idx++;
  if (Idx >= OutStr.size()) {
    llvm::errs() << RuleFile << ":Error: in rule " << RuleName
                 << ", a positive integer is expected after "
                 << Keyword << "\n";
    clang::dpct::ShowStatus(MigrationErrorCannotParseRuleFile);
    dpctExit(MigrationErrorCannotParseRuleFile);
  }

  ignoreWhitespaces(OutStr, Idx);
  int ArgIndex = 0;

  // process arg number
  unsigned i = Idx;
  for (; i < OutStr.size(); i++) {
    if (!std::isdigit(OutStr[i])) {
      if (i == Idx) {
        // report unknown KW
        llvm::errs() << RuleFile << ":Error: in rule " << RuleName
                     << ", unknown keyword: $" << OutStr.substr(Idx, 10)
                     << "\n";
        clang::dpct::ShowStatus(MigrationErrorCannotParseRuleFile);
        dpctExit(MigrationErrorCannotParseRuleFile);
      } else {
        break;
      }
    } else {
      ArgIndex = ArgIndex * 10 + (int)OutStr[i] - 48;
      if (ArgIndex < 0) {
        llvm::errs() << RuleFile << ":Error: in rule " << RuleName
                     << ", argument index out of range.\n";
        clang::dpct::ShowStatus(MigrationErrorCannotParseRuleFile);
        dpctExit(MigrationErrorCannotParseRuleFile);
      }
    }
  }
  Idx = i;

  if (Idx >= OutStr.size()) {
    llvm::errs() << RuleFile << ":Error: in rule " << RuleName
                 << ", a positive integer is expected after " << Keyword
                 << "\n";
    clang::dpct::ShowStatus(MigrationErrorCannotParseRuleFile);
    dpctExit(MigrationErrorCannotParseRuleFile);
  }

  // Adjust the index because the arg index in rules starts from $1,
  // and the arg index starts from 0 in CallExpr.
  return ArgIndex - 1;
}

// /OutStr is the string specified in rule's "Out" session
std::shared_ptr<OutputBuilder>
OutputBuilder::consumeKeyword(std::string &OutStr, size_t &Idx) {
  auto ResultBuilder = std::make_shared<OutputBuilder>();
  if (OutStr.substr(Idx, 6) == "$queue") {
    Idx += 6;
    ResultBuilder->Kind = Kind::Queue;
  } else if (OutStr.substr(Idx, 8) == "$context") {
    Idx += 8;
    ResultBuilder->Kind = Kind::Context;
  } else if (OutStr.substr(Idx, 7) == "$device") {
    Idx += 7;
    ResultBuilder->Kind = Kind::Device;
  } else if (OutStr.substr(Idx, 13) == "$type_name_of") {
    Idx += 13;
    consumeLParen(OutStr, Idx, "$type_name_of");
    ResultBuilder->Kind = Kind::TypeName;
    ResultBuilder->ArgIndex = consumeArgIndex(OutStr, Idx, "$type_name_of");
    consumeRParen(OutStr, Idx, "$type_name_of");
  } else if (OutStr.substr(Idx, 8) == "$addr_of") {
    Idx += 8;
    consumeLParen(OutStr, Idx, "$addr_of");
    ResultBuilder->Kind = Kind::AddrOf;
    ResultBuilder->ArgIndex = consumeArgIndex(OutStr, Idx, "$addr_of");
    consumeRParen(OutStr, Idx, "$addr_of");
  } else if (OutStr.substr(Idx, 11) == "$deref_type") {
    Idx += 11;
    consumeLParen(OutStr, Idx, "$deref_type");
    ResultBuilder->Kind = Kind::DerefedTypeName;
    ResultBuilder->ArgIndex = consumeArgIndex(OutStr, Idx, "$deref_type");
    consumeRParen(OutStr, Idx, "$deref_type");
  } else if (OutStr.substr(Idx, 6) == "$deref") {
    Idx += 6;
    consumeLParen(OutStr, Idx, "$deref");
    ResultBuilder->Kind = Kind::Deref;
    ResultBuilder->ArgIndex = consumeArgIndex(OutStr, Idx, "$deref");
    consumeRParen(OutStr, Idx, "$deref");
  } else {
    ResultBuilder->Kind = Kind::Arg;
    ResultBuilder->ArgIndex = consumeArgIndex(OutStr, Idx, "$");
  }
  return ResultBuilder;
}

class RefMatcherInterface
    : public clang::ast_matchers::internal::MatcherInterface<
          clang::DeclRefExpr> {
  using StringRef = llvm::StringRef;
  StringRef Name;
  bool HasExplicitTemplateArgs;
  bool HasQualifier = true;

  static bool consumeSuffix(StringRef &RefName, StringRef InputName) {
    if (InputName.startswith("::"))
      InputName = InputName.drop_front(2);

    if (!RefName.endswith(InputName))
      return false;

    RefName = RefName.drop_back(InputName.size());
    if (!RefName.empty())
      return RefName.endswith("::");

    return true;
  }

  bool matchName(const clang::DeclRefExpr &Node,
                 clang::ASTContext &Context) const {
    if (!HasQualifier) {
      if (auto FD = clang::dyn_cast<clang::FunctionDecl>(Node.getDecl())) {
        std::string NS = "";
        if (Node.getQualifier()) {
          NS = getNestedNameSpecifierString(Node.getQualifier());
        }
        if (auto ID = FD->getIdentifier()) {
          return NS + ID->getName().str() == Name;
        }
      }
    } else if (auto Qualifier = Node.getQualifier()) {
      auto RefName = Name;
      llvm::SmallString<256> InputName;
      llvm::raw_svector_ostream OS(InputName);
      auto PP = Context.getPrintingPolicy();
      Node.getNameInfo().printName(OS, PP);
      if (consumeSuffix(RefName, InputName.str())) {
        InputName.clear();
        Qualifier->print(OS, PP);
        return consumeSuffix(RefName, InputName.str()) && RefName.empty();
      }
    }
    return false;
  }

public:
  RefMatcherInterface(StringRef APIName, bool HasAnyExplicitTemplateArgs)
      : Name(APIName), HasExplicitTemplateArgs(HasAnyExplicitTemplateArgs) {
    if (Name.startswith("::"))
      Name = Name.drop_front(2);

    HasQualifier = Name.find("::") != StringRef::npos;
  }
  bool matches(const clang::DeclRefExpr &Node,
               ::clang::ast_matchers::internal::ASTMatchFinder *Finder,
               ::clang::ast_matchers::internal::BoundNodesTreeBuilder *Builder)
      const override {
    return (!Node.hasExplicitTemplateArgs() || HasExplicitTemplateArgs) &&
           matchName(Node, Finder->getASTContext());
  }
};

void clang::dpct::UserDefinedAPIRule::registerMatcher(
    clang::ast_matchers::MatchFinder &MF) {
  MF.addMatcher(callExpr(callee(expr(ignoringImpCasts(declRefExpr(
                             clang::ast_matchers::internal::makeMatcher(
                                 new RefMatcherInterface(
                                     APIName, HasExplicitTemplateArgs)))))))
                    .bind("call"),
                this);
}

void clang::dpct::UserDefinedAPIRule::runRule(
    const clang::ast_matchers::MatchFinder::MatchResult &Result) {
  if (const CallExpr *CE = getAssistNodeAsType<CallExpr>(Result, "call")) {
    dpct::ExprAnalysis EA;
    EA.analyze(CE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

void clang::dpct::UserDefinedTypeRule::registerMatcher(
    clang::ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
      typeLoc(loc(qualType(hasDeclaration(namedDecl(hasName(TypeName))))))
          .bind("typeLoc"),
      this);
}

void clang::dpct::UserDefinedTypeRule::runRule(
    const clang::ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto TL = getNodeAsType<TypeLoc>(Result, "typeLoc")) {
    auto TypeStr =
        DpctGlobalInfo::getTypeName(TL->getType().getUnqualifiedType());
    // if the TypeLoc is a TemplateSpecializationTypeLoc
    // the TypeStr should be the substr before the "<"
    if (auto TSTL = TL->getAsAdjusted<TemplateSpecializationTypeLoc>()) {
      TypeStr = TypeStr.substr(0, TypeStr.find("<"));
    }
    auto It = MapNames::TypeNamesMap.find(TypeStr);
    if (It == MapNames::TypeNamesMap.end())
      return;

    auto ReplStr = It->second->NewName;

    auto &SM = DpctGlobalInfo::getSourceManager();
    auto Range = getDefinitionRange(TL->getBeginLoc(), TL->getEndLoc());
    auto Len = Lexer::MeasureTokenLength(
        Range.getEnd(), SM, DpctGlobalInfo::getContext().getLangOpts());
    if (auto TSTL = TL->getAsAdjusted<TemplateSpecializationTypeLoc>()) {
      Range = getDefinitionRange(TSTL.getBeginLoc(), TSTL.getLAngleLoc());
      Len = 0;
    }
    Len += SM.getDecomposedLoc(Range.getEnd()).second -
           SM.getDecomposedLoc(Range.getBegin()).second;
    emplaceTransformation(
        new ReplaceText(Range.getBegin(), Len, std::move(ReplStr)));
    for (auto ItHeader = It->second->Includes.begin();
         ItHeader != It->second->Includes.end(); ItHeader++) {
      DpctGlobalInfo::getInstance().insertHeader(Range.getBegin(), *ItHeader);
    }
  }
}

void clang::dpct::UserDefinedClassFieldRule::registerMatcher(
    clang::ast_matchers::MatchFinder &MF) {
  MF.addMatcher(memberExpr(allOf(hasObjectExpression(hasType(qualType(
                                     hasCanonicalType(recordType(hasDeclaration(
                                         cxxRecordDecl(hasName(BaseName)))))))),
                                 member(hasName(FieldName))))
                    .bind("memberExpr"),
                this);
}

void clang::dpct::UserDefinedClassFieldRule::runRule(
    const clang::ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto ME = getNodeAsType<MemberExpr>(Result, "memberExpr")) {
    dpct::ExprAnalysis EA;
    EA.analyze(ME);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

void clang::dpct::UserDefinedClassMethodRule::registerMatcher(
    clang::ast_matchers::MatchFinder &MF) {
  MF.addMatcher(cxxMemberCallExpr(
                    allOf(on(hasType(hasCanonicalType(qualType(
                              hasDeclaration(namedDecl(hasName(BaseName))))))),
                          callee(cxxMethodDecl(hasName(MethodName)))))
                    .bind("memberCallExpr"),
                this);
}

void clang::dpct::UserDefinedClassMethodRule::runRule(
    const clang::ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto CMCE = getNodeAsType<CXXMemberCallExpr>(Result, "memberCallExpr")) {
    dpct::ExprAnalysis EA;
    EA.analyze(CMCE);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

void clang::dpct::UserDefinedEnumRule::registerMatcher(
    clang::ast_matchers::MatchFinder &MF) {
  MF.addMatcher(
      declRefExpr(to(enumConstantDecl(hasType(enumDecl(hasName(EnumName))))))
          .bind("EnumConstant"),
      this);
}

void clang::dpct::UserDefinedEnumRule::runRule(
    const clang::ast_matchers::MatchFinder::MatchResult &Result) {
  if (const DeclRefExpr *E =
          getNodeAsType<DeclRefExpr>(Result, "EnumConstant")) {
    dpct::ExprAnalysis EA;
    EA.analyze(E);
    emplaceTransformation(EA.getReplacement());
    EA.applyAllSubExprRepl();
  }
}

namespace pattern_rewriter{
struct SpacingElement {};

struct LiteralElement {
  char Value;
};

struct CodeElement {
  std::string Name;
  int SuffixLength = -1;
};

using Element = std::variant<SpacingElement, LiteralElement, CodeElement>;

using MatchPattern = std::vector<Element>;

struct MatchResult {
  int Start;
  int End;
  std::unordered_map<std::string, std::string> Bindings;
};

static bool isWhitespace(char Character) {
  return Character == ' ' || Character == '\t' || Character == '\n';
}

static bool isRightDelimiter(char Character) {
  return Character == '}' || Character == ']' || Character == ')';
}

static int detectIndentation(const std::string &Input, int Start) {
  int Indentation = 0;
  int Index = Start - 1;
  while (Index >= 0 && isWhitespace(Input[Index])) {
    if (Input[Index] == '\n' || Index == 0) {
      Indentation = Start - Index - 1;
      break;
    }
    Index--;
  }
  return Indentation;
}

static std::string join(const std::vector<std::string> Lines,
                        const std::string &Separator) {
  if (Lines.size() == 0) {
    return "";
  }
  std::stringstream OutputStream;
  const int Count = Lines.size();
  for (int i = 0; i < Count - 1; i++) {
    OutputStream << Lines[i];
    OutputStream << Separator;
  }
  OutputStream << Lines.back();
  return OutputStream.str();
}

static std::string trim(const std::string &Input) {
  const int Size = Input.size();
  int Index = 0;

  while (Index < Size && isWhitespace(Input[Index])) {
    Index++;
  }

  int End = Size - 1;
  while (End > (Index + 1) && isWhitespace(Input[End])) {
    End--;
  }

  return Input.substr(Index, End + 1);
}

static std::string indent(const std::string &Input, int Indentation) {
  std::vector<std::string> Output;
  const auto Indent = std::string(Indentation, ' ');
  const auto Lines = split(Input, '\n');
  for (const auto &Line : Lines) {
    const bool ContainsNonWhitespace = (trim(Line).size() > 0);
    Output.push_back(ContainsNonWhitespace ? (Indent + Line) : "");
  }
  return trim(join(Output, "\n"));
}

static std::string dedent(const std::string &Input, int Indentation) {
  std::stringstream OutputStream;
  const int Size = Input.size();
  int Index = 0;
  int Skip = 0;
  while (Index < Size) {
    char Character = Input[Index];
    if (Skip > 0 && Character == ' ') {
      Skip--;
      Index++;
      continue;
    }
    if (Character == '\n') {
      Skip = Indentation;
    }
    OutputStream << Character;
    Index++;
  }
  return OutputStream.str();
}

/*
Determines the number of pattern elements that form the suffix of a code
element. The suffix of a code element extends up to the next code element, an
unbalanced right Delimiter, or the end of the pattern. Example:

Pattern:
  if (${a} == ${b}) ${body}

${a}:
  Suffix: [Spacing, '=', '=', Spacing]
  SuffixLength: 4

${b}:
  Suffix: [')']
  SuffixLength: 1

${body}:
  Suffix: []
  SuffixLength: 0
*/
static void adjustSuffixLengths(MatchPattern &Pattern) {
  int SuffixTerminator = Pattern.size() - 1;
  for (int i = Pattern.size() - 1; i >= 0; i--) {
    auto &Element = Pattern[i];

    if (std::holds_alternative<CodeElement>(Element)) {
      auto &Code = std::get<CodeElement>(Element);
      Code.SuffixLength = SuffixTerminator - i;
      SuffixTerminator = i - 1;
      continue;
    }

    if (std::holds_alternative<LiteralElement>(Element)) {
      auto &Literal = std::get<LiteralElement>(Element);
      if (isRightDelimiter(Literal.Value)) {
        SuffixTerminator = i;
      }
      continue;
    }
  }
}

static void removeTrailingSpacingElement(MatchPattern &Pattern) {
  if (std::holds_alternative<SpacingElement>(Pattern.back())) {
    Pattern.pop_back();
  }
}

static MatchPattern parseMatchPattern(std::string Pattern) {
  MatchPattern Result;

  const int Size = Pattern.size();
  int Index = 0;
  while (Index < Size) {
    const char Character = Pattern[Index];

    if (isWhitespace(Character)) {
      if (Result.size() > 0) {
        Result.push_back(SpacingElement{});
      }
      while (Index < Size && isWhitespace(Pattern[Index])) {
        Index++;
      }
      continue;
    }

    if (Index < (Size - 1) && Character == '$' && Pattern[Index + 1] == '{') {
      Index += 2;

      const auto RightCurly = Pattern.find('}', Index);
      if (RightCurly == std::string::npos) {
        throw std::runtime_error("Invalid match pattern expression");
      }
      std::string Name = Pattern.substr(Index, RightCurly - Index);
      Index = RightCurly + 1;

      Result.push_back(CodeElement{Name});
      continue;
    }

    Result.push_back(LiteralElement{Character});
    Index++;
  }

  removeTrailingSpacingElement(Result);
  adjustSuffixLengths(Result);
  return Result;
}

static std::optional<MatchResult> findMatch(const MatchPattern &Pattern,
                                            const std::string &Input,
                                            const int Start);

static int parseCodeElement(const MatchPattern &Suffix,
                            const std::string &Input, const int Start);

static int parseBlock(char LeftDelimiter, char RightDelimiter,
                      const std::string &Input, const int Start) {
  const int Size = Input.size();
  int Index = Start;

  if (Index >= Size || Input[Index] != LeftDelimiter) {
    return -1;
  }
  Index++;

  Index = parseCodeElement({}, Input, Index);
  if (Index == -1) {
    return -1;
  }

  if (Index >= Size || Input[Index] != RightDelimiter) {
    return -1;
  }
  Index++;
  return Index;
}

static int parseCodeElement(const MatchPattern &Suffix,
                            const std::string &Input, const int Start) {
  int Index = Start;
  const int Size = Input.size();
  while (Index >= 0 && Index < Size) {
    const auto Character = Input[Index];

    if (Suffix.size() > 0) {
      const auto SuffixMatch = findMatch(Suffix, Input, Index);
      if (SuffixMatch.has_value()) {
        return Index;
      }

      if (isRightDelimiter(Character) || Index == Size - 1) {
        return -1;
      }
    }

    if (Character == '{') {
      Index = parseBlock('{', '}', Input, Index);
      continue;
    }

    if (Character == '[') {
      Index = parseBlock('[', ']', Input, Index);
      continue;
    }

    if (Character == '(') {
      Index = parseBlock('(', ')', Input, Index);
      continue;
    }

    if (isRightDelimiter(Input[Index])) {
      break;
    }

    /*
    The following parsers skip character literals, string literals, and
    comments. These tokens are skipped since they may contain unbalanced
    delimiters.
    */

    if (Character == '\'') {
      Index++;
      while (Index < Size &&
             !(Input[Index - 1] != '\\' && Input[Index] == '\'')) {
        Index++;
      }
      if (Index >= Size) {
        return -1;
      }
      Index++;
      continue;
    }

    if (Character == '"') {
      Index++;
      while (Index < Size &&
             !(Input[Index - 1] != '\\' && Input[Index] == '"')) {
        Index++;
      }
      if (Index >= Size) {
        return -1;
      }
      Index++;
      continue;
    }

    if (Character == '/' && Index < (Size - 1) && Input[Index + 1] == '/') {
      Index += 2;
      while (Index < Size && Input[Index] != '\n') {
        Index++;
      }
      if (Index >= Size) {
        return -1;
      }
      Index++;
      continue;
    }

    if (Character == '/' && Index < (Size - 1) && Input[Index + 1] == '*') {
      Index += 2;
      while (Index < Size &&
             !(Input[Index - 1] == '*' && Input[Index] == '/')) {
        Index++;
      }
      if (Index >= Size) {
        return -1;
      }
      Index++;
      continue;
    }

    Index++;
  }
  return Suffix.size() == 0 ? Index : -1;
}

static std::optional<MatchResult> findMatch(const MatchPattern &Pattern,
                                            const std::string &Input,
                                            const int Start) {
  MatchResult Result;

  int Index = Start;
  int PatternIndex = 0;
  const int PatternSize = Pattern.size();
  const int Size = Input.size();

  while (PatternIndex < PatternSize && Index < Size) {
    const auto &Element = Pattern[PatternIndex];

    if (std::holds_alternative<SpacingElement>(Element)) {
      if (!isWhitespace(Input[Index])) {
        return {};
      }
      while (Index < Size && isWhitespace(Input[Index])) {
        Index++;
      }
      PatternIndex++;
      continue;
    }

    if (std::holds_alternative<LiteralElement>(Element)) {
      const auto &Literal = std::get<LiteralElement>(Element);
      if (Input[Index] != Literal.Value) {
        return {};
      }
      Index++;
      PatternIndex++;
      continue;
    }

    if (std::holds_alternative<CodeElement>(Element)) {
      const auto &Code = std::get<CodeElement>(Element);
      MatchPattern Suffix(Pattern.begin() + PatternIndex + 1,
                          Pattern.begin() + PatternIndex + 1 +
                              Code.SuffixLength);

      int Next = parseCodeElement(Suffix, Input, Index);
      if (Next == -1) {
        return {};
      }
      const int Indentation = detectIndentation(Input, Index);
      std::string ElementContents =
          dedent(Input.substr(Index, Next - Index), Indentation);
      if (Result.Bindings.count(Code.Name)) {
        if (Result.Bindings[Code.Name] != ElementContents) {
          return {};
        }
      } else {
        Result.Bindings[Code.Name] = std::move(ElementContents);
      }
      Index = Next;
      PatternIndex++;
      continue;
    }

    throw std::runtime_error("Internal error: invalid pattern element");
  }

  Result.Start = Start;
  Result.End = Index;
  return Result;
}

static void instantiateTemplate(
    const std::string &Template,
    const std::unordered_map<std::string, std::string> &Bindings,
    const int Indentation, std::ostream &OutputStream) {
  const auto LeadingSpace = std::string(Indentation, ' ');
  const int Size = Template.size();
  int Index = 0;

  while (Index < Size && isWhitespace(Template[Index])) {
    Index++;
  }

  int End = Size - 1;
  while (End > (Index + 1) && isWhitespace(Template[End])) {
    End--;
  }

  while (Index <= End) {
    const auto Character = Template[Index];

    if (Index < (Size - 1) && Character == '$' && Template[Index + 1] == '{') {
      const int BindingStart = Index;
      Index += 2;

      const auto RightCurly = Template.find('}', Index);
      if (RightCurly == std::string::npos) {
        throw std::runtime_error("Invalid rewrite pattern expression");
      }
      std::string Name = Template.substr(Index, RightCurly - Index);
      Index = RightCurly + 1;

      const auto &BindingIterator = Bindings.find(Name);
      if (BindingIterator != Bindings.end()) {
        const int BindingIndentation =
            detectIndentation(Template, BindingStart) + Indentation;
        const std::string Contents =
            indent(BindingIterator->second, BindingIndentation);
        OutputStream << Contents;
      }
      continue;
    }

    OutputStream << Character;
    if (Character == '\n') {
      OutputStream << LeadingSpace;
    }

    Index++;
  }
}
} // namespace pattern_rewriter

bool fixLineEndings(const std::string &Input, std::string &Output) {
  std::stringstream OutputStream;
  bool isCRLF = false;
  int Index = 0;
  int Size = Input.size();
  while (Index < Size) {
    char Character = Input[Index];
    if (Character != '\r') {
      OutputStream << Character;
    } else {
      isCRLF = true;
    }
    Index++;
  }
  Output = OutputStream.str();
  return isCRLF;
}

std::string applyPatternRewriter(const MetaRuleObject::PatternRewriter &PP,
                                 const std::string &Input) {
  std::stringstream OutputStream;
  const auto Pattern = pattern_rewriter::parseMatchPattern(PP.In);
  const int Size = Input.size();
  int Index = 0;
  while (Index < Size) {
    auto Result = pattern_rewriter::findMatch(Pattern, Input, Index);

    if (Result.has_value()) {
      auto &Match = Result.value();
      for (const auto &[Name, Value] : Match.Bindings) {
        const auto &SubruleIterator = PP.Subrules.find(Name);
        if (SubruleIterator != PP.Subrules.end()) {
          Match.Bindings[Name] =
              applyPatternRewriter(SubruleIterator->second, Value);
        }
      }

      const int Indentation = pattern_rewriter::detectIndentation(Input, Index);
      pattern_rewriter::instantiateTemplate(PP.Out, Match.Bindings, Indentation,
                                            OutputStream);
      Index = Match.End;
      continue;
    }

    OutputStream << Input[Index];
    Index++;
  }

  return OutputStream.str();
}