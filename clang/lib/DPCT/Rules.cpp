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
#include "Utility.h"
#include "llvm/Support/YAMLTraits.h"
#include "NCCLAPIMigration.h"
std::vector<std::string> MetaRuleObject::RuleFiles;
std::vector<std::shared_ptr<MetaRuleObject>> MetaRules;

void registerMacroRule(MetaRuleObject &R) {
  auto It = MapNames::MacroRuleMap.find(R.In);
  if (It != MapNames::MacroRuleMap.end()) {
    if (It->second.Priority > R.Priority) {
      It->second.Id = R.RuleId;
      It->second.Priority = R.Priority;
      It->second.In = R.In;
      It->second.Out = R.Out;
      It->second.HelperFeature =
          clang::dpct::HelperFeatureEnum::no_feature_helper;
      It->second.Includes = R.Includes;
    }
  } else {
    MapNames::MacroRuleMap.emplace(
        R.In,
        MacroMigrationRule(R.RuleId, R.Priority, R.In, R.Out,
                           clang::dpct::HelperFeatureEnum::no_feature_helper,
                           R.Includes));
  }
}

void registerAPIRule(MetaRuleObject &R) {
  // register rule
  clang::dpct::ASTTraversalMetaInfo::registerRule((char *)&R, R.RuleId, [=] {
    return new clang::dpct::UserDefinedAPIRule(R.In);
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
  auto It = clang::dpct::CallExprRewriterFactoryBase::RewriterMap->find(R.In);
  if (It == clang::dpct::CallExprRewriterFactoryBase::RewriterMap->end()) {
    clang::dpct::CallExprRewriterFactoryBase::RewriterMap->emplace(
        R.In, clang::dpct::createUserDefinedRewriterFactory(R.In, R));
  } else if (It->second->Priority > R.Priority) {
    (*clang::dpct::CallExprRewriterFactoryBase::RewriterMap)[R.In] =
        clang::dpct::createUserDefinedRewriterFactory(R.In, R);
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
          clang::dpct::HelperFeatureEnum::no_feature_helper;
      It->second->Includes.insert(It->second->Includes.end(),
                                  R.Includes.begin(), R.Includes.end());
    }
  } else {
    clang::dpct::ASTTraversalMetaInfo::registerRule((char *)&R, R.RuleId, [=] {
      return new clang::dpct::UserDefinedTypeRule(R.In);
    });
    auto RulePtr = std::make_shared<TypeNameRule>(
        R.Out, clang::dpct::HelperFeatureEnum::no_feature_helper, R.Priority);
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
            clang::dpct::HelperFeatureEnum::no_feature_helper;
        ItFieldRule->second->Includes.clear();
        ItFieldRule->second->Includes.insert(
            ItFieldRule->second->Includes.end(), R.Includes.begin(),
            R.Includes.end());
      }
    } else {
      clang::dpct::ASTTraversalMetaInfo::registerRule(
          (char *)&(**ItField), BaseAndFieldName, [=] {
            return new clang::dpct::UserDefinedClassFieldRule(R.In,
                                                              (*ItField)->In);
          });
      std::shared_ptr<ClassFieldRule> RulePtr;
      if ((*ItField)->OutGetter != "") {
        RulePtr = std::make_shared<ClassFieldRule>(
            (*ItField)->OutSetter, (*ItField)->OutGetter,
            clang::dpct::HelperFeatureEnum::no_feature_helper, R.Priority);
      } else {
        RulePtr = std::make_shared<ClassFieldRule>(
            (*ItField)->Out, clang::dpct::HelperFeatureEnum::no_feature_helper,
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
    clang::dpct::ASTTraversalMetaInfo::registerRule(
        (char *)&(**ItMethod), BaseAndMethodName, [=] {
          return new clang::dpct::UserDefinedClassMethodRule(R.In,
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
          clang::dpct::HelperFeatureEnum::no_feature_helper;
      It->second->Includes.insert(It->second->Includes.end(),
                                 R.Includes.begin(), R.Includes.end());
    }
  } else {
    if(R.EnumName == ""){
      return;
    }
    clang::dpct::ASTTraversalMetaInfo::registerRule((char *)&R, R.RuleId, [=] {
      return new clang::dpct::UserDefinedEnumRule(R.EnumName);
    });
    auto RulePtr = std::make_shared<EnumNameRule>(
        R.Out, clang::dpct::HelperFeatureEnum::no_feature_helper, R.Priority);
    RulePtr->Includes.insert(RulePtr->Includes.end(), R.Includes.begin(),
                             R.Includes.end());
    clang::dpct::EnumConstantRule::EnumNamesMap.emplace(
        R.EnumName + "::" + R.In, RulePtr);
  }
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
                 << ", $ followed by a positive integer is expected after "
                 << Keyword << "\n";
    clang::dpct::ShowStatus(MigrationErrorCannotParseRuleFile);
    dpctExit(MigrationErrorCannotParseRuleFile);
  }

  // consume $
  Idx++;
  ignoreWhitespaces(OutStr, Idx);
  int ArgIndex = 0;

  // process arg number
  std::string ArgNumStr = OutStr.substr(Idx);
  std::size_t pos = 0;

  try{
    ArgIndex = std::stoi(ArgNumStr, &pos);
  } catch(std::invalid_argument const& Ex){
    llvm::errs() << RuleFile << ":Error: in rule " << RuleName
                 << ", unknown keyword: $" << ArgNumStr.substr(0, 10) << "\n";
    clang::dpct::ShowStatus(MigrationErrorCannotParseRuleFile);
    dpctExit(MigrationErrorCannotParseRuleFile);
  } catch(std::out_of_range const& Ex) {
    llvm::errs() << RuleFile << ":Error: in rule " << RuleName
                 << ", argument index out of range.\n";
    clang::dpct::ShowStatus(MigrationErrorCannotParseRuleFile);
    dpctExit(MigrationErrorCannotParseRuleFile);
  }
  Idx = Idx + pos;

  if (ArgIndex <= 0) {
    // report invalid ArgIndex
    llvm::errs() << RuleFile << ":Error: in rule " << RuleName
                 << ", expect a positive integer, found " << ArgIndex
                 << " after " << Keyword << "\n";
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
namespace clang {
namespace ast_matchers {
AST_MATCHER_P(DeclRefExpr, hasRefName, std::string, NameToMatch) {
  auto Qualifier = Node.getQualifier();
  if(!Qualifier)
    return false;
  std::string RefName = getNestedNameSpecifierString(Qualifier).c_str() +
                        Node.getNameInfo().getAsString();
  return !RefName.compare(NameToMatch);
}
} // namespace ast_matchers
} // namespace clang
using namespace clang::ast_matchers;

void clang::dpct::UserDefinedAPIRule::registerMatcher(
    clang::ast_matchers::MatchFinder &MF) {
  auto Pos = APIName.rfind("::");
  if (Pos == std::string::npos || Pos == 0) {
    MF.addMatcher(callExpr(callee(functionDecl(hasName(
                               Pos == 0 ? APIName.substr(2) : APIName))))
                      .bind("call"),
                  this);
  } else {
    MF.addMatcher(callExpr(callee(expr(ignoringImpCasts(
                               declRefExpr(hasRefName(APIName))))))
                      .bind("call"),
                  this);
  }
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
    if(auto TSTL = TL->getAs<TemplateSpecializationTypeLoc>()){
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
    if (auto TSTL = TL->getAs<TemplateSpecializationTypeLoc>()) {
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
