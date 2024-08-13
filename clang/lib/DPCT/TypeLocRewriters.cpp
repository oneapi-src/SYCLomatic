//===--------------- TypeLocRewriters.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeLocRewriters.h"
#include "ASTTraversal.h"
#include "clang/AST/TypeLoc.h"
#include "AnalysisInfo.h"
#include "Rules.h"
#include "MapNames.h"

namespace clang {
namespace dpct {

// std::function<std::string(const TypeLoc)>
// makeStringCreator(std::string TypeName) {
//   return [=](const TypeLoc TL) -> std::string {
//     return TypeName;
//   };
// }

std::function<std::string(const TypeLoc)>
makeStringCreator(std::string TypeName,
                  clang::dpct::HelperFeatureEnum RequestFeature,
                  const std::vector<std::string> &Headers) {
  return [=](const TypeLoc TL) -> std::string {
    requestFeature(RequestFeature);
    return TypeName;
  };
}

TemplateArgumentInfo getTemplateArg(const TypeLoc &TL, unsigned Idx) {
  if (auto TSTL = TL.getAs<TemplateSpecializationTypeLoc>()) {
    if (TSTL.getNumArgs() > Idx) {
      auto TAL = TSTL.getArgLoc(Idx);
      return TemplateArgumentInfo(TAL, TL.getSourceRange());
    }
  }
  return TemplateArgumentInfo("");
}

std::function<TemplateArgumentInfo(const TypeLoc)>
makeTemplateArgCreator(unsigned Idx) {
  return [=](const TypeLoc TL) -> TemplateArgumentInfo {
    return getTemplateArg(TL, Idx);
  };
}

template <typename T>
std::function<std::string(const TypeLoc)>
makeAddPointerCreator(std::function<T(const TypeLoc)> f) {
  return [=](const TypeLoc TL) {
    std::string s;
    llvm::raw_string_ostream OS(s);
    dpct::print(OS, f(TL));
    OS << " *";
    return s;
  };
}

std::function<std::string(const TypeLoc)>
makeTypeStrCreator() {
  return [=](const TypeLoc TL) {
    if(!TL)
      return std::string();
    auto PP = DpctGlobalInfo::getContext().getPrintingPolicy();
    PP.SuppressTagKeyword = true;
    PP.FullyQualifiedName = true;
    return TL.getType().getAsString(PP);
  };
}

std::function<std::string(const TypeLoc)>
makeUserDefinedTypeStrCreator(MetaRuleObject &R, TypeOutputBuilder &TOB) {
  return [=](const TypeLoc TL) {
    if (!TL)
      return std::string();
    auto Range = getDefinitionRange(TL.getBeginLoc(), TL.getEndLoc());


    //auto &SM = DpctGlobalInfo::getSourceManager();

    // auto Len = Lexer::MeasureTokenLength(
    //     Range.getEnd(), SM, DpctGlobalInfo::getContext().getLangOpts());
    // if (auto TSTL = TL->getAsAdjusted<TemplateSpecializationTypeLoc>()) {
    //   Range = getDefinitionRange(TSTL.getBeginLoc(), TSTL.getLAngleLoc());
    //   Len = 0;
    // }
    // Len += SM.getDecomposedLoc(Range.getEnd()).second -
    //        SM.getDecomposedLoc(Range.getBegin()).second;
    // emplaceTransformation(
    //     new ReplaceText(Range.getBegin(), Len, std::move(ReplStr)));

    for (auto ItHeader : R.Includes) {
      DpctGlobalInfo::getInstance().insertHeader(Range.getBegin(), ItHeader);
    }

    std::string ResultStr;
    llvm::raw_string_ostream OS(ResultStr);
    for (auto &tob : TOB.SubBuilders) {
      switch (tob->Kind) {
      case (OutputBuilder::Kind::String):
        OS << tob->Str;
        break;
      case (OutputBuilder::Kind::TemplateArg): {
        OS << getTemplateArg(TL, tob->ArgIndex).getString();
        break;
      }
      default:
        DpctDebugs() << "[OutputBuilder::Kind] Unexpected value: " << tob->Kind
                     << "\n";
        assert(0);
      }
    }
    OS.flush();
    return ResultStr;
  };
}

class CheckTemplateArgCount {
  unsigned Count;

public:
  CheckTemplateArgCount(unsigned I) : Count(I) {}
  bool operator()(const TypeLoc TL) {
    if(auto TSTL = TL.getAs<TemplateSpecializationTypeLoc>()){
      return TSTL.getNumArgs() == Count;
    }
    return false;
  }
};

inline auto CheckForPostfixDeclaratorType(unsigned Idx) {
  return [=](const TypeLoc TL){
    if (const auto TSTL = TL.getAs<TemplateSpecializationTypeLoc>()) {
      const auto TAT = TSTL.getArgLoc(Idx).getArgument().getAsType();
      const auto CT = TAT.getCanonicalType();
      return typeIsPostfix(CT);
    }
    return false;
  };
}

class CheckTypeNameAndInVarDecl {
  std::string Name;
public:
  CheckTypeNameAndInVarDecl(const std::string &N) : Name(N) {}
  bool operator()(const TypeLoc TL) {
    if (const auto *VD = DpctGlobalInfo::findAncestor<VarDecl>(&TL)) {
      return DpctGlobalInfo::getTypeName(VD->getType()) == Name;
    }
    return false;
  }
};

inline std::function<bool(const TypeLoc TL)>
checkTemplateArgSpelling(size_t Index, std::string Str) {
  auto getQualtifiedNameStr = [=](const NamedDecl *NL) -> std::string {
    if (NL == nullptr)
      return "";
    if (const auto *NSD = dyn_cast<NamespaceDecl>(NL->getDeclContext())) {
      std::string TypeQualifiedString =
          getNameSpace(NSD) + "::" + NL->getNameAsString();
      return TypeQualifiedString;
    }
    return NL->getNameAsString();
  };

  return [=](const TypeLoc TL) -> bool {
    if (const auto &TSTL = TL.getAs<TemplateSpecializationTypeLoc>()) {
      if (Index > TSTL.getNumArgs())
        return false;
      const auto TA = TSTL.getArgLoc(Index).getArgument();
      if (TA.getKind() == TemplateArgument::ArgKind::Type) {
        std::string ResStr =
            getQualtifiedNameStr(TA.getAsType()->getAsTagDecl());
        if (ResStr.empty())
          return TA.getAsType().getAsString() == Str;
        return ResStr == Str;
      } else if (TA.getKind() == TemplateArgument::ArgKind::Declaration) {
        return getQualtifiedNameStr(TA.getAsDecl()) == Str;
      } else if (TA.getKind() == TemplateArgument::ArgKind::Integral) {
        return std::to_string(TA.getAsIntegral().getExtValue()) == Str;
      } else if (TA.getKind() == TemplateArgument::ArgKind::Expression) {
        return getStmtSpelling(TA.getAsExpr()) == Str;
      }
    }
    return false;
  };
}

std::function<bool(const TypeLoc)> checkEnableJointMatrixForType() {
  return [=](const TypeLoc) -> bool {
    return DpctGlobalInfo::useExtJointMatrix();
  };
}

std::function<bool(const TypeLoc)> checkEnableGraphForType() {
  return [=](const TypeLoc) -> bool { return DpctGlobalInfo::useExtGraph(); };
}

std::function<bool(const TypeLoc)> isUseNonUniformGroupsForType() {
  return [=](const TypeLoc) -> bool {
    return DpctGlobalInfo::useExpNonUniformGroups();
  };
}

std::function<bool(const TypeLoc)> isUseLogicalGroupsForType() {
  return [=](const TypeLoc) -> bool {
    return DpctGlobalInfo::useLogicalGroup();
  };
}

// Print a templated type. Pass a STR("") as a template argument for types with
// empty template argument e.g. MyType<>, If --enable-ctad is set, the template
// arguments which could be deduced with class template argument deduction(CTAD)
// will be omitted in the generated code.
template <class TypeNameT, class... TemplateArgsT>
std::shared_ptr<TypeLocRewriterFactoryBase> createCtadTypeLocRewriterFactory(
    std::function<TypeNameT(const TypeLoc)> TypeNameCreator,
    std::function<TemplateArgsT(const TypeLoc)>... TAsCreator) {
  return std::make_shared<
      TypeLocRewriterFactory<CtadTemplateTypeLocRewriter<TypeNameT, TemplateArgsT...>,
                             std::function<TypeNameT(const TypeLoc)>,
                             std::function<TemplateArgsT(const TypeLoc)>...>>(
      std::forward<std::function<TypeNameT(const TypeLoc)>>(TypeNameCreator),
      std::forward<std::function<TemplateArgsT(const TypeLoc)>>(TAsCreator)...);
}

// Print a type without template.
template <class TypeNameT>
std::shared_ptr<TypeLocRewriterFactoryBase> createCtadTypeLocRewriterFactory(
    std::function<TypeNameT(const TypeLoc)> TypeNameCreator) {
  return std::make_shared<
      TypeLocRewriterFactory<TypeNameTypeLocRewriter<TypeNameT>,
                             std::function<TypeNameT(const TypeLoc)>>>(
      std::forward<std::function<TypeNameT(const TypeLoc)>>(TypeNameCreator));
}

std::shared_ptr<TypeLocRewriterFactoryBase> createTypeLocConditionalFactory(
    std::function<bool(const TypeLoc)> Pred,
    std::shared_ptr<TypeLocRewriterFactoryBase> &&First,
    std::shared_ptr<TypeLocRewriterFactoryBase> &&Second) {
  return std::make_shared<TypeLocConditionalRewriterFactory>(Pred, First,
                                                             Second);
}

template <typename... Args> 
std::shared_ptr<TypeLocRewriterFactoryBase>
createReportWarningTypeLocRewriterFactory(Diagnostics MsgId,
                                          Args&&... args) {
  return std::make_shared<
    TypeLocRewriterFactory<ReportWarningTypeLocRewriter, Diagnostics, Args...>>
    (MsgId, std::forward<Args>(args)...);
}

std::pair<TypeMatchingDesc, std::shared_ptr<TypeLocRewriterFactoryBase>>
createFeatureRequestFactory(
    HelperFeatureEnum Feature,
    std::pair<TypeMatchingDesc, std::shared_ptr<TypeLocRewriterFactoryBase>>
        &&Input) {
  return std::pair<TypeMatchingDesc, std::shared_ptr<TypeLocRewriterFactoryBase>>(
      std::move(Input.first),
      std::make_shared<TypeLocRewriterFactoryWithFeatureRequest>(Feature,
                                                                 Input.second));
}
template <class T>
std::pair<TypeMatchingDesc, std::shared_ptr<TypeLocRewriterFactoryBase>>
createFeatureRequestFactory(
    HelperFeatureEnum Feature,
    std::pair<TypeMatchingDesc, std::shared_ptr<TypeLocRewriterFactoryBase>>
        &&Input,
    T) {
  return createFeatureRequestFactory(Feature, std::move(Input));
}

std::shared_ptr<TypeLocRewriterFactoryBase> createHeaderInsertionFactory(
    HeaderType Header,
    std::shared_ptr<TypeLocRewriterFactoryBase> &&SubRewriterFactory) {
  return std::make_shared<HeaderInsertionRewriterFactory>(Header,
                                                          SubRewriterFactory);
}

std::unique_ptr<std::unordered_map<TypeMatchingDesc,
                                   std::shared_ptr<TypeLocRewriterFactoryBase>,
                                   TypeMatchingDesc::hash>>
    TypeLocRewriterFactoryBase::TypeLocRewriterMap;

void TypeLocRewriterFactoryBase::initTypeLocRewriterMap() {
  TypeLocRewriterMap = std::make_unique<std::unordered_map<
      TypeMatchingDesc, std::shared_ptr<TypeLocRewriterFactoryBase>,
      TypeMatchingDesc::
          hash>>(std::unordered_map<TypeMatchingDesc,
                                    std::shared_ptr<TypeLocRewriterFactoryBase>,
                                    TypeMatchingDesc::hash>({
#define STR(Str) makeStringCreator(Str)
#define TEMPLATE_ARG(Idx) makeTemplateArgCreator(Idx)
#define TYPE_REWRITE_ENTRY(Name, Factory) {Name, Factory},
#define TYPE_CONDITIONAL_FACTORY(Pred, First, Second)                          \
  createTypeLocConditionalFactory(Pred, First, Second)
#define TYPE_FACTORY(...) createTypeLocRewriterFactory(__VA_ARGS__)
#define CTAD_TYPE_FACTORY(...) createCtadTypeLocRewriterFactory(__VA_ARGS__)
#define FEATURE_REQUEST_FACTORY(FEATURE, x)                                    \
  createFeatureRequestFactory(FEATURE, x 0),
#define HEADER_INSERTION_FACTORY(HEADER, SUB)                                  \
  createHeaderInsertionFactory(HEADER, SUB)
#define TYPESTR makeTypeStrCreator()
#define WARNING_FACTORY(MSGID, ...)                                            \
  createReportWarningTypeLocRewriterFactory(MSGID, __VA_ARGS__)
#define ADD_POINTER(CREATOR) makeAddPointerCreator(CREATOR)
#include "APINamesTemplateType.inc"
#undef WARNING_FACTORY
#undef ADD_POINTER
#undef HEADER_INSERTION_FACTORY
#undef FEATURE_REQUEST_FACTORY
#undef TYPE_FACTORY
#undef TYPE_CONDITIONAL_FACTORY
#undef TYPE_REWRITE_ENTRY
#undef TEMPLATE_ARG
#undef STR
      {{"cudaDeviceProp"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "device_info",
                         HelperFeatureEnum::device_ext))},
      {{"cudaError_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "err0",
                         HelperFeatureEnum::device_ext))},
      {{"cudaError"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "err0",
                         HelperFeatureEnum::device_ext))},
      {{"CUjit_option"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"CUresult"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"CUcontext"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"CUmodule"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "kernel_library",
                         HelperFeatureEnum::device_ext))},
      {{"CUfunction"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "kernel_function",
                         HelperFeatureEnum::device_ext))},
      {{"cudaPointerAttributes"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "pointer_attributes",
                         HelperFeatureEnum::device_ext))},
      {{"dim3"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "dim3"))},
      {{"int2"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "int2"))},
      {{"double2"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "double2"))},
      {{"__half"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "half"))},
      {{"__half2"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "half2"))},
      {{"half"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "half"))},
      {{"half2"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "half2"))},
      {{"cudaEvent_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "event_ptr",
                         HelperFeatureEnum::device_ext))},
      {{"CUevent"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "event_ptr"))},
      {{"CUevent_st"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "event"))},
      {{"CUfunc_cache"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cudaStream_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "queue_ptr",
                         HelperFeatureEnum::device_ext))},
      {{"CUstream"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "queue_ptr",
                         HelperFeatureEnum::device_ext))},
      {{"CUstream_st"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "queue"))},
      {{"char1"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int8_t"))},
      {{"char2"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "char2"))},
      {{"char3"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "char3"))},
      {{"char4"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "char4"))},
      {{"double1"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("double"))},
      {{"double2"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "double2"))},
      {{"double3"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "double3"))},
      {{"double4"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "double4"))},
      {{"float1"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("float"))},
      {{"float2"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "float2"))},
      {{"float3"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "float3"))},
      {{"float4"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "float4"))},
      {{"int1"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int32_t"))},
      {{"int2"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "int2"))},
      {{"int3"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "int3"))},
      {{"int4"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "int4"))},
      {{"long1"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int64_t"))},
      {{"long2"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "long2"))},
      {{"long3"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "long3"))},
      {{"long4"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "long4"))},
      {{"longlong1"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int64_t"))},
      {{"longlong2"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "long2"))},
      {{"longlong3"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "long3"))},
      {{"longlong4"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "long4"))},
      {{"short1"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int16_t"))},
      {{"short2"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "short2"))},
      {{"short3"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "short3"))},
      {{"short4"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "short4"))},
      {{"uchar1"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("uint8_t"))},
      {{"uchar2"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "uchar2"))},
      {{"uchar3"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "uchar3"))},
      {{"uchar4"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "uchar4"))},
      {{"uint1"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("uint32_t"))},
      {{"uint2"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "uint2"))},
      {{"uint3"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "uint3"))},
      {{"uint4"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "uint4"))},
      {{"ulong1"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("uint64_t"))},
      {{"ulong2"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "ulong2"))},
      {{"ulong3"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "ulong3"))},
      {{"ulong4"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "ulong4"))},
      {{"ulonglong1"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("uint64_t"))},
      {{"ulonglong2"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "ulong2"))},
      {{"ulonglong3"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "ulong3"))},
      {{"ulonglong4"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "ulong4"))},
      {{"ushort1"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("uint16_t"))},
      {{"ushort2"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "ushort2"))},
      {{"ushort3"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "ushort3"))},
      {{"ushort4"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "ushort4"))},
      {{"cublasHandle_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "blas::descriptor_ptr",
                         HelperFeatureEnum::device_ext))},
      {{"cublasStatus_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cublasStatus"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cublasGemmAlgo_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cudaDataType_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "library_data_t",
                         HelperFeatureEnum::device_ext))},
      {{"cudaDataType"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "library_data_t",
                         HelperFeatureEnum::device_ext))},
      {{"cublasDataType_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "library_data_t",
                         HelperFeatureEnum::device_ext))},
      {{"cublasComputeType_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "compute_type"))},
      {{"cuComplex"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "float2"))},
      {{"cuFloatComplex"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "float2"))},
      {{"cuDoubleComplex"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "double2"))},
      {{"cublasFillMode_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::mkl::uplo"))},
      {{"cublasDiagType_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::mkl::diag"))},
      {{"cublasSideMode_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::mkl::side"))},
      {{"cublasOperation_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::mkl::transpose"))},
      {{"cublasPointerMode_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cublasAtomicsMode_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cublasMath_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "blas::math_mode"))},
      {{"cusparsePointerMode_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cusparseFillMode_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::mkl::uplo"))},
      {{"cusparseDiagType_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::mkl::diag"))},
      {{"cusparseIndexBase_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::mkl::index_base"))},
      {{"cusparseMatrixType_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() +
                             "sparse::matrix_info::matrix_type",
                         HelperFeatureEnum::device_ext))},
      {{"cusparseOperation_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::mkl::transpose"))},
      {{"cusparseAlgMode_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cusparseSolveAnalysisInfo_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::shared_ptr<" + MapNames::getDpctNamespace() +
                             "sparse::optimize_info>",
                         HelperFeatureEnum::device_ext))},
      // {{"thrust::device_ptr"},
      //  clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "device_pointer",
      //                    HelperFeatureEnum::device_ext))},
      {{"thrust::device_reference"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "device_reference",
                         HelperFeatureEnum::device_ext))},
      {{"thrust::device_vector"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "device_vector",
                         HelperFeatureEnum::device_ext))},
      {{"thrust::device_malloc_allocator"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() +
                             "deprecated::usm_device_allocator",
                         HelperFeatureEnum::device_ext))},
      {{"thrust::maximum"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::dpl::maximum"))},
      {{"thrust::multiplies"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::multiplies"))},
      {{"thrust::plus"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::plus"))},
      {{"thrust::seq"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::dpl::execution::seq"))},
      {{"thrust::device"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::dpl::execution::dpcpp_default"))},
      {{"thrust::host"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::dpl::execution::seq"))},
      {{"thrust::minus"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::minus"))},
      {{"thrust::nullopt"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::nullopt"))},
      {{"thrust::greater"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::greater"))},
      {{"thrust::equal_to"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::dpl::equal_to"))},
      {{"thrust::less"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::dpl::less"))},
      {{"thrust::negate"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::negate"))},
      {{"thrust::logical_or"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::logical_or"))},
      {{"thrust::divides"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::divides"))},
      {{"thrust::tuple"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::tuple"))},
      {{"thrust::pair"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::pair"))},
      {{"thrust::host_vector"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::vector"))},
      {{"thrust::complex"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::complex"))},
      {{"thrust::counting_iterator"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::dpl::counting_iterator"))},
      {{"thrust::permutation_iterator"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::dpl::permutation_iterator"))},
      {{"thrust::transform_iterator"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::dpl::transform_iterator"))},
      {{"thrust::iterator_difference"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::iterator_traits"))},
      {{"thrust::tuple_element"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::tuple_element"))},
      {{"thrust::tuple_size"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::tuple_size"))},
      {{"thrust::swap"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::swap"))},
      {{"thrust::zip_iterator"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "zip_iterator",
                         HelperFeatureEnum::device_ext))},
      {{"cusolverDnHandle_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "queue_ptr"))},
      {{"cusolverEigType_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int64_t"))},
      {{"cusolverEigMode_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::mkl::job"))},
      {{"cusolverStatus_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cusolverDnParams_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"gesvdjInfo_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"syevjInfo_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cudaChannelFormatDesc"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "image_channel",
                         HelperFeatureEnum::device_ext))},
      {{"cudaChannelFormatKind"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "image_channel_data_type",
                         HelperFeatureEnum::device_ext))},
      {{"cudaArray"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(DpctGlobalInfo::useExtBindlessImages()
                             ? MapNames::getDpctNamespace() +
                                   "experimental::image_mem_wrapper"
                             : MapNames::getDpctNamespace() + "image_matrix",
                         HelperFeatureEnum::device_ext))},
      {{"cudaArray_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(DpctGlobalInfo::useExtBindlessImages()
                             ? MapNames::getDpctNamespace() +
                                   "experimental::image_mem_wrapper_ptr"
                             : MapNames::getDpctNamespace() + "image_matrix_p",
                         HelperFeatureEnum::device_ext))},
      {{"cudaMipmappedArray"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(DpctGlobalInfo::useExtBindlessImages()
                             ? MapNames::getDpctNamespace() +
                                   "experimental::image_mem_wrapper"
                             : "cudaMipmappedArray"))},
      {{"cudaMipmappedArray_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(DpctGlobalInfo::useExtBindlessImages()
                             ? MapNames::getDpctNamespace() +
                                   "experimental::image_mem_wrapper_ptr"
                             : "cudaMipmappedArray_t"))},
      {{"cudaTextureDesc"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "sampling_info",
                         HelperFeatureEnum::device_ext))},
      {{"cudaResourceDesc"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "image_data",
                         HelperFeatureEnum::device_ext))},
      {{"cudaTextureObject_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(
           DpctGlobalInfo::useExtBindlessImages()
               ? MapNames::getClNamespace() +
                     "ext::oneapi::experimental::sampled_image_handle"
               : MapNames::getDpctNamespace() + "image_wrapper_base_p",
           HelperFeatureEnum::device_ext))},
      {{"textureReference"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "image_wrapper_base",
                         HelperFeatureEnum::device_ext))},
      {{"cudaTextureAddressMode"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "addressing_mode"))},
      {{"cudaTextureFilterMode"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "filtering_mode"))},
      {{"curandGenerator_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "rng::host_rng_ptr",
                         HelperFeatureEnum::device_ext))},
      {{"curandRngType_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "rng::random_engine_type",
                         HelperFeatureEnum::device_ext))},
      {{"curandRngType"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "rng::random_engine_type",
                         HelperFeatureEnum::device_ext))},
      {{"curandStatus_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"curandStatus"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"curandOrdering_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "rng::random_mode"))},
      {{"cusparseStatus_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cusparseMatDescr_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::shared_ptr<" + MapNames::getDpctNamespace() +
                             "sparse::matrix_info>",
                         HelperFeatureEnum::device_ext))},
      {{"cusparseHandle_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "sparse::descriptor_ptr"))},
      {{"cudaMemoryAdvise"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cudaStreamCaptureStatus"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(DpctGlobalInfo::useExtGraph()
                             ? MapNames::getClNamespace() +
                                   "ext::oneapi::experimental::queue_state"
                             : "cudaStreamCaptureStatus"))},
      {{"CUmem_advise"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cudaPos"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "id<3>"))},
      {{"cudaExtent"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "range<3>"))},
      {{"cudaPitchedPtr"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "pitched_data",
                         HelperFeatureEnum::device_ext))},
      {{"cudaMemcpyKind"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "memcpy_direction"))},
      {{"CUDA_ARRAY3D_DESCRIPTOR"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(DpctGlobalInfo::useExtBindlessImages()
                             ? MapNames::getClNamespace() +
                                   "ext::oneapi::experimental::image_descriptor"
                             : MapNames::getDpctNamespace() + "image_matrix_desc"))},
      {{"CUDA_ARRAY_DESCRIPTOR"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(DpctGlobalInfo::useExtBindlessImages()
                             ? MapNames::getClNamespace() +
                                   "ext::oneapi::experimental::image_descriptor"
                             : MapNames::getDpctNamespace() + "image_matrix_desc"))},
      {{"cudaMemcpy3DParms"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "memcpy_parameter"))},
      {{"CUDA_MEMCPY3D"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "memcpy_parameter"))},
      {{"cudaMemcpy3DPeerParms"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "memcpy_parameter"))},
      {{"CUDA_MEMCPY3D_PEER"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "memcpy_parameter"))},
      {{"CUDA_MEMCPY2D"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "memcpy_parameter"))},
      {{"cudaComputeMode"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cudaSharedMemConfig"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cufftReal"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("float"))},
      {{"cufftDoubleReal"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("double"))},
      {{"cufftComplex"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "float2"))},
      {{"cufftDoubleComplex"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "double2"))},
      {{"cufftResult_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cufftResult"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cufftType_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "fft::fft_type",
                         HelperFeatureEnum::device_ext))},
      {{"cufftType"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "fft::fft_type",
                         HelperFeatureEnum::device_ext))},
      {{"cufftHandle"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "fft::fft_engine_ptr",
                         HelperFeatureEnum::device_ext))},
      {{"CUdevice"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"CUarray_st"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(DpctGlobalInfo::useExtBindlessImages()
                             ? MapNames::getDpctNamespace() +
                                   "experimental::image_mem_wrapper"
                             : MapNames::getDpctNamespace() + "image_matrix",
                         HelperFeatureEnum::device_ext))},
      {{"CUarray"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(DpctGlobalInfo::useExtBindlessImages()
                             ? MapNames::getDpctNamespace() +
                                   "experimental::image_mem_wrapper_ptr"
                             : MapNames::getDpctNamespace() + "image_matrix_p",
                         HelperFeatureEnum::device_ext))},
      {{"CUarray_format"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "image_channel_type"))},
      {{"CUarray_format_enum"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "image_channel_type"))},
      {{"CUtexObject"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "image_wrapper_base_p",
                         HelperFeatureEnum::device_ext))},
      {{"CUDA_RESOURCE_DESC"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "image_data",
                         HelperFeatureEnum::device_ext))},
      {{"CUDA_TEXTURE_DESC"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "sampling_info",
                         HelperFeatureEnum::device_ext))},
      {{"CUaddress_mode"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "addressing_mode"))},
      {{"CUaddress_mode_enum"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "addressing_mode"))},
      {{"CUfilter_mode"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "filtering_mode"))},
      {{"CUfilter_mode_enum"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "filtering_mode"))},
      {{"CUdeviceptr"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "device_ptr",
                         HelperFeatureEnum::device_ext))},
      {{"CUresourcetype_enum"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "image_data_type",
                         HelperFeatureEnum::device_ext))},
      {{"CUresourcetype"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "image_data_type",
                         HelperFeatureEnum::device_ext))},
      {{"cudaResourceType"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "image_data_type",
                         HelperFeatureEnum::device_ext))},
      {{"CUtexref"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "image_wrapper_base_p",
                         HelperFeatureEnum::device_ext))},
      {{"cudaDeviceAttr"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"__nv_bfloat16"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "ext::oneapi::bfloat16"))},
      {{"__nv_bfloat162"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "marray<" + MapNames::getClNamespace() +
                         "ext::oneapi::bfloat16, 2>"))},
      {{"nv_bfloat16"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "ext::oneapi::bfloat16"))},
      {{"nv_bfloat162"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getClNamespace() + "marray<" + MapNames::getClNamespace() +
                         "ext::oneapi::bfloat16, 2>"))},
      {{"libraryPropertyType_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "version_field",
                         HelperFeatureEnum::device_ext))},
      {{"libraryPropertyType"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "version_field",
                         HelperFeatureEnum::device_ext))},
      {{"ncclUniqueId"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::ccl::kvs::address_type",
                         HelperFeatureEnum::device_ext))},
      {{"ncclComm_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "ccl::comm_ptr",
                         HelperFeatureEnum::device_ext))},
      {{"ncclRedOp_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::ccl::reduction"))},
      {{"ncclDataType_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::ccl::datatype"))},
      {{"cuda::std::tuple"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::tuple"))},
      {{"cuda::std::complex"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::complex"))},
      {{"cuda::std::array"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::array"))},
      {{"cusolverEigRange_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::mkl::rangev"))},
      {{"cudaUUID_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::array<unsigned char, 16>"))},
      {{"CUuuid"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::array<unsigned char, 16>"))},
      {{"cusparseIndexType_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "library_data_t"))},
      {{"cusparseFormat_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "sparse::matrix_format"))},
      {{"cusparseDnMatDescr_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::shared_ptr<" + MapNames::getDpctNamespace() +
                         "sparse::dense_matrix_desc>"))},
      {{"cusparseConstDnMatDescr_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::shared_ptr<" + MapNames::getDpctNamespace() +
                         "sparse::dense_matrix_desc>"))},
      {{"cusparseOrder_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::mkl::layout"))},
      {{"cusparseDnVecDescr_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::shared_ptr<" + MapNames::getDpctNamespace() +
                         "sparse::dense_vector_desc>"))},
      {{"cusparseConstDnVecDescr_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator("std::shared_ptr<" + MapNames::getDpctNamespace() +
                         "sparse::dense_vector_desc>"))},
      {{"cusparseSpMatDescr_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "sparse::sparse_matrix_desc_t"))},
      {{"cusparseConstSpMatDescr_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() + "sparse::sparse_matrix_desc_t"))},
      {{"cusparseSpMMAlg_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cusparseSpMVAlg_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cusolverDnFunction_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cusolverAlgMode_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cusparseSpGEMMDescr_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator("oneapi::mkl::sparse::matmat_descr_t"))},
      {{"cusparseSpSVDescr_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cusparseSpGEMMAlg_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cusparseSpSVAlg_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"__half_raw"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("uint16_t"))},
      {{"cudaFuncAttributes"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::MapNames::getDpctNamespace() +
                         "kernel_function_info"))},
      {{"ncclResult_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cudaLaunchAttributeValue"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cusparseSpSMDescr_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cusparseSpSMAlg_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cublasLtHandle_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() +
                         "blas_gemm::experimental::descriptor_ptr"))},
      {{"cublasLtMatmulDesc_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() +
                         "blas_gemm::experimental::matmul_desc_ptr"))},
      {{"cublasLtOrder_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() +
                         "blas_gemm::experimental::order_t"))},
      {{"cublasLtPointerMode_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() +
                         "blas_gemm::experimental::pointer_mode_t"))},
      {{"cublasLtMatrixLayout_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() +
                         "blas_gemm::experimental::matrix_layout_ptr"))},
      {{"cublasLtMatrixLayoutAttribute_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(
           MapNames::getDpctNamespace() +
           "blas_gemm::experimental::matrix_layout_t::attribute"))},
      {{"cublasLtMatmulDescAttributes_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() +
                         "blas_gemm::experimental::matmul_desc_t::attribute"))},
      {{"cublasLtMatmulAlgo_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cublasLtEpilogue_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() +
                         "blas_gemm::experimental::epilogue_t"))},
      {{"cublasLtMatmulPreference_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cublasLtMatmulHeuristicResult_t"}, clang::dpct::createTypeLocRewriterFactory(makeStringCreator("int"))},
      {{"cublasLtMatrixTransformDesc_t"},
       clang::dpct::createTypeLocRewriterFactory(makeStringCreator(MapNames::getDpctNamespace() +
                         "blas_gemm::experimental::transform_desc_ptr"))},
      // ...

  }));
}
} // namespace dpct
} // namespace clang
