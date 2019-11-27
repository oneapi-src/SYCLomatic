//===---MigrationRules.inc------------------------------------------===//
//
// Copyright (C) 2018 - 2019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
RULE(IterationSpaceBuiltinRule)
RULE(ErrorHandlingIfStmtRule)
RULE(AlignAttrsRule)
RULE(AtomicFunctionRule)
RULE(TypeInDeclRule)
RULE(VectorTypeNamespaceRule)
RULE(VectorTypeMemberAccessRule)
RULE(VectorTypeOperatorRule)
RULE(VectorTypeCtorRule)
RULE(ReplaceDim3CtorRule)
RULE(Dim3MemberFieldsRule)
RULE(ReturnTypeRule)
RULE(DevicePropVarRule)
RULE(EnumConstantRule)
RULE(ErrorConstantsRule)
RULE(FunctionCallRule)
RULE(KernelCallRule)
RULE(SharedMemVarRule)
RULE(ConstantMemVarRule)
RULE(DeviceMemVarRule)
RULE(MemoryMigrationRule)
RULE(ErrorTryCatchRule)
RULE(KernelIterationSpaceRule)
RULE(UnnamedTypesRule)
RULE(MathFunctionsRule)
RULE(SyncThreadsRule)
RULE(KernelFunctionInfoRule)
RULE(TypeCastRule)
RULE(EventAPICallRule)
RULE(StreamAPICallRule)
RULE(BLASEnumsRule)
RULE(BLASFunctionCallRule)
RULE(SOLVEREnumsRule)
RULE(SOLVERFunctionCallRule)
RULE(TextureRule)
RUlE(CXXNewExprRule)
RULE(ErrorHandlingHostAPIRule)
