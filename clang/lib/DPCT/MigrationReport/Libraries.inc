#ifndef LIBRARY
#define LIBRARY
#endif

LIBRARY(MKL, "oneAPI Math Kernel Library (oneMKL)", RuleGroupKind::RK_Sparse,
        RuleGroupKind::RK_BLas, RuleGroupKind::RK_Solver, RuleGroupKind::RK_Rng,
        RuleGroupKind::RK_FFT)

LIBRARY(DPL, "oneAPI DPC++ Library (oneDPL)", RuleGroupKind::RK_Thrust,
        RuleGroupKind::RK_CUB)

LIBRARY(DNN, "oneAPI Deep Neural Network Library (oneDNN)",
        RuleGroupKind::RK_DNN)

LIBRARY(CCL, "oneAPI Collective Communications Library (oneCCL)",
        RuleGroupKind::RK_NCCL)

#undef LIBRARY