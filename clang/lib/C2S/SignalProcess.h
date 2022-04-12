//===--- SignalProcess.h -----------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===---------------------- --------------------------------------===//

#ifndef C2S_SIGNAL_PROCESS_H
#define C2S_SIGNAL_PROCESS_H

#if defined(__linux__) || defined(_WIN64)
void InstallSignalHandle(void);
#endif

#endif
