//===--------------- SignalProcess.h --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DPCT_SIGNAL_PROCESS_H
#define DPCT_SIGNAL_PROCESS_H

#if defined(__linux__) || defined(_WIN64)
void InstallSignalHandle(void);
#endif

#endif
