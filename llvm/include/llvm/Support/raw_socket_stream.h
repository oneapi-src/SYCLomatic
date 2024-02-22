//===-- llvm/Support/raw_socket_stream.h - Socket streams --*- C++ -*-===//
// SYCLomatic_CUSTOMIZATION
//
// INTEL CONFIDENTIAL
//
// Modifications, Copyright (C) 2024 Intel Corporation
//
// This software and the related documents are Intel copyrighted materials, and
// your use of them is governed by the express license under which they were
// provided to you ("License"). Unless the License provides otherwise, you may
// not use, modify, copy, publish, distribute, disclose or transmit this
// software or the related documents without Intel's prior written permission.
//
// This software and the related documents are provided as is, with no express
// or implied warranties, other than those that are expressly stated in the
// License.
//
// end SYCLomatic_CUSTOMIZATION
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains raw_ostream implementations for streams to communicate
// via UNIX sockets
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_RAW_SOCKET_STREAM_H
#define LLVM_SUPPORT_RAW_SOCKET_STREAM_H

#if SYCLomatic_CUSTOMIZATION

// This is here to trigger failures when pulling down any changes that
// try to use sockets. See CMPLRLLVM-55472.
#error "Socket support is disabled for Intel builds."

#else // SYCLomatic_CUSTOMIZATION

#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class raw_socket_stream;

// Make sure that calls to WSAStartup and WSACleanup are balanced.
#ifdef _WIN32
class WSABalancer {
public:
  WSABalancer();
  ~WSABalancer();
};
#endif // _WIN32

class ListeningSocket {
  int FD;
  std::string SocketPath;
  ListeningSocket(int SocketFD, StringRef SocketPath);
#ifdef _WIN32
  WSABalancer _;
#endif // _WIN32

public:
  static Expected<ListeningSocket> createUnix(
      StringRef SocketPath,
      int MaxBacklog = llvm::hardware_concurrency().compute_thread_count());
  Expected<std::unique_ptr<raw_socket_stream>> accept();
  ListeningSocket(ListeningSocket &&LS);
  ~ListeningSocket();
};
class raw_socket_stream : public raw_fd_stream {
  uint64_t current_pos() const override { return 0; }
#ifdef _WIN32
  WSABalancer _;
#endif // _WIN32

public:
  raw_socket_stream(int SocketFD);
  /// Create a \p raw_socket_stream connected to the Unix domain socket at \p
  /// SocketPath.
  static Expected<std::unique_ptr<raw_socket_stream>>
  createConnectedUnix(StringRef SocketPath);
  ~raw_socket_stream();
};

} // end namespace llvm

#endif // SYCLomatic_CUSTOMIZATION

#endif
