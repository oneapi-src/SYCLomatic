//===--- GAnalytics.cpp-------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
#include "GaHelper.h"
#include "gahelper_impl.h"

#include <iostream>
#include <string>

using namespace gahelper1;

int GAnalytics(std::string Data) {
  auto analytics = IAnalytics::create();
  if (analytics->getUserConsentValue() == UserConsent::optedIn) {
    // std::cout << "send message .....\n";
    if (Data == "") {
      analytics->postEvent("client.cli.start", "start", nullptr, "");
    } else {
      analytics->postEvent("client.cli.start", "API", nullptr, Data);
    }
  }
  // analytics->dumpStat();
  // analytics->destroy();
  return 0;
}

#ifdef GA_STANDALONE_TEST
int main(int argc, const char *argv[]) { GAnalytics(); }
#endif
