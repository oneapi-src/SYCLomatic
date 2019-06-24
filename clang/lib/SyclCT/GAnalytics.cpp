//===--- GAnalytics.cpp-------------------------*- C++ -*---===//
//
// Copyright (C) 2018-1019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
#if defined(__linux__)
#include "GaHelper.h"
#include "gahelper_impl.h"
#include "Config.h"

#include <iostream>
#include <string>

using namespace gahelper1;
int GAnalytics() {
  UuidPersistenceProvider uuidProvider("");
  ActiveUserPersistenceProvider activeUserPersistenceProvider("");
  AnalyticsCreateParams params = {0};
  params.flags=ALLOW_COLLECTION_FROM_INTEL_NETWORK;

  params.appName = "DPC++ Compatibility Tool";
  std::string AppVersion= std::to_string(SYCLCT_VERSION_MAJOR) + "." +
     std::to_string(SYCLCT_VERSION_MINOR) + "." + std::to_string(SYCLCT_VERSION_PATCH);
  params.appVersion = AppVersion.c_str();
  params.tid = "UA-17808594-22"; // this one is Analyzers GA sandbox
  params.uuidPersistenceProvider = &uuidProvider;
  params.activeUserPersistenceProvider = &activeUserPersistenceProvider;
  auto analytics = IAnalytics::create(params);
  if (analytics->getUserConsentValue() == UserConsent::optedIn) {
      //std::cout << "send message .....\n";
      analytics->postEvent("client.cli.start", nullptr, nullptr);
      analytics->postEvent("client.cli.finish", nullptr, nullptr);

  }
  //analytics->dumpStat();
 
  analytics->destroy();
  return 0;

}

#ifdef  GA_STANDALONE_TEST
int main(int argc, const char *argv[]) {
    GAnalytics();
}
#endif 

#endif
