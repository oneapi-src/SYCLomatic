//===--- GaHelper.h-------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
#pragma once
#include "GaNamespace.h"
#include <stdint.h>
#include <string>

/*
#ifdef _WIN32
#  ifdef GAHELPER_EXPORTS
#    define GAHELPER_API __declspec(dllexport)
#  else
#    define GAHELPER_API __declspec(dllimport)
#  endif
#else
#  define GAHELPER_API
#endif
*/

GAHELPER_NS_BEGIN

class IUuidPersistenceProvider
{
public:
    // implementation should provide appropriate locking during operation,
    // because several processes started (even on different computers having share HOME directory) can cause 
    // race condition.
    virtual bool store(const char* uuid) = 0;
    virtual const char* load() = 0; //returned pointer should be valid till next load() call
};

class IActiveUserPersistenceProvider
{
public:
    class WriterCallback
    {
    public:
        virtual bool getItem(int* outYear, int* outMon, int *outDay) = 0;
    };
    class ReaderCallback
    {
    public:
        virtual bool onItem(int year, int mon, int day) = 0;
    };
    //TODO: Add actual load store methods (thinking aon appropriate method sinatures still)
    virtual bool read(ReaderCallback* callback) = 0;
    virtual bool write(WriterCallback* callback) = 0;
};


const uint64_t ALLOW_COLLECTION_FROM_INTEL_NETWORK = 1;

struct AnalyticsCreateParams
{
    uint64_t flags;
    //Tracking ID / Property ID. (UA-XXXXX)
    const char* tid;
    //Application name (e.g. "amplxe" for Amplifier)
    const char* appName;
    //Application version string (e.g. "19.3")
    const char* appVersion;
    //Caller implemented persistence providers
    IUuidPersistenceProvider* uuidPersistenceProvider;
    IActiveUserPersistenceProvider* activeUserPersistenceProvider;
    //Used for testing purposes only. If null default GA url is used.
    const char* overrideCollectionUrl;
};

enum class UserConsent
{
    pending,
    optedIn,
    optedOut
};

class IAnalytics
{
public:
    static IAnalytics* create();
    // virtual void destroy() = 0;
    virtual UserConsent getUserConsentValue() const = 0;
    virtual void setStatisticCollectionEnabled(bool value) = 0;
    // In Analysers events sent surrently with categry filled by event name such as "client.gui.start"
    // and category="run" with label=<ANALYSIS_TYPE_NAME> for starting collection
    // 3 custom field are sent with each event: OS, OS Version, IsActiveUser along with standard dimensions
    // for application name, application version, and user UUID
    virtual void postEvent(const char *category, const char *action, const char *label, const std::string &Data, int value = 0) = 0;
    virtual void  dumpStat() =0;
};

GAHELPER_NS_END
