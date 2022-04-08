//===--- gahelper_impl.h-------------------------*- C++ -*---===//
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
#include "GaHelper.h"
#include "http_connector.h"
#include "network_util.h"
#include "filesystem_util.h"
#include "active_user_detector.h"

#ifdef _WIN32
#include <Windows.h>
#include <Shlobj.h>
#else
#endif
#include <string>


GAHELPER_NS_BEGIN

	
class UuidPersistenceProvider : public IUuidPersistenceProvider
{
public:
	UuidPersistenceProvider(const ustring rootDir);
	virtual bool store(const char* uuid) override;
	virtual const char* load() override;
private:
	std::string m_uuid;
	ustring m_uuidFilePath;
};

class ActiveUserPersistenceProvider : public IActiveUserPersistenceProvider
{
public:
	ActiveUserPersistenceProvider(const ustring rootDir);
	~ActiveUserPersistenceProvider();
	virtual bool read(ReaderCallback* callback) override;
	virtual bool write(WriterCallback* callback) override;
private:
	ustring m_activeUserFilePath;
};

class AnalyticsImpl : public IAnalytics
{
public:
    AnalyticsImpl(const AnalyticsCreateParams& params);
    // virtual void destroy() override;
    virtual UserConsent getUserConsentValue() const override;
    virtual void setStatisticCollectionEnabled(bool value) override;
    virtual void postEvent(const char *category, const char *action, const char *label, const std::string &Data, int value = 0) override;
    virtual void  dumpStat() override;
private:
    ustring getIsipFilePath();
    bool isEnabled();
private:
    HttpConnector m_connector;
    // ActiveUserDetector m_activeUserDetector;
    std::string m_osType;
    std::string m_osVersion;
    std::string m_constantPostFields;
    std::string m_uuid;
    ustring m_isipFilePath;
    uint64_t m_flags;
    bool m_enabled;
};


GAHELPER_NS_END
