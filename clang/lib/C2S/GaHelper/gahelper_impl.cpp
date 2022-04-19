//===--- gahelper_impl.cpp-------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
#include "gahelper_impl.h"
#include "uuid.h"
#include "os_specific.h"
#include "Config.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>


GAHELPER_NS_BEGIN

#define STR(x) STR_X(x)
#define STR_X(x) #x

ustring getStatisticDir()
{
    /*auto productLocations = cfgmgr2::IProductLocations::get();
    if (productLocations == nullptr)
    {
        return _U("");
    }*/

#ifdef _WIN32
    WCHAR  buf[MAX_PATH];
    HRESULT hr = SHGetFolderPathW(NULL, CSIDL_LOCAL_APPDATA, NULL, SHGFP_TYPE_CURRENT, buf);
    if FAILED(hr)
    {
        return _U("");
    }
//    auto intelRootPath = gh2::bpath(buf) / _U("Intel Corporation");
	ustring intelRootPath(ustring(buf)+ ustring(_U("/")) + ustring(_U("Intel Corporation")));

#else
    char* home = getenv("HOME");
    // Ignore this environment variable if its length is greater than or euqal to INT_MAX
    if (home && std::find(home, home + INT_MAX, 0) - home >= INT_MAX)
      home = nullptr;
    ustring homePath(home ? home : _U("."));
    auto intelRootPath = homePath + "/" + _U("intel");
#endif
    //TODO: update intelRootPath from env variable just as in cfgmgr in order
	  //fixme: exception disabled in disabled when build with dpct.
    //to have this directory sandboxed for testing purposes.
    //try
    //{
        auto statisticDir = ustring(intelRootPath) + ustring(_U("/")) + ustring(_U("DPCPPCT")) ;
        createDirectories(statisticDir);
        return statisticDir;
    //}
    //catch(...)
    //{
    //    return _U("");
    //}
}

///////////////////////////////////////////////////////////////////////////////

UuidPersistenceProvider::UuidPersistenceProvider(const ustring rootDir)
{
    auto statDir = getStatisticDir();
    if (statDir.empty())
    {
        //can not create full path to statistic directory (readonly fs for example)
        return;
    }
    m_uuidFilePath = (statDir) + _U("/") + _U("uuid.xml");
}


bool UuidPersistenceProvider::store(const char * uuid)
{
    if (m_uuidFilePath.empty())
    {
        return false;
    }
    m_uuid = uuid;
    
    std::ofstream ostream(m_uuidFilePath);
    if (ostream)
    {
        std::string contents;
        ostream << m_uuid;
        return true;
    }
    //TODO: add locking here ,  save  uuid to file.
    //return gh2::ecgOk == gh2::save_variant_bag2(bag, m_uuidFilePath.c_str());
    return false;
}

const char * UuidPersistenceProvider::load()
{
   
    std::ifstream stream(m_uuidFilePath);
    if (stream)
    {
        stream >> m_uuid;
       
    } else {
        m_uuid = "";
    }
    //TODO: add locking here, load frim file
    return m_uuid.c_str();
}

ActiveUserPersistenceProvider::ActiveUserPersistenceProvider(const ustring rootDir)
{
    auto statDir = getStatisticDir();
    if (statDir.empty())
    {
        return;
    }
    m_activeUserFilePath = statDir + _U("/") + _U("last_active_dates.xml");
}

ActiveUserPersistenceProvider::~ActiveUserPersistenceProvider()
{
}

bool ActiveUserPersistenceProvider::read(ReaderCallback * callback)
{
    
    std::ifstream stream(m_activeUserFilePath);
    if (stream)
    {
        std::string line;
        while (std::getline(stream, line))
        {
            int year = 0, month = 0, day = 0;
            std::istringstream iss(line);
            if (!(iss >> year >> month >> day)) {
                break; 
            }
            callback->onItem(year, month, day);
        }
        return true;
        
    } else {
       return false;
    }
    
    /*gh2::variant_bag_t bag;
    filelock_t lock(m_activeUserFilePath);
    for (int i = 0; i < 5; i++)
    {
        if (lock.lock())
        {
            break;
        }
#ifdef _WIN32
        Sleep(200);
#else
        usleep(20000);
#endif
    }
    if (!lock.is_locked())
    {
        return false;
    }
    if (gh2::ecgOk != gh2::load_variant_bag2(bag, m_activeUserFilePath.c_str()))
    {
        return false;
    }
    for (auto it = bag.begin_by_name<gh2::variant_bag_t>(ITEM_TAG); !it.at_end(); it.next())
    {
        int year, month, day;
        auto& item = it.get_value();
        if (getVariantInt(item, YEAR_TAG, &year) &&
            getVariantInt(item, MONTH_TAG, &month) &&
            getVariantInt(item, DAY_TAG, &day))
        {
            callback->onItem(year, month, day);
        }
    }*/
    return true;
}

bool ActiveUserPersistenceProvider::write(WriterCallback * callback)
{
    std::ofstream stream(m_activeUserFilePath);
    if (stream)
    {
        int year, month, day;
        while (callback->getItem(&year, &month, &day)) {
            stream<< year << " " << month << " " << day << "\n"; 
        }
        return true;
    } else {
       return false;
    }

/*
    gh2::variant_bag_t bag;
    int year, month, day;

    while (callback->getItem(&year, &month, &day))
    {
        gh2::variant_bag_t& dateEntry = bag.add_variant_bag(ITEM_TAG);
        dateEntry.put<gh2::variant_t>(YEAR_TAG, year);
        dateEntry.put<gh2::variant_t>(MONTH_TAG, month);
        dateEntry.put<gh2::variant_t>(DAY_TAG, day);
    }
    filelock_t lock(m_activeUserFilePath);
    //If we can not lock, probably another active product instance writing, so we just
    //can skip write, because file will have same contents
    if (lock.lock())
    {
        return gh2::ecgOk != gh2::save_variant_bag2(bag, m_activeUserFilePath.c_str());
    }*/
}

AnalyticsImpl::AnalyticsImpl(const AnalyticsCreateParams& params) :
    m_connector(params.overrideCollectionUrl ? params.overrideCollectionUrl : "http://www.google-analytics.com/collect")
{
    m_enabled = false;
    m_flags = params.flags;
    // auto uuid = params.uuidPersistenceProvider->load();
    // m_uuid = uuid ? uuid : "";
    // if (m_uuid.empty())
    // {
    m_uuid = generate_uuid4();
    // params.uuidPersistenceProvider->store(m_uuid.c_str());
    // }
    m_osVersion = getOsName();
    #ifdef _WIN32
    m_osType = STR(Windows);
    #else
    m_osType = STR(Linux);
    #endif
    std::stringstream ss;
    ss << "v=1&tid=" << params.tid << "&cid=" << m_uuid;
    ss << "&an=" << m_connector.urlencode(params.appName);
    ss << "&av=" << m_connector.urlencode(params.appVersion);
    ss << "&cd1=" << m_connector.urlencode(m_osType);
    ss << "&cd2=" << m_connector.urlencode(m_osVersion);
    m_constantPostFields = ss.str();
    m_isipFilePath = getIsipFilePath();
    m_enabled = isEnabled();
}

UserConsent AnalyticsImpl::getUserConsentValue() const
{
    if (m_isipFilePath.empty())
    {
        // unable to find where consent file is located. For example HOME env var is not set
        // on linux.
        return UserConsent::optedOut;
    }
    std::ifstream stream(m_isipFilePath);
    if (stream)
    {
        std::string contents;
        stream >> contents;
        return (contents == "1") ? UserConsent::optedIn : UserConsent::optedOut;
    }
    return UserConsent::pending;
}

void AnalyticsImpl::setStatisticCollectionEnabled(bool value)
{
    createDirectories(dirname(m_isipFilePath));
    std::ofstream stream(m_isipFilePath);
    if (stream)
    {
        stream << (value ? "1" : "0");
    }
}

void AnalyticsImpl::postEvent(const char *category, const char *action,
                              const char *label, const std::string &Data,
                              int value) {
    if (!m_enabled)
    {
        return;
    }
    // m_activeUserDetector.notify_activity();
    std::stringstream ss;
    ss << m_constantPostFields;
    ss << "&t=event";
    if (category)
    {
        ss << "&ec=" << m_connector.urlencode(category);
    }
    if (action)
    {
        ss << "&ea=" << m_connector.urlencode(action);
    }
    if (label)
    {
        ss << "&el=" << m_connector.urlencode(label);
    }
    if (value)
    {
        ss << "&ev=" << value;
    }
    ss << "&cd3=y";
    // We use the "cd4" field to save and post the API info
    if (!Data.empty()) {
      ss << "&cd4=" << m_connector.urlencode(Data);
    }
    m_connector.post(ss.str());
}

/*
From: : Boikov, Pavel
Hi All,
Let define the following files for storing opt-in status flag:
Windows:
"%LOCALAPPDATA%\Intel Corporation\isip"
Example: "C:\Users\pboikov\AppData\Local\Intel Corporation\isip"
Linux, mac:
~/intel/isip

File name: isip
File type: Plain text, ANSII
File size: 1 byte
File content:
'1' - ASCII char - allow to send statistic
'0' - ASCII char - do not allow to send statistic
*/

#ifdef _WIN32
ustring AnalyticsImpl::getIsipFilePath()
{
    ustring ret;
    HRESULT hr;
    PWSTR path;
    hr = SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, NULL, &path);
    if (hr == S_OK)
    {
        ret = path; //path from SHGetKnownFolderPath does not contain trailing backslash as documented
        ret += L"\\Intel Corporation\\isip";
        CoTaskMemFree(path);
    }
    return ret;
}

#else
ustring AnalyticsImpl::getIsipFilePath()
{
    ustring ret;
    auto home = getenv("HOME");
    if (home)
    {
        ret = home;
        if (!ret.empty())
        {
            if (*ret.rbegin() != '/')
            {
                ret += '/';
            }
            ret += "intel/isip";
        }
    }
    return ret;
}
#endif

bool AnalyticsImpl::isEnabled()
{
    for (auto envName : { "INTEL_STATISTICS_DISABLE", "INTEL_DONOTSEND_FEEDBACK", "INTEL_DISABLE_FEEDBACK" })
    {
        if (getenv(envName))
        {
            return false;
        }
    }
    switch (getNetworkStatus())
    {
    case network_status_t::inside_intel_network://do not want to spoil statistic by testing etc.
        if (m_flags & ALLOW_COLLECTION_FROM_INTEL_NETWORK)
        {
            break;
        }
        return false;
    case network_status_t::network_error://looks like name resolution fail, attempt to send statistic will fail also
        return false;
    case network_status_t::outside_intel_network:
        break;
    }
    return getUserConsentValue() == UserConsent::optedIn;
}

void AnalyticsImpl::dumpStat() {
    std::cout<< "\n------------------------------\n";
    std::cout<< "m_osType: " << m_osType << "\n";
    std::cout<< "m_osVersion: " << m_osVersion << "\n";
    std::cout<< "m_constantPostFields: " << m_constantPostFields << "\n";
    std::cout<< "m_uuid: " << m_uuid << "\n";
#ifdef _WIN32
    std::wcout<< "m_isipFilePath: " << m_isipFilePath.c_str() << "\n";
#else
    std::cout << "m_isipFilePath: " << m_isipFilePath.c_str() << "\n";
#endif
    std::cout<< "m_flags: " << m_flags << "\n";
    std::cout<< "m_enabled: " << m_enabled << "\n";
    // m_activeUserDetector.dumpStat();
    m_connector.dumpStats();
    std::cout<< "------------------------------\n\n";

}

//#ifdef GAHELPER_EXPORTS
static AnalyticsImpl &getAnalyticsImplInstance() {
  // UuidPersistenceProvider uuidProvider(_U(""));
  // ActiveUserPersistenceProvider activeUserPersistenceProvider(_U(""));
  AnalyticsCreateParams params = {0};
  params.flags = ALLOW_COLLECTION_FROM_INTEL_NETWORK;

  params.appName = "Intel(R) DPC++ Compatibility Tool";
  std::string AppVersion = std::string(DPCT_VERSION_MAJOR) + "." +
                           std::string(DPCT_VERSION_MINOR) + "." +
                           std::string(DPCT_VERSION_PATCH);
  params.appVersion = AppVersion.c_str();
  params.tid = "UA-17808594-22"; // this one is Analyzers GA sandbox
  // params.uuidPersistenceProvider = &uuidProvider;
  // params.activeUserPersistenceProvider = &activeUserPersistenceProvider;
  static AnalyticsImpl AImpl(params);
  return AImpl;
}

IAnalytics *IAnalytics ::create() { return &getAnalyticsImplInstance(); }

//void AnalyticsImpl::destroy()
//{
//    delete this;
//}

//#endif

GAHELPER_NS_END
