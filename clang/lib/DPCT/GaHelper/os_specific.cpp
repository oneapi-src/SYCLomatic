//===--- os_specific.cpp-------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
#if defined(__linux__)
#include <iostream>
#include <fstream>
#include "os_specific.h"


GAHELPER_NS_BEGIN

namespace {

void lstripChars(std::string& str, const char* what)
{
    auto found = str.find_first_not_of(what);
    if (found != std::string::npos)
    {
        str.erase(0, found);
    }
}

void rstripChars(std::string& str, const char* what)
{
    auto found = str.find_last_not_of(what);
    if (found != std::string::npos)
    {
        str.erase(found + 1);
    }
    else
    {
        str.clear();
    }
}

}

void collectProxyInfo(const char* for_url, std::vector<std::string>* result)
{
    const char* envvar = getenv("http_proxy");
    if (envvar)
    {
        result->push_back(envvar);
    }
}

std::string get_name_from_etc_redhat_release()
{
    std::string result;
    const char* path = "/etc/redhat-release";
    std::ifstream stream(path);
    if (stream)
    {
        getline(stream, result);
    }
    return result;

}

std::string get_name_from_etc_os_release()
{
    std::string result;
    std::string line;

    const char* path = "/etc/os-release";
    std::string start("PRETTY_NAME=");
    std::ifstream stream(path);
    if (stream)
    {
        while (getline(stream, line))
        {
            if (line.substr(0, start.size()) == start)
            {
                result = line.substr(start.size(), std::string::npos);
                break;
            }
        }
    }
    //Now let's strip unneeded characters
    lstripChars(result, " \"");
    rstripChars(result, " \"");
    return result;
}

std::string getOsName()
{
    std::string result;
    result = get_name_from_etc_os_release();
    if (!result.empty())
    {
        return result;
    }
    result = get_name_from_etc_redhat_release();
    if (!result.empty())
    {
        return result;
    }
    result = "Undefined Linux Version";
    return result;
}

GAHELPER_NS_END
#else


#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winhttp.h>
#include "os_specific.h"
#include <memory>

GAHELPER_NS_BEGIN

std::string toUtf8(const wchar_t* input)
{
    auto requiredSize = WideCharToMultiByte(CP_UTF8, 0, input, -1, nullptr, 0, nullptr, nullptr);
    std::unique_ptr<char> buffer(new char[requiredSize]); //does + 1 needed for 0 ???
    WideCharToMultiByte(CP_UTF8, 0, input, -1, buffer.get(), requiredSize, nullptr, nullptr);
    return buffer.get();
}

std::wstring toUtf16(const char* input)
{
    auto requiredSize = MultiByteToWideChar(CP_UTF8, 0, input, -1, nullptr, 0);
    std::unique_ptr<wchar_t> buffer(new wchar_t[requiredSize]);
    MultiByteToWideChar(CP_UTF8, 0, input, -1, buffer.get(), requiredSize);
    return buffer.get();
}

std::string getOsName()
{
    DWORD type;
    wchar_t data[256];
    DWORD data_size = sizeof(data);
    HKEY hkey;
    std::string result = "Undefined Windows Version";
    LSTATUS status = RegOpenKeyExW(HKEY_LOCAL_MACHINE,
        L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", NULL, KEY_QUERY_VALUE, &hkey);
    if (status == ERROR_SUCCESS)
    {
        status = RegQueryValueExW(hkey, L"ProductName", NULL, &type,
            (LPBYTE)(data), &data_size);
        if (status == ERROR_SUCCESS && type == REG_SZ)
        {
            result = toUtf8(data);
        }
        RegCloseKey(hkey);
    }
    return result;
}

void safe_global_free(LPWSTR& ptr)
{
    if (ptr)
    {
        GlobalFree(ptr);
        ptr = NULL;
    }
}

struct WINHTTP_PROXY_INFO_WRAPPER : public WINHTTP_PROXY_INFO
{
    WINHTTP_PROXY_INFO_WRAPPER()
    {
        ZeroMemory(this, sizeof(*this));
    }
    ~WINHTTP_PROXY_INFO_WRAPPER()
    {
        clear();
    }
    void clear()
    {
        safe_global_free(lpszProxy);
        safe_global_free(lpszProxyBypass);
    }
};

void collectProxyInfo(const char* for_url, std::vector<std::string>* result)
{
    std::wstring url(toUtf16(for_url));
    WINHTTP_PROXY_INFO_WRAPPER pi;
    WINHTTP_AUTOPROXY_OPTIONS apo;

    apo.dwFlags = WINHTTP_AUTOPROXY_AUTO_DETECT;
    apo.dwAutoDetectFlags = WINHTTP_AUTO_DETECT_TYPE_DHCP | WINHTTP_AUTO_DETECT_TYPE_DNS_A;
    apo.lpszAutoConfigUrl = NULL;
    apo.lpvReserved = NULL;
    apo.dwReserved = 0;
    apo.fAutoLogonIfChallenged = TRUE;

    WINHTTP_CURRENT_USER_IE_PROXY_CONFIG iepi;
    if (WinHttpGetIEProxyConfigForCurrentUser(&iepi))
    {
        if (iepi.lpszProxy)
        {
            result->push_back(toUtf8(iepi.lpszProxy));
        }
        safe_global_free(iepi.lpszProxy);
        safe_global_free(iepi.lpszProxyBypass);
        safe_global_free(iepi.lpszAutoConfigUrl);
    }

    HINTERNET hSession = WinHttpOpen(L"",
        WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, 
        WINHTTP_NO_PROXY_NAME,
        WINHTTP_NO_PROXY_BYPASS,
        0);
    if (hSession)
    {
        if (WinHttpGetProxyForUrl(hSession, url.c_str(), &apo, &pi) && pi.lpszProxy && pi.dwAccessType == WINHTTP_ACCESS_TYPE_NAMED_PROXY)
        {
            result->push_back(toUtf8(pi.lpszProxy));
        }
        WinHttpCloseHandle(hSession);
    }
    
    pi.clear();
    // Retrieve the default proxy configuration.
    if (WinHttpGetDefaultProxyConfiguration(&pi) && pi.lpszProxy && pi.dwAccessType == WINHTTP_ACCESS_TYPE_NAMED_PROXY)
    {
        result->push_back(toUtf8(pi.lpszProxy));
    }

    const char* envvar = getenv("http_proxy");
    if (envvar && result->size() == 0)
    {
        result->push_back(envvar);
    }
}

GAHELPER_NS_END
#endif

