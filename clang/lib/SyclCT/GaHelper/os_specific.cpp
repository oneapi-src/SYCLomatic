//===--- os_specific.cpp-------------------------*- C++ -*---===//
//
// Copyright (C) 2018-1019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
#include <iostream>
#include <fstream>
#include "os_specific.h"


//TODO:  remove the macro to enable windows support.
#if defined(__linux__)
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
#endif
