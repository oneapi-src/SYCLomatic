//===---  filesystem_util.cpp-----------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
#include "filesystem_util.h"
#ifdef _WIN32
#  include <direct.h>
#else
#include <bits/stdc++.h> 
#include <iostream> 
#include <sys/stat.h> 
#include <sys/types.h> 
#endif


GAHELPER_NS_BEGIN

ustring dirname(const ustring & path)
{
    auto pos = ustring::npos;
    auto pos1 = path.rfind(_U("\\"));
    if (pos1 != ustring::npos)
    {
        pos = pos1;
    }
    pos1 = path.rfind(_U("/"));
    if (pos1 != ustring::npos)
    {
        if (pos == ustring::npos || pos1 > pos)
        {
            pos = pos1;
        }
    }
    if (pos == ustring::npos)
    {
        return _U("");
    }
    return path.substr(0, pos);
}

bool createDirectories(const ustring& path)
{
    if (path.empty())
    {
        return false;
    }
#ifdef _WIN32
    auto ret = _wmkdir(path.c_str());
    if (ret == 0)
    {
        return true;
    }
    switch (errno)
    {
    case 0:
    case EEXIST:
        return true;
    case ENOENT: //no parent directory exists
        if (createDirectories(dirname(path)))
        {
            return _wmkdir(path.c_str()) == 0;
        }
    default:
        return false;
    }
#else
 // Creating a directory 
    if (mkdir(path.c_str(), 0744) == -1)
        return false;
    else
        return true;

#endif
    return false;
}

GAHELPER_NS_END
