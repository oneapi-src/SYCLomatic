//===--- uuid.cpp-------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
#ifdef _WIN32
#  define _CRT_RAND_S 
#  include <stdlib.h>
#else
#include <fcntl.h>
#  include <iostream>
#  include <fstream>
#endif

#include <sstream>
#include <iomanip>
#include "uuid.h"
const size_t UUID_SIZE = 16;

GAHELPER_NS_BEGIN

const int uuid_groups[] = { 4, 2, 2, 2, 6 };

std::string uuid_bytes_to_str(void* bytes)
{
    const unsigned char* ptr = static_cast<const unsigned char*>(bytes);
    auto end_ptr = ptr + UUID_SIZE;
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::uppercase;
    for (auto x : uuid_groups)
    {
        for (int i = 0; i < x; i++)
        {
            ss << std::setw(2) << static_cast<unsigned>(*ptr);
            ptr++;
        }
        if (ptr < end_ptr)
        {
            ss << "-";
        }
    }
    return ss.str();
}

std::string generate_uuid4()
{
#ifdef _WIN32
    unsigned uuid[UUID_SIZE / sizeof(unsigned)];
    static_assert(UUID_SIZE % sizeof(unsigned) == 0, "Illegal UUID_SIZE");
    for (auto i = 0; i < sizeof(uuid) / sizeof(uuid[0]); i++)
        rand_s(uuid + i);
    return uuid_bytes_to_str(uuid);
#else
    unsigned char uuid[UUID_SIZE] = {0};
    std::ifstream urandom;
    urandom.open("/dev/urandom", std::ios::binary | std::ios::in);
    if (urandom)
    {
        urandom.read(reinterpret_cast<char*>(uuid), sizeof(uuid));
        if (urandom)
        {
            return uuid_bytes_to_str(uuid);
        }
    }
    return "";
#endif

}

GAHELPER_NS_END
