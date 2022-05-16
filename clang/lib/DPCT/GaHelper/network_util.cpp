//===--- network_util.cpp-------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
#include "network_util.h"

#ifdef _WIN32
#  include <ws2tcpip.h>
#  define HOST_NOT_FOUND_ERROR WSAHOST_NOT_FOUND
#  define HOST_NO_DATA_ERROR WSANO_DATA
#else
#  include <sys/types.h>
#  include <sys/socket.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  define HOST_NOT_FOUND_ERROR EAI_NONAME
#  ifndef __FreeBSD__
#    define HOST_NO_DATA_ERROR EAI_NODATA
#  else
#    define HOST_NO_DATA_ERROR EAI_NONAME //EAI_NODATA is deprecated on FREEBSDs
#  endif
#endif

GAHELPER_NS_BEGIN

network_status_t getNetworkStatus()
{
#ifdef _WIN32
    WSADATA wsa_data;
    WORD  version_requested = MAKEWORD(1, 1);
    if (WSAStartup(version_requested, &wsa_data))
    {
        return network_status_t::network_error;
    }
#endif
    network_status_t ret = network_status_t::network_error;
    struct addrinfo *info = nullptr;
    struct addrinfo hints = { 0 };

    struct sockaddr_in  *sockaddr_ipv4;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_protocol = IPPROTO_UDP;

    auto retval = getaddrinfo("corp.intel.com", "domain", &hints, &info);
    if (HOST_NOT_FOUND_ERROR == retval || HOST_NO_DATA_ERROR == retval)
    {
        ret = network_status_t::outside_intel_network;
    }
    else if (retval)
    {
        ret = network_status_t::network_error;
    }
    else
    {
        ret = network_status_t::outside_intel_network;
        for (auto ptr = info; ptr; ptr = ptr->ai_next)
        {
            if (ptr->ai_family != AF_INET)
            {
                continue;
            }
            sockaddr_ipv4 = (struct sockaddr_in *) ptr->ai_addr;
            if ((sockaddr_ipv4->sin_addr.s_addr & 0xff) == 10) //IP address is in not globally routed because is 10.x.x.x
            {
                ret = network_status_t::inside_intel_network;
                break;
            }
        }
    }
    freeaddrinfo(info);
#ifdef _WIN32
    WSACleanup();
#endif
    return ret;
}

GAHELPER_NS_END
