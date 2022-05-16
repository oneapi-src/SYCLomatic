//===--- http_connector.cpp-------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
//TODO:  remove the macro to enable windows support.
#include "http_connector.h"
#include "os_specific.h"
#include <iostream>

namespace
{
    size_t empty_wite_callback(char *ptr, size_t size, size_t nmemb, void *userdata)
    {
        return size * nmemb;
    }
}


GAHELPER_NS_BEGIN

HttpConnector::HttpConnector(const char* server_url) :
    m_serverUrl(server_url),
    m_currentProxyIndex(0),
    m_enabled(true)
{
    curl_global_init(CURL_GLOBAL_ALL);
    m_curl = curl_easy_init();
    curl_easy_setopt(m_curl, CURLOPT_URL, m_serverUrl.c_str());
    curl_easy_setopt(m_curl, CURLOPT_WRITEFUNCTION, empty_wite_callback);
    curl_easy_setopt(m_curl, CURLOPT_TIMEOUT, HTTP_TIMEOUT);
    // background thread can start only after all preparations are done.
    m_backgound_thread.reset(new std::thread(&HttpConnector::backgroundThread, this));
}

HttpConnector::~HttpConnector()
{
    m_queue.push(""); // worker thread treat emty string as command to terminate
    m_backgound_thread->join();
    curl_easy_cleanup(m_curl);
    curl_global_cleanup();
}

void HttpConnector::post(std::string && post_fields_urlencoded)
{
    m_queue.push(std::move(post_fields_urlencoded));
}

std::string HttpConnector::urlencode(const std::string& input)
{
    char* buff = curl_easy_escape(m_curl, input.c_str(), 0);
    std::string ret(buff);
    curl_free(buff);
    return ret;
}

void HttpConnector::perform_post(std::string item)
{
    if (m_curl == nullptr || m_enabled == false)
    {
        return;
    }

    CURLcode res;
    curl_easy_setopt(m_curl, CURLOPT_POSTFIELDS, item.c_str());

    for (size_t i = 0; i < m_proxies.size(); i++)
    {
        auto& proxy = m_proxies[m_currentProxyIndex];
        auto proxyCstr = proxy.size() ? proxy.c_str() : nullptr; // for treating empty proxy as no proxy (curl requires NULL)
        curl_easy_setopt(m_curl, CURLOPT_PROXY, proxyCstr);

        res = curl_easy_perform(m_curl);
        if (res == CURLE_OK)
        {
            long httpCode;
            res = curl_easy_getinfo(m_curl, CURLINFO_RESPONSE_CODE, &httpCode);
            if (httpCode / 100 == 2) // http code 2xx
            {
                return;
            }
        }
        m_currentProxyIndex = (m_currentProxyIndex + 1) % m_proxies.size();
    }
    //we've tried all known connections methods, but does not succeed; giving up
    m_enabled = false;
}

void HttpConnector::backgroundThread()
{
    collectProxyInfo(m_serverUrl.c_str(), &m_proxies);
    m_proxies.push_back(""); // empty string mean no_proxy
    while (true)
    {
        auto x = m_queue.pop();
        if (x.empty())
        {
            break;
        }
        perform_post(x);
        //TODO: add calculated sleep in order to comply with rate limits. (Not needed while we are not sending too much)
    }
}
void HttpConnector::dumpStats(){
    std::cout << "HttpConnector status: \n";
    std::cout << "\tm_enabled: " << m_enabled << "\n";
    std::cout << "\tm_currentProxyIndex: " << m_currentProxyIndex << "\n";
    std::cout << "\tm_serverUrl: " << m_serverUrl << "\n";
    std::cout << "\tm_proxies: " << m_proxies[m_currentProxyIndex] << "\n";
    std::cout << "\n";
        

}

GAHELPER_NS_END
