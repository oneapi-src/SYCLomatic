//===--- http_connector.h-------------------------*- C++ -*---===//
//
// Copyright (C) 2018-1019 Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
#pragma once
#include "GaNamespace.h"
#include <curl/curl.h>
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

//TODO:  remove the macro to enable windows support.
#if defined(__linux__)

GAHELPER_NS_BEGIN
const int HTTP_TIMEOUT = 5;

template <typename T>class ConcurrentQueue
{
public:
    void push(const T& item)
    {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_queue.push(item);
        }
        m_cond.notify_one();
    }
    T pop()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        while (m_queue.empty())
        {
            m_cond.wait(lock);
        }
        T ret = m_queue.front();
        m_queue.pop();
        return ret;
    }
private:
    std::mutex m_mutex;
    std::condition_variable m_cond;
    std::queue<T> m_queue;
};

class HttpConnector
{
public:
    HttpConnector(const char* server_url);
    ~HttpConnector();
    void post(std::string && post_fields_urlencoded);
    std::string urlencode(const std::string& input);
	void dumpStats();
private:
    void backgroundThread();
    void perform_post(std::string item);
private:
    ConcurrentQueue<std::string> m_queue;
    std::unique_ptr<std::thread> m_backgound_thread;
    CURL* m_curl;
    std::string m_serverUrl;
    std::vector<std::string> m_proxies;
    int m_currentProxyIndex;
    bool m_enabled;
};

GAHELPER_NS_END
#endif
