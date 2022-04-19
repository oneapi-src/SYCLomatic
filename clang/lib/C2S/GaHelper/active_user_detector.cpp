//===--- active_user_detector.cpp-------------------------*- C++ -*---===//
//
// Copyright (C) Intel Corporation. All rights reserved.
//
// The information and source code contained herein is the exclusive
// property of Intel Corporation and may not be disclosed, examined
// or reproduced in whole or in part without explicit written authorization
// from the company.
//
//===-----------------------------------------------------------------===//
//todo
#include "active_user_detector.h"
//#include "filelock.h"

#ifdef _WIN32
#    include <windows.h>
#else
#    include <time.h>
#endif
#include <iostream>
GAHELPER_NS_BEGIN

ActiveUserDetector::ActiveUserDetector(IActiveUserPersistenceProvider* activeUserPersistenceProvider)
{
    m_should_save = false;
    m_activeUserPersistenceProvider = activeUserPersistenceProvider;
    load();
}

ActiveUserDetector::~ActiveUserDetector()
{
    if (m_should_save)
    {
        save();
    }
}

void ActiveUserDetector::notify_activity()
{
    date_t date;
    get_current_date(&date);

    if (m_last_active_dates.size())
    {
        auto& previous_date = m_last_active_dates.back();
        if (previous_date != date)
        {
            m_last_active_dates.push_back(date);
            m_should_save = true;
        }
        while (m_last_active_dates.size() > DAY_PER_YEAR_MAKES_ACTIVE_USER)
        {
            m_last_active_dates.erase(m_last_active_dates.begin());
        }
    }
    else
    {
        m_should_save = true;
        m_last_active_dates.push_back(date);
    }
}

bool ActiveUserDetector::is_active_user()
{
    if (m_last_active_dates.size() < DAY_PER_YEAR_MAKES_ACTIVE_USER)
    {
        return false;
    }
    date_t d1;
    get_current_date(&d1);
    //Year ago, date might be not accurate for example for 29 february in leap year, but this is 
    //sufficient for our purposes
    d1.year--;
    date_t& d2 = m_last_active_dates.front();
    return d2 > d1;
}

void ActiveUserDetector::get_current_date(ActiveUserDetector::date_t* result)
{
#ifdef _WIN32
    //Need to use WinAPI since there is no thread safe gmtime() function.
    SYSTEMTIME st;
    GetSystemTime(&st);

    result->year = st.wYear;
    result->mon = st.wMonth;
    result->day = st.wDay;
#else
    auto t = time(0);
    struct tm date_time;
    localtime_r(&t, &date_time);
    result->year = date_time.tm_year + 1900;
    result->mon = date_time.tm_mon + 1;
    result->day = date_time.tm_mday;

#endif
}

bool ActiveUserDetector::save()
{
    if (m_last_active_dates.empty() || m_activeUserPersistenceProvider == nullptr)
    {
        return false;
    }
    WriterCallback callback(m_last_active_dates);
    return m_activeUserPersistenceProvider->write(&callback);
}

bool ActiveUserDetector::load()
{
    if (m_activeUserPersistenceProvider == nullptr)
    {
        return false;
    }
    m_last_active_dates.clear();
    ReaderCallback callback(m_last_active_dates);
    return m_activeUserPersistenceProvider->read(&callback);
}

bool ActiveUserDetector::WriterCallback::getItem(int* outYear, int* outMon, int* outDay)
{
    if (m_iter == m_data.end())
    {
        return false;
    }
    *outYear = m_iter->year;
    *outMon = m_iter->mon;
    *outDay = m_iter->day;
    ++m_iter;
    return true;
}

bool ActiveUserDetector::ReaderCallback::onItem(int year, int mon, int day)
{
    date_t d;
    d.year = year;
    d.mon = mon;
    d.day = day;

    m_data.push_back(d);
    return true;
}
void ActiveUserDetector::dumpStat(){
    std::cout <<"ActiveUserDetector status:\n";
    std::cout << "\tm_should_save: " << m_should_save << "\n";
    std::cout << "\tis_active_user: " << is_active_user() << "\n";
    std::cout << "\tcontent of m_last_active_dates: \n";
    for (auto i : m_last_active_dates) {
        std::cout << "\t\t" << i.year << i.mon << i.day << "\n";
    }
}


GAHELPER_NS_END
