//===--- active_user_detector.h-------------------------*- C++ -*---===//
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
#include <vector>

GAHELPER_NS_BEGIN

//Detects if current user is active user in marketing terms:
//4 days or more use per year (DAY_PER_YEAR_MAKES_ACTIVE_USER const)
//cannot be implemented on GA reporting side.

static const int DAY_PER_YEAR_MAKES_ACTIVE_USER = 4;
class ActiveUserDetector
{
public:
    ActiveUserDetector(IActiveUserPersistenceProvider* activeUserPersistenceProvider);
    ~ActiveUserDetector();
    void notify_activity();
    bool is_active_user();
	
	void dumpStat();
private:
    struct date_t
    {
        int year;
        int mon;
        int day;
        bool operator ==(const date_t& other) { return other.year == year && other.mon == mon && other.day == day; }
        bool operator !=(const date_t& other) { return !(*this == other); }
        bool operator > (const date_t& other)
        {
            //TODO think maight be better compare year<<9 + month << 5 + day
            if (year > other.year)
            {
                return true;
            }
            else if (year < other.year)
            {
                return false;
            }
            else if (mon > other.mon)
            {
                return true;
            }
            else if (mon < other.mon)
            {
                return false;
            }
            else if (day > other.day)
            {
                return true;
            }
            else if (day < other.day)
            {
                return false;
            }
            return false;
        }
    };

    typedef std::vector<date_t> DateContainer;

    class WriterCallback : public IActiveUserPersistenceProvider::WriterCallback
    {
    public:
        WriterCallback(const DateContainer& data) : m_data(data), m_iter(m_data.begin()) {}
        virtual bool getItem(int* outYear, int* outMon, int* outDay) override;
    private:
        const DateContainer& m_data;
        DateContainer::const_iterator m_iter;
    };

    class ReaderCallback : public IActiveUserPersistenceProvider::ReaderCallback
    {
    public:
        ReaderCallback(DateContainer& data) : m_data(data) {}
        virtual bool onItem(int year, int mon, int day) override;
    private:
        DateContainer& m_data;
    };

    void get_current_date(date_t* result);
    bool save();
    bool load();
    DateContainer m_last_active_dates;
    IActiveUserPersistenceProvider* m_activeUserPersistenceProvider;
    bool m_should_save;
};

GAHELPER_NS_END

