#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <CL/cl.hpp>
#include "Utils.h"
#include <chrono>

#include "Kernel.h"

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::high_resolution_clock::time_point Timer;

void print_help()
{
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

/// Data Record
//Each column corresponds to the following category :
//1. weather station name
//2. year the record was collected
//3. month
//4. day
//5 time(HHMM)
//6 air temperature(degree Celsius)

/// Holds the statistical data for each station.
struct Data 
{
	float min, max, sum, avg, median, variance, stdv, LQT, HQT;
	unsigned short int readTime, sortTime, minTime, maxTime, sumTime, stdvTime;
};

///Reads the data and loads it into vectors.
std::vector<float> ReadFile(const char* path);
Data SortData(int platform_id, int device_id, std::vector<float> temperatures);
std::vector<float> SortTest(int count, Kernel sortKernel, std::vector<float> temperatures);
std::vector<float> Sort(Kernel sortKernel, std::vector<float> temperatures);