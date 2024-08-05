/*********************************************************************
 * @file   pch.h
 * @brief  Precompiled headers.
 *
 * @author redblacksoup
 * @date   May 2023
 *********************************************************************/
#pragma once

#ifdef _WIN64
#define _CRT_SECURE_NO_WARNINGS
#endif // _WIN64

#define _USE_MATH_DEFINES
#define F_PI 3.14159265358979323846f

#include <cmath>
#include <vector>
#include <array>
#include <list>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <future>
#include <chrono>
#include <stdio.h>
#include <cstring>
#include <queue>
#include <functional>
#include <filesystem>
#ifdef _HAS_CXX20
#include <format>
#include <string_view>
#endif // _HAS_CXX20
#include <stdlib.h>
#include <stdio.h>

// oneTBB library
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/concurrent_vector.h>
#include <oneapi/tbb/tick_count.h>
#include <oneapi/tbb/parallel_for_each.h>
#include <oneapi/tbb/parallel_pipeline.h>

#define TICK(x) auto bench_##x = tbb::tick_count::now();
#define TOCK(x) std::cout << #x ": " << (tbb::tick_count::now() - bench_##x).seconds() << "s" << std::endl;
#define setbit(x,y) x|=(uint64_t(1)<<y) 
#define clrbit(x,y) x&=~(uint64_t(1)<<y) 
