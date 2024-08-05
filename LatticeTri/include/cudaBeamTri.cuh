#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "pch.h"
#include "core.h"
#include "cuda_fp16.h"

const int ThreadsPerBlock = 128;
const int ThreadsPerGroup = 32;
const int GroupsPerBlock = 4;

inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

__host__ __device__ inline uint64_t ummin(uint32_t a, uint32_t b)
{
	return a > b ? uint64_t(b) : uint64_t(a);
}

struct edge
{
	float st, len;
	bool valid, hole;
	float3 basis_a;
	float3 basis_b;

	__host__ __device__ inline bool get_valid()
	{
		return valid;
	}
	__host__ __device__ inline void set_valid(bool res)
	{
		valid = res;
	}

	__host__ __device__ inline bool get_hole()
	{
		return hole;
	}
	__host__ __device__ inline void set_hole(bool res)
	{
		hole = res;
	}

	__host__ __device__ inline float3 getA()
	{
		return basis_a;
	}
	__host__ __device__ inline float3 getB()
	{
		return basis_b;
	}
	__host__ __device__ inline float get_start()
	{
		return st;
	}
	__host__ __device__ inline float get_length()
	{
		return len;
	}
};


__host__ __device__ inline float3 getAxis(const uint64_t& data, const float& len)
{
	uint32_t bin_phi = uint32_t(data);
	bin_phi &= 0b11111111111111111111;
	float phi = (float)bin_phi * F_PI / 0b11111111111111111111;
	uint32_t bin_theta = uint32_t(data >> 20);
	bin_theta &= 0b111111111111111111111;
	float theta = (float)bin_theta * F_PI / 1048576.f - F_PI;
	return make_float3(len * sin(phi) * cos(theta), len * sin(phi) * sin(theta), len * cos(phi));
}

struct edge_soa
{
	float* a;
	float* b;
	uint64_t* data_a;
	uint64_t* data_b;

	inline void deviceMemAlloc(int size)
	{
		checkCuda(cudaMalloc((void**)&a, size * sizeof(float)));
		checkCuda(cudaMalloc((void**)&b, size * sizeof(float)));
		checkCuda(cudaMalloc((void**)&data_a, size * sizeof(uint64_t)));
		checkCuda(cudaMalloc((void**)&data_b, size * sizeof(uint64_t)));
	}
	
	inline void hostMemAlloc(int size)
	{
		this->a = new float[size];
		this->b = new float[size];
		this->data_a = new uint64_t[size];
		this->data_b = new uint64_t[size];
	}

	inline void deviceMemFree()
	{
		cudaFree(a);
		cudaFree(b);
		cudaFree(data_a);
		cudaFree(data_b);
	}

	inline void hostMemFree()
	{
		delete[] this->a;
		delete[] this->b;
		delete[] this->data_a;
		delete[] this->data_b;
	}
};

struct edge_cuda
{
	float a, b;
	uint64_t data_a, data_b;
	__host__ __device__ inline float get_phi_a()
	{
		uint32_t bin = uint32_t(data_a);
		bin &= 0b11111111111111111111;
		return (float)bin * F_PI / 0b11111111111111111111;
	}

	__host__ __device__ inline float get_phi_b()
	{
		uint32_t bin = uint32_t(data_b);
		bin &= 0b11111111111111111111;
		return (float)bin * F_PI / 0b11111111111111111111;
	}

	__host__ __device__ inline float get_theta_a()
	{
		uint32_t bin = uint32_t(data_a >> 20);
		bin &= 0b111111111111111111111;
		return (float)bin * F_PI / 1048576.f - F_PI;
	}

	__host__ __device__ inline float get_theta_b()
	{
		uint32_t bin = uint32_t(data_b >> 20);
		bin &= 0b111111111111111111111;
		return (float)bin * F_PI / 1048576.f - F_PI;
	}

	__host__ __device__ inline void set_valid(bool res)
	{
		if (res)
		{
			data_a |= (0x8000000000000000);
		}
		else
		{
			data_a &= ~(0x8000000000000000);
		}
	}

	__host__ __device__ inline void set_hole(bool res)
	{
		if (res)
		{
			data_b |= (0x8000000000000000);
		}
		else
		{
			data_b &= ~(0x8000000000000000);
		}
	}

	__host__ __device__ inline bool get_valid()
	{
		uint64_t tmp = 0x8000000000000000;
		if ((data_a & tmp) != 0)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	__host__ __device__ inline bool get_hole()
	{
		uint64_t tmp = 0x8000000000000000;
		if ((data_b & tmp) != 0)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	__host__ __device__ inline void setA(float phi, float theta)
	{
		uint32_t bin_phi = (uint32_t)(phi / F_PI * (0b11111111111111111111));
		uint32_t bin_theta = (uint32_t)((theta + F_PI) / F_PI * (0b11111111111111111111 + 1));

		uint64_t tmp;
		tmp = ummin(0b11111111111111111111, bin_phi);
		tmp &= 0b11111111111111111111;
		data_a |= tmp;

		tmp = bin_theta;
		tmp &= 0b111111111111111111111;
		data_a |= (tmp << 20);
	}

	__host__ __device__ inline void setB(float phi, float theta)
	{
		uint32_t bin_phi = (uint32_t)(phi / F_PI * (0b11111111111111111111));
		uint32_t bin_theta = (uint32_t)((theta + F_PI) / F_PI * (0b11111111111111111111 + 1));

		uint64_t tmp;
		tmp = ummin(0b11111111111111111111, bin_phi);
		tmp &= 0b11111111111111111111;
		data_b |= tmp;

		tmp = bin_theta;
		tmp &= 0b111111111111111111111;
		data_b |= (tmp << 20);
	}

	__host__ __device__ inline float3 getA()
	{
		float phi = get_phi_a();
		float theta = get_theta_a();
		return make_float3(a * sin(phi) * cos(theta), a * sin(phi) * sin(theta), a * cos(phi));
	}

	__host__ __device__ inline float3 getB()
	{
		float phi = get_phi_b();
		float theta = get_theta_b();
		return make_float3(b * sin(phi) * cos(theta), b * sin(phi) * sin(theta), b * cos(phi));
	}

	__device__ inline void set_start(float st)
	{
		uint32_t bin_st = (uint32_t)((st + F_PI) / F_PI * 2097152.f);
		uint64_t tmp;
		tmp = bin_st & 0b1111111111111111111111;
		data_a |= (tmp << 41);
	}

	__device__ inline void set_length(float len)
	{
		uint32_t bin_len = (uint32_t)(len / (2 * F_PI) * (0b1111111111111111111111));
		uint64_t tmp;
		tmp = ummin(0b1111111111111111111111, bin_len);
		data_b |= (tmp << 41);
	}

	__host__ __device__ inline float get_start()
	{
		uint32_t bin = uint32_t(data_a >> 41);
		bin &= 0b1111111111111111111111;
		return (float)bin * F_PI / 2097152.f - F_PI;
	}

	__host__ __device__ inline float get_length()
	{
		uint32_t bin = uint32_t(data_b >> 41);
		bin &= 0b1111111111111111111111;
		return (float)(bin) * 2 * F_PI / 0b1111111111111111111111;
	}
};

void batchStream_cuda(int batchBegin, int batchSize, float radius, edge_soa& loop, std::vector<int2>& edgesLst, std::vector<int>& prefixSumArr);

void transferDataToDevice(std::vector<float3>& nodesLst, std::vector<int2>& edgesLst, std::vector<int>& prefixSumArr, std::vector<int>& adjMat);

void releaseDevice();

