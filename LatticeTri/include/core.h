/*********************************************************************
 * @file   core.h
 * @brief  Basic data structure
 *
 * @author redblacksoup
 * @date   May 2023
 *********************************************************************/
#pragma once
#include "pch.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

typedef Eigen::Vector3f vec3f;
typedef Eigen::Vector3d vec3d;
typedef Eigen::Matrix3f mat;

inline double3 cross(const double3& a, const double3& b)
{
	return make_double3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline double dot(const double3& a, const double3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline double norm(const double3& a)
{
	return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

inline void normalize(double3& a)
{
	double n = norm(a);
	a.x /= n;
	a.y /= n;
	a.z /= n;
}

inline float3 cross(const float3& a, const float3& b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline float dot(const float3& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float norm(const float3& a)
{
	return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

inline void normalize(float3& a)
{
	float n = norm(a);
	a.x /= n;
	a.y /= n;
	a.z /= n;
}

struct vertex
{
	float props;
	vec3f p;
	vertex(float _x, float _y, float _z, float props_ = 0.0)
	{
		this->p[0] = _x;
		this->p[1] = _y;
		this->p[2] = _z;
		this->props = props_;
	};
	vertex(vec3f v, float props_ = 0.0)
	{
		this->p[0] = v[0];
		this->p[1] = v[1];
		this->p[2] = v[2];
		this->props = props_;
	}
	float& operator[](int i)
	{
		switch (i)
		{
		case 0:
			return this->p[0];
		case 1:
			return this->p[1];
		case 2:
			return this->p[2];
		default:
			throw std::out_of_range("Vertex subscript out of range!");
		}
	};
};

/**
 * Face struct.
 */
struct face
{
	float s[13];

	face(const vec3f& p1_, const vec3f& p2_, const vec3f& p3_)
	{
		s[0] = 0.f;
		s[1] = 0.f;
		s[2] = 0.f;
		s[3] = p1_[0];
		s[4] = p1_[1];
		s[5] = p1_[2];
		s[6] = p2_[0];
		s[7] = p2_[1];
		s[8] = p2_[2];
		s[9] = p3_[0];
		s[10] = p3_[1];
		s[11] = p3_[2];
		s[12] = 0.f;
	}
};

struct beamParcel
{
	uint32_t st, ed, stNeighs, edNeighs;
	uint32_t* stLst = nullptr, * edLst = nullptr;

	~beamParcel()
	{
		if (stLst != nullptr)
		{
			delete[] stLst;
		}
		if (edLst != nullptr)
		{
			delete[] edLst;
		}
	}
};