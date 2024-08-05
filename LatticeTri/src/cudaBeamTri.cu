#include "cudaBeamTri.cuh"
// Primary header is compatible with pre-C++11, collective algorithm headers require C++11
#include <cooperative_groups.h>
// Optionally include for memcpy_async() collective
#include <cooperative_groups/memcpy_async.h>
// Optionally include for reduce() collective
#include <cooperative_groups/reduce.h>
// Optionally include for inclusive_scan() and exclusive_scan() collectives
float3* dev_nodesLst;
int2* dev_edgesLst;
int* dev_prefixSumArr;
int* dev_adjMat;
// device symbol
__device__ float3* ds_nodesLst;
__device__ int2* ds_edgesLst;
__device__ int* ds_prefixSumArr;
__device__ int* ds_adjMat;

using namespace cooperative_groups;
__device__ bool intersection(float st, float ed, float st_, float ed_, float& r1, float& r2)
{
	r1 = fmax(st, st_);
	r2 = fmin(ed, ed_);
	return r2 > r1 + 1e-3f;
}

__device__ float3 dev_cross(float3& a, float3& b)
{
	return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ float dev_dot(const float3& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float dev_norm(const float3& a)
{
	return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

__device__ void dev_normalize(float3& a)
{
	float invLen = rsqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
	a.x *= invLen;
	a.y *= invLen;
	a.z *= invLen;
	return;
}

__device__ void compressData(float3& d0, float3& d1, bool valid, bool hole, float& a, float& b, float& eSt, float& eEd, edge_soa& eLoop, int off)
{
	float phi_a, phi_b, theta_a, theta_b;
	theta_a = atan2(d0.y, d0.x);
	phi_a = atan2(sqrt(d0.x * d0.x + d0.y * d0.y), d0.z);
	theta_b = atan2(d1.y, d1.x);
	phi_b = atan2(sqrt(d1.x * d1.x + d1.y * d1.y), d1.z);

	uint64_t data_sectorA = valid;
	uint64_t data_sectorB = hole;

	{
		data_sectorA <<= 22;
		uint32_t bin_st = (uint32_t)((eSt + F_PI) / F_PI * 2097152.f);
		data_sectorA |= (bin_st & 0x3fffffu);
		data_sectorA <<= 21;
		// edgeLoop[tx + offset[eIdx]].setA(phi_a, theta_a);
		uint32_t bin_phi = (uint32_t)(phi_a / F_PI * (0b11111111111111111111));
		uint32_t bin_theta = (uint32_t)((theta_a + F_PI) / F_PI * (0b11111111111111111111 + 1));

		data_sectorA |= (bin_theta & 0x1fffffu);
		data_sectorA <<= 20;
		data_sectorA |= ummin(0xfffffu, bin_phi);
	}

	{
		data_sectorB <<= 22;
		uint32_t bin_len = (uint32_t)(eEd / (2 * F_PI) * (0b1111111111111111111111));
		data_sectorB |= ummin(0b1111111111111111111111, bin_len);
		data_sectorB <<= 21;

		// edgeLoop[tx + offset[eIdx]].setB(phi_b, theta_b);
		uint32_t bin_phi = (uint32_t)(phi_b / F_PI * (0b11111111111111111111));
		uint32_t bin_theta = (uint32_t)((theta_b + F_PI) / F_PI * (0b11111111111111111111 + 1));

		data_sectorB |= (bin_theta & 0x1fffffu);
		data_sectorB <<= 20;

		data_sectorB |= ummin(0xfffffu, bin_phi);
	}

	eLoop.data_a[off] = data_sectorA;
	eLoop.data_b[off] = data_sectorB;
	eLoop.a[off] = a;
	eLoop.b[off] = b;
	return;
}

__device__ void halfWarpTask(int nodeX, int nodeY, int sz1, int sz2, int eLoopOff, float radius, edge_soa& eLoop)
{
	int tx = threadIdx.x & 0x0f;
	int wid = threadIdx.x / 16;

	int stIdx, edIdx, size, off;
	if (wid == 0)
	{
		stIdx = nodeX;
		edIdx = nodeY;
		size = sz1;
		off = eLoopOff;
	}
	else if (wid == 1)
	{
		stIdx = nodeY;
		edIdx = nodeX;
		size = sz2;
		off = eLoopOff + sz1;
	}

	unsigned mask = __ballot_sync(0xffffffff, tx < size);
	if (tx >= size)	return;

	bool valid = true;
	bool hole = false;

	// initialize edge_cuda
	float3 d[3];
	float a;
	float3 dik, dij, delta;

	int nIdx = ds_adjMat[ds_prefixSumArr[stIdx] + tx];
	// neighbors[0] - node
	dij = make_float3(ds_nodesLst[edIdx].x - ds_nodesLst[stIdx].x, ds_nodesLst[edIdx].y - ds_nodesLst[stIdx].y, ds_nodesLst[edIdx].z - ds_nodesLst[stIdx].z);
	dev_normalize(dij);
	// neighbos[x] - node
	dik = make_float3(ds_nodesLst[nIdx].x - ds_nodesLst[stIdx].x, ds_nodesLst[nIdx].y - ds_nodesLst[stIdx].y, ds_nodesLst[nIdx].z - ds_nodesLst[stIdx].z);
	dev_normalize(dik);
	delta = dev_cross(dik, dij);
	if (dev_norm(delta) < 1e-3f)
	{
		d[2] = make_float3(-dij.x, -dij.y, -dij.z);
		d[0] = make_float3(copysign(d[2].z, d[2].x), copysign(d[2].z, d[2].y), -copysign(d[2].x, d[2].z) - copysign(d[2].y, d[2].z));
		dev_normalize(d[0]);
		d[1] = dev_cross(d[2], d[0]);
		dev_normalize(d[1]);
		a = radius;
		if (nIdx != edIdx)
		{
			valid = false;
		}
		else
		{
			hole = true;
		}
	}
	else
	{
		d[1] = make_float3(delta.x, delta.y, delta.z);
		dev_normalize(d[1]);
		d[0] = make_float3(dij.x + dik.x, dij.y + dik.y, dij.z + dik.z);
		dev_normalize(d[0]);
		d[2] = dev_cross(d[0], d[1]);
		dev_normalize(d[2]);
		float theta = acos(dev_dot(dik, dij));
		a = radius / cos(F_PI / 2 - theta / 2);
	}

	if (!valid)
	{
		d[2] = make_float3(0.f, 0.f, 0.f);
	}

	float eSt = -2 * F_PI, eEd = 2 * F_PI;
	float3 ai = make_float3(d[0].x * a, d[0].y * a, d[0].z * a);
	float3 bi = make_float3(d[1].x * radius, d[1].y * radius, d[1].z * radius);

	for (int j = 0; j < size; j++)
	{
		float3 n;
		n.x = __shfl_sync(mask, d[2].x, j, 16);
		n.y = __shfl_sync(mask, d[2].y, j, 16);
		n.z = __shfl_sync(mask, d[2].z, j, 16);
		if (j == tx || dev_norm(n) < 1e-3f || !valid)
		{
			continue;
		}
		// Ellipse intersect with plane
		float na, nb, st;
		// n = cross(edgeLoop[j].basis[0], edgeLoop[j].basis[1]).normalized();
		// na = n @ a, nb = n @ b
		na = ai.x * n.x + ai.y * n.y + ai.z * n.z;
		nb = bi.x * n.x + bi.y * n.y + bi.z * n.z;

		st = atan2(na, -nb);

		// stEdgeLoop[i] substract [st, ed]
		if (st + F_PI < eSt)
		{
			valid = intersection(eSt - 2 * F_PI, eEd - 2 * F_PI, st, st + F_PI, eSt, eEd);
		}
		else if (eEd < st)
		{
			valid = intersection(eSt, eEd, st - 2 * F_PI, st - F_PI, eSt, eEd);
		}
		else
		{
			valid = intersection(eSt, eEd, st, st + F_PI, eSt, eEd);
		}
	}
	eEd = fmin(eEd - eSt, float(2 * F_PI));

	compressData(d[0], d[1], valid, hole, a, radius, eSt, eEd, eLoop, tx + off);
	return;
}

__device__ void fulWarpTask(int stIdx, int edIdx, int size, int off, float radius, edge_soa& eLoop)
{
	auto& tx = threadIdx.x;

	unsigned mask = __ballot_sync(0xffffffff, tx < size);
	if (tx >= size) return;

	bool valid = true;
	bool hole = false;

	// initialize edge_cuda
	float3 d[3];
	float a;
	float3 dik, dij, delta;

	int nIdx = ds_adjMat[ds_prefixSumArr[stIdx] + tx];
	// neighbors[0] - node
	dij = make_float3(ds_nodesLst[edIdx].x - ds_nodesLst[stIdx].x, ds_nodesLst[edIdx].y - ds_nodesLst[stIdx].y, ds_nodesLst[edIdx].z - ds_nodesLst[stIdx].z);
	dev_normalize(dij);
	// neighbos[x] - node
	dik = make_float3(ds_nodesLst[nIdx].x - ds_nodesLst[stIdx].x, ds_nodesLst[nIdx].y - ds_nodesLst[stIdx].y, ds_nodesLst[nIdx].z - ds_nodesLst[stIdx].z);
	dev_normalize(dik);
	delta = dev_cross(dik, dij);
	if (dev_norm(delta) < 1e-3f)
	{
		d[2] = make_float3(-dij.x, -dij.y, -dij.z);
		d[0] = make_float3(copysign(d[2].z, d[2].x), copysign(d[2].z, d[2].y), -copysign(d[2].x, d[2].z) - copysign(d[2].y, d[2].z));
		dev_normalize(d[0]);
		d[1] = dev_cross(d[2], d[0]);
		dev_normalize(d[1]);
		a = radius;
		if (nIdx != edIdx)
		{
			valid = false;
		}
		else
		{
			hole = true;
		}
	}
	else
	{
		d[1] = make_float3(delta.x, delta.y, delta.z);
		dev_normalize(d[1]);
		d[0] = make_float3(dij.x + dik.x, dij.y + dik.y, dij.z + dik.z);
		dev_normalize(d[0]);
		d[2] = dev_cross(d[0], d[1]);
		dev_normalize(d[2]);
		float theta = acos(dev_dot(dik, dij));
		a = radius / cos(F_PI / 2 - theta / 2);
	}

	if (!valid)
	{
		d[2] = make_float3(0.f, 0.f, 0.f);
	}

	float eSt = -2 * F_PI, eEd = 2 * F_PI;
	float3 ai = make_float3(d[0].x * a, d[0].y * a, d[0].z * a);
	float3 bi = make_float3(d[1].x * radius, d[1].y * radius, d[1].z * radius);

	for (int j = 0; j < size; j++)
	{
		float3 n;
		n.x = __shfl_sync(mask, d[2].x, j, 32);
		n.y = __shfl_sync(mask, d[2].y, j, 32);
		n.z = __shfl_sync(mask, d[2].z, j, 32);
		if (j == tx || dev_norm(n) < 1e-3f || !valid)
		{
			continue;
		}
		// Ellipse intersect with plane
		float na, nb, st;
		// n = cross(edgeLoop[j].basis[0], edgeLoop[j].basis[1]).normalized();
		// na = n @ a, nb = n @ b
		na = ai.x * n.x + ai.y * n.y + ai.z * n.z;
		nb = bi.x * n.x + bi.y * n.y + bi.z * n.z;

		st = atan2(na, -nb);

		// stEdgeLoop[i] substract [st, ed]
		if (st + F_PI < eSt)
		{
			valid = intersection(eSt - 2 * F_PI, eEd - 2 * F_PI, st, st + F_PI, eSt, eEd);
		}
		else if (eEd < st)
		{
			valid = intersection(eSt, eEd, st - 2 * F_PI, st - F_PI, eSt, eEd);
		}
		else
		{
			valid = intersection(eSt, eEd, st, st + F_PI, eSt, eEd);
		}
	}
	eEd = fmin(eEd - eSt, float(2 * F_PI));
	compressData(d[0], d[1], valid, hole, a, radius, eSt, eEd, eLoop, tx + off);
	return;
}

__global__ void hugeTask(int stIdx, int edIdx, edge_soa eLoop, int offset, int size, float radius)
{
	// nNeighs = edgeNum
	auto tx = threadIdx.x + blockDim.y * threadIdx.y;

	if (tx >= size)
	{
		return;
	}

	bool valid = true;
	bool hole = false;

	// initialize edge_cuda
	float3 d[3];
	float a;
	float3 dik, dij, delta;
	__shared__ float3 shared_norm[128];

	int nIdx = ds_adjMat[ds_prefixSumArr[stIdx] + tx];
	// neighbors[0] - node
	dij = make_float3(ds_nodesLst[edIdx].x - ds_nodesLst[stIdx].x, ds_nodesLst[edIdx].y - ds_nodesLst[stIdx].y, ds_nodesLst[edIdx].z - ds_nodesLst[stIdx].z);
	dev_normalize(dij);
	// neighbos[x] - node
	dik = make_float3(ds_nodesLst[nIdx].x - ds_nodesLst[stIdx].x, ds_nodesLst[nIdx].y - ds_nodesLst[stIdx].y, ds_nodesLst[nIdx].z - ds_nodesLst[stIdx].z);
	dev_normalize(dik);
	delta = dev_cross(dik, dij);
	if (dev_norm(delta) < 1e-3f)
	{
		d[2] = make_float3(-dij.x, -dij.y, -dij.z);
		d[0] = make_float3(copysign(d[2].z, d[2].x), copysign(d[2].z, d[2].y), -copysign(d[2].x, d[2].z) - copysign(d[2].y, d[2].z));
		dev_normalize(d[0]);
		d[1] = dev_cross(d[2], d[0]);
		dev_normalize(d[1]);
		a = radius;
		if (nIdx != edIdx)
		{
			valid = false;
		}
		else
		{
			hole = true;
		}
	}
	else
	{
		d[1] = make_float3(delta.x, delta.y, delta.z);
		dev_normalize(d[1]);
		d[0] = make_float3(dij.x + dik.x, dij.y + dik.y, dij.z + dik.z);
		dev_normalize(d[0]);
		d[2] = dev_cross(d[0], d[1]);
		dev_normalize(d[2]);
		float theta = acos(dev_dot(dik, dij));
		a = radius / cos(F_PI / 2 - theta / 2);
	}

	if (!valid)
	{
		shared_norm[tx] = make_float3(0.f, 0.f, 0.f);
	}
	else
	{
		shared_norm[tx] = d[2];
	}

	float3 ai = make_float3(d[0].x * a, d[0].y * a, d[0].z * a);
	float3 bi = make_float3(d[1].x * radius, d[1].y * radius, d[1].z * radius);
	float eSt = -2 * F_PI, eEd = 2 * F_PI;

	for (int j = 0; j < size; j++)
	{
		float3 n = shared_norm[j];
		if (j == tx || dev_norm(n) < 1e-3f || !valid)
		{
			continue;
		}
		// Ellipse intersect with plane
		float na, nb, st;
		// na = n @ a, nb = n @ b
		na = ai.x * n.x + ai.y * n.y + ai.z * n.z;
		nb = bi.x * n.x + bi.y * n.y + bi.z * n.z;

		st = atan2(na, -nb);

		// stEdgeLoop[i] substract [st, ed]
		if (st + F_PI < eSt)
		{
			valid = intersection(eSt - 2 * F_PI, eEd - 2 * F_PI, st, st + F_PI, eSt, eEd);
		}
		else if (eEd < st)
		{
			valid = intersection(eSt, eEd, st - 2 * F_PI, st - F_PI, eSt, eEd);
		}
		else
		{
			valid = intersection(eSt, eEd, st, st + F_PI, eSt, eEd);
		}
	}

	eEd = fmin(eEd - eSt, float(2 * F_PI));

	compressData(d[0], d[1], valid, hole, a, radius, eSt, eEd, eLoop, tx + offset);
}


__global__ void beamKernel(edge_soa eloop, int* offset, int begin, int size, float radius)
{
	//printf("blockDIm.y: %d, blockIdx.x: %d, threadIdx.y: %d\n", blockDim.y, blockIdx.x, threadIdx.y);
	int eIdx = blockDim.y * blockIdx.x + threadIdx.y;
	if (eIdx >= size)
	{
		return;
	}

	int	stIdx = ds_edgesLst[eIdx + begin].x;
	int	edIdx = ds_edgesLst[eIdx + begin].y;

	short g1_neighs = ds_prefixSumArr[stIdx + 1] - ds_prefixSumArr[stIdx];
	short g2_neighs = ds_prefixSumArr[edIdx + 1] - ds_prefixSumArr[edIdx];

	// coalesced or sequential
	if (g1_neighs <= 16 && g2_neighs <= 16)
	{
		halfWarpTask(stIdx, edIdx, g1_neighs, g2_neighs, offset[eIdx], radius, eloop);
	}
	else
	{
		//sequenctial
		if (g1_neighs > 32)
		{
			if (threadIdx.x == 0)
				hugeTask << < 1, 128 >> > (stIdx, edIdx, eloop, offset[eIdx], g1_neighs, radius);
		}
		else
		{
			fulWarpTask(stIdx, edIdx, g1_neighs, offset[eIdx], radius, eloop);
		}

		if (g2_neighs > 32)
		{
			if (threadIdx.x == 0)
				hugeTask << < 1, 128 >> > (edIdx, stIdx, eloop, offset[eIdx] + g1_neighs, g2_neighs, radius);
		}
		else
		{
			fulWarpTask(edIdx, stIdx, g2_neighs, offset[eIdx] + g1_neighs, radius, eloop);
		}
	}

	return;
}

void batchStream_cuda(int batchBegin, int batchSize, float radius, edge_soa& loop, std::vector<int2>& edgesLst, std::vector<int>& prefixSumArr)
{
	int* offset, * dev_offset;
	offset = new int[batchSize];
	int size = 0;
	for (size_t i = 0; i < batchSize; i++)
	{
		offset[i] = size;
		size += prefixSumArr[edgesLst[i + batchBegin].x + 1] - prefixSumArr[edgesLst[i + batchBegin].x];
		size += prefixSumArr[edgesLst[i + batchBegin].y + 1] - prefixSumArr[edgesLst[i + batchBegin].y];
	}

	edge_soa dev_loop;
	dev_loop.deviceMemAlloc(size);
	checkCuda(cudaMalloc((void**)&dev_offset, batchSize * sizeof(int)));
	checkCuda(cudaMemcpy(dev_offset, offset, batchSize * sizeof(int), cudaMemcpyHostToDevice));

	//! Each Blcok Run One Beam Task Group
	dim3 blockDim(32, 4);
	dim3 gridDim = (batchSize - 1) / 4 + 1;
	beamKernel << <gridDim, blockDim >> > (dev_loop, dev_offset, batchBegin, batchSize, radius);

	checkCuda(cudaGetLastError());
	checkCuda(cudaDeviceSynchronize());
	loop.hostMemAlloc(size);

	checkCuda(cudaMemcpy(loop.a, dev_loop.a, size * sizeof(float), cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(loop.b, dev_loop.b, size * sizeof(float), cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(loop.data_a, dev_loop.data_a, size * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(loop.data_b, dev_loop.data_b, size * sizeof(uint64_t), cudaMemcpyDeviceToHost));

	dev_loop.deviceMemFree();
	cudaFree(dev_offset);
	delete[] offset;
}

void transferDataToDevice(std::vector<float3>& nodesLst, std::vector<int2>& edgesLst, std::vector<int>& prefixSumArr, std::vector<int>& adjMat)
{
	auto nNodes = nodesLst.size();
	auto nEdges = edgesLst.size();
	cudaSetDevice(0);
	cudaMalloc((void**)&dev_nodesLst, nNodes * sizeof(float3));
	cudaMalloc((void**)&dev_edgesLst, nEdges * sizeof(int2));
	cudaMalloc((void**)&dev_prefixSumArr, (nNodes + 1) * sizeof(int));
	cudaMalloc((void**)&dev_adjMat, nEdges * 2 * sizeof(int));

	checkCuda(cudaMemcpy(dev_nodesLst, &nodesLst[0], sizeof(float3) * nNodes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dev_edgesLst, &edgesLst[0], sizeof(int2) * nEdges, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dev_prefixSumArr, &prefixSumArr[0], sizeof(int) * (nNodes + 1), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(dev_adjMat, &adjMat[0], sizeof(int) * nEdges * 2, cudaMemcpyHostToDevice));

	checkCuda(cudaMemcpyToSymbol(ds_nodesLst, &dev_nodesLst, sizeof(float3*), size_t(0), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(ds_prefixSumArr, &dev_prefixSumArr, sizeof(int*), size_t(0), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(ds_adjMat, &dev_adjMat, sizeof(int*), size_t(0), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(ds_edgesLst, &dev_edgesLst, sizeof(int2*), size_t(0), cudaMemcpyHostToDevice));
}

void releaseDevice()
{
	cudaFree(dev_nodesLst);
	cudaFree(dev_edgesLst);
	cudaFree(dev_prefixSumArr);
	cudaFree(dev_adjMat);
}