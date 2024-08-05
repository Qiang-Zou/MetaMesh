/**
 * @file LSLT.h
 * @author redblacksoup (redblacksoup@gmail.com)
 * @brief
 * @version 1.0
 * @date 2023-05-05
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once
#include "pch.h"
#include "core.h"
#include "cudaBeamTri.cuh"


struct Data {
	int batchBegin;
	int batchSize;
	edge_soa eLoop;
};

class LatticeTri
{
private:
	const static int WRITE_BUFFER_SIZE = 10000;
	const static int STREAM_NUM = 2;

	double URADIAN;
	FILE* fp;

	// Lattice structure info
	std::vector<float3> nodesLst;
	std::vector<int2> edgesLst;
	std::vector<int> prefixSumArr;
	std::vector<int> adjMat;
	std::unordered_map<int, std::list<std::vector<vec3f>>> holes_map;
	std::mutex mMutex;
	// Support function
	int createTriangle(std::vector<face>& mesh, const vec3f& p1, const vec3f& p2, const vec3f& p3);
	int secToTri(edge_soa& eLoop, int num, int offset, int nIdx, const mat& basisMat, std::vector<vertex>& vlst);
public:
	// Parameter used for triangulation
	double RADIUS = 0.3;
	double CHORD_ERROR = 0.05;
	// Basic attributes
	int nodeNum = 0;
	int edgeNum = 0;
	int batchBeams = 0;
	int batchNum = 0;
	unsigned long nFaces = 0;
	std::string path;
	// Step 1: Load file
	Data loadFile(int batchIdx);
	// Step 4: Triangulate & Output
	void beamTriangulation(Data dat, std::vector<face>& mesh);
	// Step 5: Write mesh to .stl file
	int writeSTL(std::vector<face>& mesh);
	// Postprocess: Fill holes
	int holeFill();
	// Basic function
	LatticeTri(std::string path_, int batchBeams_ = 4096, double radius_ = 0.3, double chord_error_ = 0.05);

	~LatticeTri();
};
