#include "pch.h"
#include "LatticeTri.h"

LatticeTri::LatticeTri(std::string path_, int batchBeams_, double radius_, double chord_error_)
{
	this->RADIUS = radius_;
	this->CHORD_ERROR = chord_error_;
	this->URADIAN = 2 * acos(1 - CHORD_ERROR);
	
	this->path = path_;
	this->batchBeams = batchBeams_;
	// Open .node & .edge file
	// Create .stl file header
	std::string::size_type iPos;
	if (strstr(path_.c_str(), "\\"))
	{
		iPos = path_.find_last_of('\\') + 1;
	}
	else
	{
		iPos = path.find_last_of('/') + 1;
	}
	std::string modelName = path_.substr(iPos, path_.length() - iPos);
	std::cout << "modelName: " << modelName << "\n";
	FILE* eStream = fopen((path_ + "/" + modelName + "_compressed.edge").c_str(), "rb");
	if (eStream)
	{
		fread((uint32_t*)&edgeNum, sizeof(uint32_t), 1, eStream);
		edgesLst.resize(edgeNum);
		for (size_t i = 0; i < edgeNum; i++)
		{
			uint32_t s[2];
			fread(s, sizeof(uint32_t), 2, eStream);
			edgesLst[i] = make_int2(s[0], s[1]);
		}
		fclose(eStream);
	}
	else
	{
		printf("Error: .edge file could not be opened!\n");
	}

	FILE* nStream = fopen((path_ + "/" + modelName + "_compressed.node").c_str(), "rb");
	if (nStream)
	{
		fread(&nodeNum, sizeof(uint32_t), 1, nStream);
		nodesLst.resize(nodeNum);
		for (size_t i = 0; i < nodeNum; i++)
		{
			float s[3];
			fread(s, sizeof(float), 3, nStream);
			nodesLst[i] = make_float3(s[0], s[1], s[2]);
		}
		fclose(nStream);
	}
	else
	{
		printf("Error: .node file could not be opened!\n");
	}

	FILE* adjStream = fopen((path_ + "/" + modelName + "_compressed.adj").c_str(), "rb");
	if (adjStream)
	{
		uint32_t s1, s2;
		fread((uint32_t*)&s1, 4, 1, adjStream);
		fread((uint32_t*)&s2, 4, 1, adjStream);
		prefixSumArr.resize(nodeNum + 1);
		prefixSumArr[0] = 0;
		adjMat.resize(edgeNum * 2);
		fread(&prefixSumArr[1], 4, nodeNum, adjStream);
		fread(&adjMat[0], 4, edgeNum * 2, nStream);
		fclose(adjStream);
	}
	else
	{
		printf("Error: .adj file could not be opened!\n");
	}

	fp = fopen((path_ + "/" + modelName + ".stl").c_str(), "wb");
	if (fp)
	{
		char head[80];
		memset(head, 0, 80);
		strcpy(head, "solid STL generated by lslt");
		fwrite(head, 80, 1, fp);
		unsigned long nTriLong = 0;
		fwrite((char*)&nTriLong, 4, 1, fp);
	}
	else
	{
		printf("Error: can not open file\n");
	}

	assert(prefixSumArr[nodeNum] == edgeNum * 2);
	printf("Dataset: %s\nnode num: %d\nedge num: %d\n", path_.c_str(), nodeNum, edgeNum);
	// initialize batch plan
	this->batchNum = (edgeNum - 1) / batchBeams + 1;
	printf("Total batch num: %d\nbatch size: %d\n", batchNum, batchBeams);
	printf("Begin transfer data from host to device.\n");
	transferDataToDevice(nodesLst, edgesLst, prefixSumArr, adjMat);
}

LatticeTri::~LatticeTri()
{
	if (fseek(fp, 80, SEEK_SET) == -1)
	{
		printf("Error: can not seek to 80!\n");
	}
	unsigned long nTriLong = (unsigned long)this->nFaces;
	fwrite((char*)&nTriLong, 4, 1, fp);
	printf("Total triangles: %d\n", this->nFaces);
	releaseDevice();
	fclose(fp);
}

Data LatticeTri::loadFile(int batchIdx)
{
	if (batchIdx >= batchNum)
	{
		std::cout << "Error: batchIdx overflow!\n";
		return Data{};
	}
	int batchBegin = batchBeams * batchIdx;
	int batchSize = std::min(batchBeams, edgeNum - batchBegin);
	printf("%d. batch size: %d, batch begin: %d, ", batchIdx, batchSize, batchBegin);
	edge_soa eLoop;
	batchStream_cuda(batchBegin, batchSize, RADIUS, eLoop, edgesLst, prefixSumArr);
	return Data{ batchBegin, batchSize, eLoop };
}

void LatticeTri::beamTriangulation(Data dat, std::vector<face>& mesh)
{
	int batchBegin = dat.batchBegin;
	int batchSize = dat.batchSize;
	int ptrOff = 0;
	// Trianglate result
	mesh.reserve(batchSize * 32);
	for (int k = 0; k < batchSize; k++)
	{
		int stIdx = edgesLst[batchBegin + k].x;
		int edIdx = edgesLst[batchBegin + k].y;

		int stNum = prefixSumArr[stIdx + 1] - prefixSumArr[stIdx];
		int edNum = prefixSumArr[edIdx + 1] - prefixSumArr[edIdx];

		mat bBasisMat;
		float3 d[3];
		d[2] = make_float3(nodesLst[edIdx].x - nodesLst[stIdx].x, nodesLst[edIdx].y - nodesLst[stIdx].y, nodesLst[edIdx].z - nodesLst[stIdx].z);
		normalize(d[2]);
		d[0] = make_float3(copysign(d[2].z, d[2].x), copysign(d[2].z, d[2].y), -copysign(d[2].x, d[2].z) - copysign(d[2].y, d[2].z));
		normalize(d[0]);
		d[1] = cross(d[2], d[0]);
		normalize(d[1]);
		bBasisMat << d[0].x, d[0].y, d[0].z,
			d[1].x, d[1].y, d[1].z,
			d[2].x, d[2].y, d[2].z;

		std::vector<vertex> stVlst, edVlst;
		// Discretize section to vertices
		secToTri(dat.eLoop, stNum, ptrOff, stIdx, bBasisMat, stVlst);
		secToTri(dat.eLoop, edNum, ptrOff + stNum, edIdx, bBasisMat, edVlst);
		ptrOff += (stNum + edNum);

		stVlst.push_back({ stVlst[0].p, stVlst[0].props + 2.f * F_PI + 1 });
		edVlst.push_back({ edVlst[0].p, edVlst[0].props + 2.f * F_PI + 1 });

		// suture sections, get triangles
		int i = 0, j = 0;
		bool stFlag = (stVlst[0].props < edVlst[0].props) ? true : false;
		do
		{
			if (stFlag)
			{
				createTriangle(mesh, stVlst[i].p, stVlst[i + 1].p, edVlst[j].p);
				if (stVlst[i + 1].props > edVlst[j].props)
				{
					stFlag = false;
				}
				i++;
			}
			else
			{
				createTriangle(mesh, edVlst[j].p, stVlst[i].p, edVlst[j + 1].p);
				if (edVlst[j + 1].props > stVlst[i].props)
				{
					stFlag = true;
				}
				j++;
			}
		} while (i < stVlst.size() - 1 || j < edVlst.size() - 1);
	}
	dat.eLoop.hostMemFree();
	return;
}

/**
 * @brief Get points on sections used to triangulation, interpolation strategy is defined here.
 *
 * @param sec
 * @param vlst
 * @return success 1, fail 0
 */
int LatticeTri::secToTri(edge_soa& eLoop, int num, int offset, int nIdx, const mat& basisMat, std::vector<vertex>& vlst)
{
	// add vertices to vertex list
	vlst.clear();
	vlst.reserve(32);
	// verts used to record vertices forming a hole
	std::vector<vec3f> verts;
	verts.reserve(16);
	// get Node i
	vec3f Ni(nodesLst[nIdx].x, nodesLst[nIdx].y, nodesLst[nIdx].z);
	for (int i = 0; i < num; i++)
	{
		uint64_t data_a = eLoop.data_a[offset + i];
		uint64_t data_b = eLoop.data_b[offset + i];
		float ua = eLoop.a[offset + i];
		float ub = eLoop.b[offset + i];
		// get_valid() = false
		if ((data_a & 0x8000000000000000) == 0)
		{
			continue;
		}

		// st, len
		float t, segLength;
		{
			uint32_t bin = uint32_t(data_a >> 41);
			bin &= 0b1111111111111111111111;
			t = (float)bin * F_PI / 2097152.f - F_PI;
		}
		{
			uint32_t bin = uint32_t(data_b >> 41);
			bin &= 0b1111111111111111111111;
			segLength = (float)(bin) * 2 * F_PI / 0b1111111111111111111111;
		}

		int frag = int(segLength / URADIAN) + 1;
		float u = segLength / frag;
		// major axis A , minor axis B
		float3 fa = getAxis(data_a, ua);
		float3 fb = getAxis(data_b, ub);
		vec3f a(fa.x, fa.y, fa.z);
		vec3f b(fb.x, fb.y, fb.z);
		vec3f v;
		for (int j = 0; j <= frag; j++)
		{
			// float t = st + u * j;
			// get point
			v = a * cos(t) + b * sin(t);
			if (j != frag)
			{
				// sort vertices according to radian on section, props means its radian in section basis
				vec3f transVec = basisMat * v;
				float norm = sqrt(transVec[0] * transVec[0] + transVec[1] * transVec[1]);
				float props = atan2(transVec[1] / norm, transVec[0] / norm);
				vlst.emplace_back(v + Ni, props);
			}
			// get_hole() == true
			if ((data_b & 0x8000000000000000) != 0)
			{
				// add vertices to related holes_list
				verts.emplace_back(v + Ni);
			}
			t += u;
		}
	}

	if (!verts.empty())
	{
		std::lock_guard<std::mutex> lock(this->mMutex);
		holes_map[nIdx].push_back(verts);
	}
	assert(vlst.size() > 0);
	// sort vertices from the project radians with section basis
	std::sort(vlst.begin(), vlst.end(), [](const vertex& a, const vertex& b)
		{ return a.props < b.props; });
	return 1;
}

int LatticeTri::holeFill()
{
	printf("Not holes triangles: %d\n", this->nFaces);
	std::vector<face> mesh;
	mesh.reserve(1024 * 1024);
	for (auto& it : holes_map)
	{
		vec3f Ni(nodesLst[it.first].x, nodesLst[it.first].y, nodesLst[it.first].z);
		vec3f barycenter;
		if (it.second.size() == 1)
		{
			// only one circle -> a beam section without intersection
			barycenter = Ni;
		}
		else
		{
			// get barycenter of vertices forming the hole
			int cnt = 0;
			barycenter = vec3f(0, 0, 0);
			for (auto& vlst : it.second)
			{
				for (int i = 0; i < vlst.size() - 1; i++)
				{
					barycenter += vlst[i];
					cnt++;
				}
			}
			barycenter /= cnt;
			barycenter = barycenter - Ni;
			barycenter = Ni + RADIUS * (barycenter).normalized();
		}

		for (auto& vlst : it.second)
		{
			for (int i = 0; i < vlst.size() - 1; i++)
			{
				createTriangle(mesh, barycenter, vlst[i], vlst[i + 1]);
			}
		}
	}
	writeSTL(mesh);

	return 1;
}

int LatticeTri::createTriangle(std::vector<face>& mesh, const vec3f& p1, const vec3f& p2, const vec3f& p3)
{
	mesh.emplace_back(p1, p2, p3);
	return mesh.size() - 1;
}

/**
 * @brief Write mesh to STL files.
 *
 * @param path
 * @return
 */
int LatticeTri::writeSTL(std::vector<face>& mesh)
{
	this->nFaces += mesh.size();
	int write_ptr = 0;
	char wtbuf[WRITE_BUFFER_SIZE];
	const int c = 50;
	for (int i = 0; i < mesh.size(); i++)
	{
		// write to buffer
		{
			if (write_ptr + c > WRITE_BUFFER_SIZE)
			{
				fwrite(wtbuf, write_ptr, 1, fp);
				write_ptr = 0;
			}
			memcpy(&wtbuf[write_ptr], mesh[i].s, c);
			write_ptr += c;
		}
	}
	if (write_ptr)
	{
		fwrite(wtbuf, write_ptr, 1, fp);
		write_ptr = 0;
	}
	return 1;
}