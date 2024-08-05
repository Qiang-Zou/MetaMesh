#pragma once
#include "pch.h"

struct node
{
	int idx;
	std::array<double, 3> p;
};

void loadModel(std::string nodePath, std::string edgePath, std::vector<node>& nodesList, std::vector<std::array<int, 2>>& edgesList)
{
	int nNodes, nEdges;
	int start_from_zero = true;
	nodesList.clear();
	edgesList.clear();
	std::ifstream rfile(nodePath, std::ios::in);
	if (rfile)
	{
		std::string line;
		std::getline(rfile, line);
		std::istringstream ss(line.c_str());
		ss >> nNodes;
		nodesList.resize(nNodes);
		for (int i = 0; i < nNodes; i++)
		{
			ss.clear();
			std::getline(rfile, line);
			ss.str(line.c_str());
			int idx;
			double x, y, z;
			ss >> idx >> x >> y >> z;
			nodesList[i] = { idx, x, y, z };
			if (i == 0)
			{
				switch (idx)
				{
				case 0:
					start_from_zero = true;
					break;
				case 1:
					start_from_zero = false;
					break;
				default:
					std::cerr << "Error: Undefined start number!" << std::endl;
					return;
				}
			}
		}
	}
	else
	{
		std::cerr << "Error: file not exists!" << std::endl;
		return;
	}
	rfile.close();

	rfile.open(edgePath, std::ios::in);
	if (rfile)
	{
		std::string line;
		std::getline(rfile, line);
		std::istringstream ss(line.c_str());
		ss >> nEdges;
		edgesList.resize(nEdges);
		for (int i = 0; i < nEdges; i++)
		{
			ss.clear();
			std::getline(rfile, line);
			ss.str(line.c_str());
			int idx, e1, e2;
			ss >> idx >> e1 >> e2;
			if (start_from_zero == false)
			{
				e1--;
				e2--;
			}
			assert(e1 < nNodes && e1 >= 0 && e2 < nNodes && e2 >= 0);
			edgesList[i] = { e1, e2 };
		}
	}
	else
	{
		std::cerr << "Error: file not exists!" << std::endl;
		return;
	}
	rfile.close();
}

/**
 * @brief Compress TXT graph file into binary expression.
 *
 * @param nodePath
 * @param edgePath
 */
void binaryFile(std::string nodePath, std::string edgePath)
{
	// Read .node & .edge file, save into list
	// nodeLst from 0 to nNodes
	// edgeLst[i][0] >= 0 & edgeLst[i][1] >= 0
	std::vector<std::array<int, 2>> edgesList;
	std::vector<node> nodesList;
	loadModel(nodePath, edgePath, nodesList, edgesList);
	uint32_t nNodes = nodesList.size();
	uint32_t nEdges = edgesList.size();
	std::string basename = edgePath.substr(0, edgePath.rfind("."));

	// Create and fill the adjacency matrix
	std::cout << "Begin create sparse matrix!\n";
	Eigen::SparseMatrix<uint32_t, Eigen::RowMajor>* adj_mat = new Eigen::SparseMatrix<uint32_t, Eigen::RowMajor>(nNodes, nNodes);
	{
		// This fragment needs a lot of memory usage
		std::vector<Eigen::Triplet<uint32_t>> tripletList;
		tripletList.reserve(2 * nEdges);
		for (size_t i = 0; i < nEdges; i++)
		{
			int a = edgesList[i][0];
			int b = edgesList[i][1];
			tripletList.emplace_back(a, b, 1);
			tripletList.emplace_back(b, a, 1);
		}
		adj_mat->setFromTriplets(tripletList.begin(), tripletList.end());
	}

	// write temp file
	FILE* fp = nullptr;
	printf("Begin write binary node file!\n");

	fp = fopen((basename + "_compressed.node").c_str(), "wb");
	if (fp)
	{
		uint32_t nNodesLong = nNodes;
		fwrite((uint32_t*)&nNodesLong, 4, 1, fp);
		for (size_t i = 0; i < nNodes; i++)
		{
			float s[3]{ float(nodesList[i].p[0]), float(nodesList[i].p[1]), float(nodesList[i].p[2]) };
			fwrite(s, 3 * sizeof(float), 1, fp);
		}
		fclose(fp);
	}
	else
	{
		printf("File could not be opened\n");
	}

	printf("Begin write binary edge file!\n");
	fp = fopen((basename + "_compressed.edge").c_str(), "wb");
	if (fp)
	{
		uint32_t nEdgesLong = nEdges;
		fwrite((uint32_t*)&nEdgesLong, 4, 1, fp);

		for (size_t i = 0; i < nEdges; i++)
		{
			uint32_t s[2]{ edgesList[i][0], edgesList[i][1] };
			fwrite(s, 2 * sizeof(uint32_t), 1, fp);
		}
		fclose(fp);
	}
	else
	{
		printf("File could not be opened\n");
	}

	printf("Begin write binary adj file!\n");
	fp = fopen((basename + "_compressed.adj").c_str(), "wb");
	if (fp)
	{
		fwrite((uint32_t*)&nNodes, 4, 1, fp);
		fwrite((uint32_t*)&nEdges, 4, 1, fp);
		// find neighbor of end vertices
		int count = 0;
		for (size_t stIdx = 0; stIdx < nNodes; stIdx++)
		{
			for (Eigen::SparseMatrix<uint32_t, Eigen::RowMajor>::InnerIterator it(*adj_mat, stIdx); it; ++it)
			{
				++count;
			}
			fwrite((uint32_t*)&count, sizeof(uint32_t), 1, fp);
		}
		for (size_t stIdx = 0; stIdx < nNodes; stIdx++)
		{
			std::vector<uint32_t> neighbors;
			neighbors.reserve(20);
			for (Eigen::SparseMatrix<uint32_t, Eigen::RowMajor>::InnerIterator it(*adj_mat, stIdx); it; ++it)
			{
				neighbors.push_back(it.col());
			}
			if (neighbors.size() > 0)
			{
				fwrite(&(neighbors[0]), sizeof(uint32_t), neighbors.size(), fp);
			}
		}
		fclose(fp);
	}
	else
	{
		printf("File could not be opened\n");
	}
	delete adj_mat;
	return;
}

class PreProcess
{
public:
	std::filesystem::path target_dir;

	PreProcess(std::string tar_dir)
	{
		target_dir = tar_dir;
	};

	void processing(bool cover = true)
	{
		std::string model_name = target_dir.filename().string();
		std::filesystem::path file_path = target_dir / model_name;

		auto node_path = file_path.replace_extension("node");
		auto edge_path = file_path.replace_extension("edge");

		// if compressed file is exist
		std::filesystem::path compressed_path = target_dir / (model_name + "_compressed");
		auto adjc_compressed_path = compressed_path.replace_extension("adj");
		auto node_compressed_path = compressed_path.replace_extension("node");
		auto edge_compressed_path = compressed_path.replace_extension("edge");

		if (std::filesystem::exists(adjc_compressed_path) && std::filesystem::exists(node_compressed_path) && std::filesystem::exists(edge_compressed_path))
		{
			return;
		}

		if (!std::filesystem::exists(node_path) && !std::filesystem::exists(edge_path))
		{
			std::cerr << "Error: No usable .node and .edge file" << std::endl;
		}
		else
		{
			binaryFile(node_path.string(), edge_path.string());
		}
		return;
	};

	std::string get_dir()
	{
		return this->target_dir.string();
	}

	void set_dir(std::string dir)
	{
		this->target_dir = dir;
	}
};