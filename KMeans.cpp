/***************************************************************************
Module Name:
	KMeans

History:
	2003/10/16	Fei Wang
	2013 luxiaoxun
	2019 caomiao
***************************************************************************/
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "KMeans.h"

using namespace std;
using namespace Eigen;

KMeans::KMeans(int dimNum, int clusterNum)
{
	m_dimNum = dimNum;
	m_clusterNum = clusterNum;
	m_means = MatrixXd(m_clusterNum, m_dimNum);
	m_initMode = InitRandom;
	m_maxIterNum = 100;
	m_endError = 0.001;
}


/*
void KMeans::Cluster(const char* sampleFileName, const char* labelFileName)
{
	// Check the sample file
	ifstream sampleFile(sampleFileName, ios_base::binary);
	assert(sampleFile);

	int size = 0;
	int dim = 0;
	sampleFile.read((char*)&size, sizeof(int));
	sampleFile.read((char*)&dim, sizeof(int));
	assert(size >= m_clusterNum);
	assert(dim == m_dimNum);

	// Initialize model
	Init(sampleFile);

	// Recursion
	double* x = new double[m_dimNum];	// Sample data
	int label = -1;		// Class index
	double iterNum = 0;
	double lastCost = 0;
	double currCost = 0;
	int unchanged = 0;
	bool loop = true;
	int* counts = new int[m_clusterNum];
	double** next_means = new double*[m_clusterNum];	// New model for reestimation
	for (int i = 0; i < m_clusterNum; i++)
	{
		next_means[i] = new double[m_dimNum];
	}

	while (loop)
	{
		memset(counts, 0, sizeof(int) * m_clusterNum);
		for (int i = 0; i < m_clusterNum; i++)
		{
			memset(next_means[i], 0, sizeof(double) * m_dimNum);
		}

		lastCost = currCost;
		currCost = 0;

		sampleFile.clear();
		sampleFile.seekg(sizeof(int) * 2, ios_base::beg);

		// Classification
		for (int i = 0; i < size; i++)
		{
			sampleFile.read((char*)x, sizeof(double) * m_dimNum);
			currCost += GetLabel(x, &label);

			counts[label]++;
			for (int d = 0; d < m_dimNum; d++)
			{
				next_means[label][d] += x[d];
			}
		}
		currCost /= size;

		// Reestimation
		for (int i = 0; i < m_clusterNum; i++)
		{
			if (counts[i] > 0)
			{
				for (int d = 0; d < m_dimNum; d++)
				{
					next_means[i][d] /= counts[i];
				}
				memcpy(m_means[i], next_means[i], sizeof(double) * m_dimNum);
			}
		}

		// Terminal conditions
		iterNum++;
		if (fabs(lastCost - currCost) < m_endError * lastCost)
		{
			unchanged++;
		}
		if (iterNum >= m_maxIterNum || unchanged >= 3)
		{
			loop = false;
		}
		//DEBUG
		//cout << "Iter: " << iterNum << ", Average Cost: " << currCost << endl;
	}

	// Output the label file
	ofstream labelFile(labelFileName, ios_base::binary);
	assert(labelFile);

	labelFile.write((char*)&size, sizeof(int));
	sampleFile.clear();
	sampleFile.seekg(sizeof(int) * 2, ios_base::beg);

	for (int i = 0; i < size; i++)
	{
		sampleFile.read((char*)x, sizeof(double) * m_dimNum);
		GetLabel(x, &label);
		labelFile.write((char*)&label, sizeof(int));
	}

	sampleFile.close();
	labelFile.close();

	delete[] counts;
	delete[] x;
	for (int i = 0; i < m_clusterNum; i++)
	{
		delete[] next_means[i];
	}
	delete[] next_means;
}
*/

//N 
void KMeans::Cluster(MatrixXd data, int N, int *Label)
{
	int size = N;

	assert(size >= m_clusterNum);

	// Initialize model
	Init(data,N);
	
	// Recursion
	RowVectorXd x(m_dimNum);	// Sample data
	int label = -1;		// Class index
	double iterNum = 0;
	double lastCost = 0;
	double currCost = 0;
	int unchanged = 0;
	bool loop = true;
	int* counts = new int[m_clusterNum];
	MatrixXd next_means(m_clusterNum, m_dimNum);   // New model for reestimation
	
	while (loop)
	{
		memset(counts, 0, sizeof(int) * m_clusterNum);
		next_means = MatrixXd::Zero(m_clusterNum, m_dimNum);
		lastCost = currCost;
		currCost = 0;

		
		// Classification
		for (int i = 0; i < size; i++)
		{
			x = data.row(i);
			currCost += GetLabel(x, &label);
			
			counts[label]++;
			next_means.row(label) += x;
			
		}
		currCost /= size;
		// Reestimation
		for (int i = 0; i < m_clusterNum; i++)
		{
			if (counts[i] > 0)
			{
				next_means.row(i) /= counts[i];
				m_means.row(i) = next_means.row(i);
			}
		}

		// Terminal conditions
		iterNum++;
		if (fabs(lastCost - currCost) < m_endError * lastCost)
		{
			unchanged++;
		}
		if (iterNum >= m_maxIterNum || unchanged >= 3)
		{
			loop = false;
		}

		//DEBUG
		//cout << "Iter: " << iterNum << ", Average Cost: " << currCost << endl;
	}

	// Output the label file
	for (int i = 0; i < size; i++)
	{
		x = data.row(i);
		
		GetLabel(x, &label);
		Label[i] = label;
	}
	delete[] counts;
	
}

void KMeans::Init(Eigen::MatrixXd data, int N)
{
	int size = N;

	if (m_initMode ==  InitRandom)
	{
		RowVectorXd sample(m_dimNum);
		vector<int> select;
		// Seed the random-number generator with current time
		srand((unsigned)time(NULL));
		for(int i = 0; i < m_clusterNum; i++)
		{
			while(true)
			{
				int tmp = rand() % size;
				if(find(select.begin(), select.end(), tmp) == select.end())
				{
					m_means.row(i) = data.row(tmp);
					break;
				}
			}
			
		}
	}
	else if (m_initMode == InitUniform)
	{
		for (int i = 0; i < m_clusterNum; i++)
		{
			int select = i * size / m_clusterNum;
			m_means.row(i) = data.row(select);
		}
	}
	else if (m_initMode == InitManual)
	{
		// Do nothing
	}
}
/*
void KMeans::Init(ifstream& sampleFile)
{
	int size = 0;
	sampleFile.seekg(0, ios_base::beg);
	sampleFile.read((char*)&size, sizeof(int));

	if (m_initMode ==  InitRandom)
	{
		int inteval = size / m_clusterNum;
		double* sample = new double[m_dimNum];

		// Seed the random-number generator with current time
		srand((unsigned)time(NULL));

		for (int i = 0; i < m_clusterNum; i++)
		{
			int select = inteval * i + (inteval - 1) * rand() / RAND_MAX;
			int offset = sizeof(int) * 2 + select * sizeof(double) * m_dimNum;

			sampleFile.seekg(offset, ios_base::beg);
			sampleFile.read((char*)sample, sizeof(double) * m_dimNum);
			memcpy(m_means[i], sample, sizeof(double) * m_dimNum);
		}

		delete[] sample;
	}
	else if (m_initMode == InitUniform)
	{
		double* sample = new double[m_dimNum];

		for (int i = 0; i < m_clusterNum; i++)
		{
			int select = i * size / m_clusterNum;
			int offset = sizeof(int) * 2 + select * sizeof(double) * m_dimNum;

			sampleFile.seekg(offset, ios_base::beg);
			sampleFile.read((char*)sample, sizeof(double) * m_dimNum);
			memcpy(m_means[i], sample, sizeof(double) * m_dimNum);
		}

		delete[] sample;
	}
	else if (m_initMode == InitManual)
	{
		// Do nothing
	}
}
*/
double KMeans::GetLabel(const RowVectorXd sample, int* label)
{
	double dist = -1;
	for (int i = 0; i < m_clusterNum; i++)
	{
		double temp = CalcDistance(sample, m_means.row(i));
		if (temp < dist || dist == -1)
		{
			dist = temp;
			*label = i;
		}
	}
	return dist;
}

double KMeans::CalcDistance(const RowVectorXd x, const RowVectorXd u)
{
	double result = 0;
	RowVectorXd tmp = x - u;
	result = tmp * tmp.transpose();
	return sqrt(result);
}

ostream& operator<<(ostream& out, KMeans& kmeans)
{
	out << "<KMeans>" << endl;
	out << "<DimNum> " << kmeans.m_dimNum << " </DimNum>" << endl;
	out << "<ClusterNum> " << kmeans.m_clusterNum << " </CluterNum>" << endl;

	out << "<Mean>" << endl;
	for (int i = 0; i < kmeans.m_clusterNum; i++)
	{
		
		out << kmeans.m_means.row(i);
		
		out << endl;
	}
	out << "</Mean>" << endl;

	out << "</KMeans>" << endl;
	return out;
}
