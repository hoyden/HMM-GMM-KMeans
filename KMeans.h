/***************************************************************************
Module Name:
	KMeans

History:
	2003/10/16	Fei Wang
	2013 luxiaoxun
***************************************************************************/

#pragma once
//#include <fstream>
#include <string.h>
#include <assert.h>
#include <Eigen/Dense>

class KMeans
{
public:
    //初始化方法
	enum InitMode
	{
		InitRandom,
		InitManual,
		InitUniform,
	};
    //构造函数，数据维度和类别总数
	KMeans(int dimNum = 1, int clusterNum = 1);
	//~KMeans(){}

	void SetMean(int i, const Eigen::RowVectorXd u){ m_means.row(i) = u; }
	void SetInitMode(int i)				{ m_initMode = i; }
	void SetMaxIterNum(int i)			{ m_maxIterNum = i; }
	void SetEndError(double f)			{ m_endError = f; }

	Eigen::RowVectorXd GetMean(int i)	{ return m_means.row(i); }
	int GetInitMode()		{ return m_initMode; }
	int GetMaxIterNum()		{ return m_maxIterNum; }
	double GetEndError()	{ return m_endError; }


	/*	SampleFile: <size><dim><data>...
		LabelFile:	<size><label>...
	*/
	//void Cluster(const char* sampleFileName, const char* labelFileName);
	void Cluster(Eigen::MatrixXd data, int N, int *Label);
	//void Init(std::ifstream& sampleFile);
	void Init(Eigen::MatrixXd data, int N);
	friend std::ostream& operator<<(std::ostream& out, KMeans& kmeans);

private:
	int m_dimNum;
	int m_clusterNum;
	//double** m_means;  
	Eigen::MatrixXd m_means;  //聚类中心
 	int m_initMode;
	int m_maxIterNum;		// The stopping criterion regarding the number of iterations
	double m_endError;		// The stopping criterion regarding the error

	double GetLabel(const Eigen::RowVectorXd sample, int* label); //根据距离计算sample的类别号，返回距离
	double CalcDistance(const Eigen::RowVectorXd x, const Eigen::RowVectorXd u); //计算向量x和u的距离
};
