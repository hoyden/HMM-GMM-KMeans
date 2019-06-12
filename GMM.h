/***************************************************************************
Module Name:
	Gaussian Mixture Model with Diagonal Covariance Matrix

History:
	2003/11/01	Fei Wang
	2013 luxiaoxun
***************************************************************************/

#pragma once
#include <fstream>
#include <Eigen/Dense>
class GMM
{
public:


	enum InitMode
	{
		InitKMeans,
		InitUniform,
	};
	GMM(int dimNum = 1, int mixNum = 1);
	~GMM();

	// void Copy(GMM* gmm);

	void SetMaxIterNum(int i)	{ m_maxIterNum = i; }
	void SetEndError(double f)	{ m_endError = f; }

	int GetDimNum()			{ return m_dimNum; }
	int GetMixNum()			{ return m_mixNum; }
	int GetMaxIterNum()		{ return m_maxIterNum; }
	double GetEndError()	{ return m_endError; }

	double& Prior(int i)	{ return m_priors[i]; }
	Eigen::MatrixXd Mean(int i)		{ return m_means.row(i); }
	Eigen::MatrixXd Variance(int i)	{ return m_vars[i]; }

	void setPrior(int i,double val)	{  m_priors[i]=val; }
	void setMean(int i,Eigen::RowVectorXd val)		{ m_means.row(i) = val; }
	void setVariance(int i,Eigen::MatrixXd val)	{ m_vars[i] = val; }

	double GetProbability(const Eigen::RowVectorXd sample);

	/*	SampleFile: <size><dim><data>...*/
    //void Init(const char* sampleFileName);
	//void Train(const char* sampleFileName);
	void Init(Eigen::MatrixXd data, int N);
	void Train(Eigen::MatrixXd data, int N);

	void DumpSampleFile(const char* fileName);

	friend std::ostream& operator<<(std::ostream& out, GMM& gmm);
	// friend std::istream& operator>>(std::istream& in, GMM& gmm);

private:
	int m_dimNum;		// 
	int m_mixNum;		// Gaussian 分量个数
	double* m_priors;	// Gaussian 分量概率
	Eigen::MatrixXd m_means;  // Gaussian 均值
	Eigen::MatrixXd* m_vars;	// Gaussian 方差 假设各个维度无关

	// A minimum variance is required. Now, it is the overall variance * 0.01.
	Eigen::RowVectorXd m_minVars;
	int m_maxIterNum;		// The stopping criterion regarding the number of iterations
	double m_endError;		// The stopping criterion regarding the error
	int m_initMode;
private:
	// Return the "j"th pdf, p(x|j).
	double GetProbability(const Eigen::RowVectorXd x, int j);
};
