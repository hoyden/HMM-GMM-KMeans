#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "KMeans.h"
#include "GMM.h"
using namespace std;
using namespace Eigen;
int main()
{
    MatrixXd data(10,3);
    data << 0.0, 0.2, 0.4,
            0.3, 0.2, 0.4,
            0.4, 0.2, 0.4,
            0.5, 0.2, 0.4,
            5.0, 5.2, 8.4,
            6.0, 5.2, 7.4,
            4.0, 5.2, 4.4,
            10.3, 10.4, 10.5,
            10.1, 10.6, 10.7,
            11.3, 10.2, 10.9;

    const int size = 10; //Number of samples
    const int dim = 3;   //Dimension of feature
    //const int cluster_num = 4; //Cluster number
    
    MatrixXd test_data(4,3);
    test_data << 0.1, 0.2, 0.3, 
                 0.4, 0.5, 0.6, 
                 5.0, 6.2, 8.4, 
                 10.3, 10.4, 10.5;

	//Test GMM
    
    GMM *gmm = new GMM(dim,3); //GMM has 3 SGM
    
    gmm->Train(data,size); //Training GMM
    
    printf("\nTest GMM:\n");
    for(int i = 0; i < 4; ++i)
	{   
        cout << "The Probability of " << test_data.row(i) << " is " << gmm->GetProbability(test_data.row(i)) << endl;
	}

    //save GMM to file
	ofstream gmm_file("gmm.txt");
	assert(gmm_file);
	gmm_file<<*gmm;
	gmm_file.close();

	delete gmm;

    return 0;
}