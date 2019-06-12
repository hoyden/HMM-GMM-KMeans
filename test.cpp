#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <time.h>
using namespace std;
using namespace Eigen;

double division(int a, int b)
{
   if( b == 0 )
   {
      throw "Division by zero condition!";
   }
   return (a/b);
}
int main()
{

    MatrixXd m(3, 3);
	m << 0.1, 0, 0,
        0, 0.2, 0,
        0, 0, 0.4;
    RowVector3d x, u;
    x << 0.5, 0.6, 0.7;
    u << 0.3, 0.2, 0.5;
    cout << m.determinant() << endl;
    cout << m.inverse() << endl;
    cout << m.row(0).size() << endl;
	
    cout << pow(2 * 3.14159, m.row(0).size()/2.) << endl;
	

    cout << x - u << endl;
    cout << (x - u) * m.inverse() << endl;
	cout << exp(-0.5 * (x - u) * m.inverse() * (x - u).transpose()) << endl;
	/*
	VectorXd r(4);
	r = m.row(1);
	
	double* data = r;
	cout << data << endl;
	cout << r <<endl;
	cout << m.row(1) * m.row(2).transpose() << endl;
	*/
    srand((unsigned)time(NULL));
    cout << MatrixXd::Zero(3,3) << endl;

    cout << MatrixXd::Random(3,3) << endl;
    
    cout << MatrixXd::Identity(3,3) << endl;

    MatrixXd a = (x - u).transpose() * (x - u);
    cout << a << endl;
    

    ArrayXXd b = a.array();
    ArrayXXd c = MatrixXd::Identity(3,3);

    cout << b << endl;
    cout << c << endl;

    cout << b * c << endl;
    

    MatrixXd* d = new MatrixXd[3];
    for(int i = 0; i < 3; i++)
    {
        d[i] = MatrixXd::Identity(3,3);
    }
    d[0] /= 3;
    cout << d[0] << endl;
    cout << d[1] << endl;
    cout << d[2] << endl;

    
    cout << "!!!!!!!!!!!!!!!!!!!!" << endl;

    RowVectorXd e = RowVectorXd::Random(5);
    cout << e << endl;
    ArrayXd tmp = e;
    e = tmp * tmp;
    cout << e << endl;
    cout << e.transpose()*e << endl;
	return 0;
    


}