#include "../headerspace/WatershedAlg.h"
#include<opencv2/ximgproc.hpp>
using namespace cv;
using namespace ximgproc;


void WatershedAlg::processImage(Mat &image,Mat &duplImage,Array2D<int> &threshmat0,Array2D<int> & markers,Array2D<bool> &visArr,Array1D &plotx,Array1D &ploty,Array2D<int>& plots,Array2D<bool> &inprioq,Array2D<int> &markerMap,Array2D<int> &temp,Array2D<int>& nextSet,int**** arr4D,int**** mat4D,Bool2D &visBool,int **platmarker,int **distance) {
    image = makeImageGrayScale(image);
    Array2D<int>testMat(image.rows,image.cols,0);	
	
    int morph_size = 2;
    Mat dialelement = getStructuringElement(cv::MORPH_RECT, Size(2 * morph_size + 1,2 * morph_size + 1),
    Point(morph_size, morph_size));

    Mat dill(image.rows,image.cols,CV_8UC1,Scalar::all(0));//

    dilate(image, dill, dialelement,Point(-1, -1), 1);//
      
    Mat eroelement = getStructuringElement(cv::MORPH_RECT, Size(2 * morph_size + 1,2 * morph_size + 1),
    Point(morph_size, morph_size));
    Mat eroimg(image.rows,image.cols,CV_8UC1,Scalar::all(0));////
    cv::erode(dill,eroimg,eroelement);////
	
    Mat tarImg;

    dill.copyTo(tarImg);

    cv::Canny(dill,tarImg,12,20);
    #pragma omp parallel for
    for(int i=0;i<image.rows;i++){
     for(int j=0;j<image.cols;j++){
        threshmat0(i,j)=(int)tarImg.at<uchar>(i,j);
     }
    }
     

      threshmat0=antiInverseImage(threshmat0,image.rows,image.cols);
      threshmat0=distanceTransform(threshmat0,markers,image.rows,image.cols,plots,visArr,plotx,ploty,arr4D,mat4D,visBool,platmarker,distance);
      int id_num=0;
      image = watershed(threshmat0, markers,duplImage,image.rows,image.cols,inprioq,markerMap,temp,nextSet,id_num,testMat);
  //     cv::imshow("watershed result",image);
    //  cv::waitKey(0);






    }
