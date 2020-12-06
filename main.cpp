#include <opencv2/opencv.hpp>
#include <vector>


using namespace cv;


int main(int argc, char** argv) {
    //1
    Mat imgLena;
    imgLena = imread("../sample/lena.jpg",IMREAD_COLOR);
    //imshow("Image", imgLena);

    //2
    Mat imgLenaGray;
    cvtColor(imgLena,imgLenaGray,COLOR_BGR2GRAY);
    //imshow("Image2",imgLenaGray);

    //3
    Mat lenaContr;
    equalizeHist(imgLenaGray,lenaContr);
    //imshow("Histogram",lenaContr);

    //4
    Mat lenaCanny;
    Canny(imgLenaGray,lenaCanny,40,255);
    //imshow("Canny",lenaCanny);

    //5
    Mat gray = imgLenaGray.clone();
    std::vector<cv::Point2f> corners;
    goodFeaturesToTrack(gray,corners,1000,0.01,10);
    int x = 0, y = 0;
    for(size_t i = 0; i < corners.size(); i++){
        circle(gray,corners.at(i),2,255);
    }
    imshow("lol",imgLenaGray);
    //imshow("5 Corners",gray);
    

    //6
    Mat dst;
    Mat finalDst;
    // Mat test(1920,1080,CV_8UC1,255);
    // test.at<uchar>(960,540) -= 255;
    // imshow("test",test);
    // distanceTransform(test,test,DIST_L1,3);
    // imshow("disttest",test);
    // std::cout << std::endl;
    // std::cout << test;
    // normalize(test,test,0.4,1,NORM_MINMAX);
    // std::cout << std::endl;
    // std::cout << test;
    // imshow("normal",test);

    //imshow("gray",gray);
    distanceTransform(255 - gray,dst,DIST_L1,3);
    normalize(dst,finalDst,0,1.0,NORM_MINMAX);
    //imshow("result",finalDst);
    //imshow("orig",dst);
    std::string ty =  std::to_string( gray.type() );
    printf("Matrix: %s %dx%d \n", ty.c_str(), dst.cols, dst.rows ); 
    
    
    // Mat integ;
	// integral(imgLenaGray, integ, -1);

	// Mat lenaGrayClone = imgLenaGray;
	// for (int i = 0; i < imgLena.rows; i++)
	// 	{
	// 		for (int j = 0; j < imgLena.cols; j++)
	// 		{
	// 			int kernel = dst.at<uchar>(i, j);
	// 			if (kernel % 2 == 0)
	// 				kernel++;
	// 			double kernel2 = kernel / 2;
	// 			if ((kernel <= 1) || (i <= kernel2) || (j <= kernel2) || (i > imgLena.rows - kernel2- 1) || (j > imgLena.cols - kernel2 - 1)) continue;
	// 			float sum = (integ.at<uchar>(i - kernel2 - 1, j - kernel2 - 1) - integ.at<uchar>(i - kernel2 - 1, j + kernel2)
	// 					   + integ.at<uchar>(i + kernel2, j + kernel2) - integ.at<uchar>(i + kernel2, j - kernel2 - 1));    //7 ôèëüòðàöèÿ
	// 			lenaGrayClone.at<uchar>(i, j) = sum / (float)(pow(kernel, 2));
	// 		}
	// }
	// imshow("lenaGrayClone", lenaGrayClone);

    // 7
    // int sum = 0;
    // for(int i = 0; i < imgLena.rows; i++){
    //     for(int j = 0; j < imgLena.cols; j++){
    //         int kernel = dst.at<int>(i,j);
    //         sum = kernel * imgLena;
    //     }s
    // }
    
    // Mat kernel(10,10,CV_32F, 1.0);
    // kernel *= 1.0/100;
    
    
    Mat lenaGrayClone;
    imgLenaGray.convertTo(lenaGrayClone,CV_32F);
    //std::cout << lenaGrayClone;


    //imshow("filteredOrigin", lenaGrayClone);
    //imshow("dst", dst);
    float k = 0.1;
   
    for (int x = 0; x < imgLena.rows; x++) {
		for (int y = 0; y < imgLena.cols; y++) {
            
            int distance = (int)(dst.at<float>(x,y) * k);
            
            Mat kernel(distance,distance,CV_32F,1.0);
            
            kernel *= 1.0/(distance*distance);
            
            if(distance!= 0){
                lenaGrayClone.at<float>(x,y) = .0f;
            }
            
            for(int i = 0; i < kernel.rows; i++){
                for (int j = 0; j < kernel.cols; j++) {
                    lenaGrayClone.at<float>(x,y) += kernel.at<float>(i,j) * (int)imgLenaGray.at<uint8_t>(x+i,y+j); 
                }
            }
        }
    }

    //imshow("Filtered2", lenaGrayClone);
    
    normalize(lenaGrayClone,lenaGrayClone,0,1.0,NORM_MINMAX);
    
    //imshow("lenaGrayClone", lenaGrayClone);




    //7

    Mat integ;
	integral(imgLenaGray, integ, -1);
    Mat filtered = imgLenaGray;
    Mat grayM(imgLena.size(),CV_64F);
	
    
     int kerSize = 3;
    for(int i=kerSize; i < imgLena.rows - kerSize; i++){
        for(int j=kerSize; j < imgLena.cols - kerSize; j++){
            grayM.at<double>(i,j) = (integ.at<double>(i+kerSize,j+kerSize) + integ.at<double>(i-kerSize,j-kerSize)
                    - integ.at<double>(i+kerSize,j-kerSize) - integ.at<double>(i-kerSize,j+kerSize))/(kerSize*2+1)/(kerSize*2+1);
        }
    }
    std::cout << "\n type "  << integ.type() << std::endl;
    cv::normalize(grayM,grayM,255,0,cv::NORM_L2);
    cv::imshow("gray",gray/255);
    cv::imshow("gray filter",grayM);




    waitKey(0);


}
