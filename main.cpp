#include <opencv2/opencv.hpp>
#include <vector>


using namespace cv;


int main(int argc, char** argv) {
    //1
    Mat imgLena;
    imgLena = imread("../sample/lena.jpg", IMREAD_COLOR);
    imshow("Image", imgLena);

    //2
    Mat imgLenaGray;
    cvtColor(imgLena, imgLenaGray, COLOR_BGR2GRAY);
    imshow("Image2",imgLenaGray);

    //3
    Mat lenaContr;
    equalizeHist(imgLenaGray, lenaContr);

    //imshow("Histogram",lenaContr);

    //4
    Mat lenaCanny;
    Canny(imgLenaGray, lenaCanny, 40, 255);

    //imshow("Canny",lenaCanny);

    //5
    Mat gray = imgLenaGray.clone();
    std::vector<cv::Point2f> corners;
    goodFeaturesToTrack(gray, corners, 1000, 0.01, 10);
    int x = 0, y = 0;
    for (size_t i = 0; i < corners.size(); i++) {
        circle(gray, corners.at(i), 2, 255);
    }

    //imshow("5 Corners",gray);


    //6
    Mat dst;
    Mat finalDst;

    distanceTransform(255 - gray, dst, DIST_L1, 3);

    normalize(dst, finalDst, 0, 1.0, NORM_MINMAX);
    
    //imshow("result",finalDst);

    std::string ty = std::to_string(gray.type());
    printf("Matrix: %s %dx%d \n", ty.c_str(), dst.cols, dst.rows);



    Mat lenaGrayClone;

    imgLenaGray.convertTo(lenaGrayClone, CV_32F);

    Mat lenaGrayResizeBorder = imgLenaGray.clone();
    
    copyMakeBorder(lenaGrayResizeBorder, lenaGrayResizeBorder, 0, 300, 0, 200, BORDER_REPLICATE);
    
    float k = 0.1;

    for (int x = 0; x < imgLena.rows; x++) {
        for (int y = 0; y < imgLena.cols; y++) {

            int distance = (int)(dst.at<float>(x, y) * k);

            Mat kernel(distance, distance, CV_32F, 1.0);

            kernel *= 1.0 / (distance * distance);

            if (distance != 0) {
                lenaGrayClone.at<float>(x, y) = .0f;
            }

            for (int i = 0; i < kernel.rows; i++) {
                for (int j = 0; j < kernel.cols; j++) {
                    lenaGrayClone.at<float>(x, y) += kernel.at<float>(i, j) * (int)lenaGrayResizeBorder.at<uint8_t>(x + i, y + j);
                }
            }
        }
    }

    normalize(lenaGrayClone, lenaGrayClone, 0, 1.0, NORM_MINMAX);

    //imshow("Filtered", lenaGrayClone);


    //7


    Mat lenaInteg;
    integral(imgLenaGray, lenaInteg, -1);

    copyMakeBorder(lenaInteg, lenaInteg, 0, 300, 0, 300, BORDER_REPLICATE);
    
    Mat filtered = imgLenaGray;
    filtered.convertTo(filtered, CV_32F);

    float param = 0.1;


    for (int i = 0; i < imgLena.rows; i++)
    {
        for (int j = 0; j < imgLena.cols; j++)
        {
            int kernel = (int)(dst.at<float>(i, j) * param);
            if (kernel != 0) {
                filtered.at<float>(i, j) = .0f;
                float sum = lenaInteg.at<int>(i, j) +
                    lenaInteg.at<int>(i + kernel, j + kernel) -
                    lenaInteg.at<int>(i + kernel, j) -
                    lenaInteg.at<int>(i, j + kernel);
                filtered.at<float>(i, j) = sum / (kernel * kernel);
            }
        }
    }

    normalize(filtered, filtered, 0, 1.0, NORM_MINMAX);
    imshow("IntegralFiltered", filtered);



    waitKey(0);


}
