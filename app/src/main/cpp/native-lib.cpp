#include <jni.h>
#include<stdlib.h>
#include <string>
#include <vector>

#include <opencv/cv.h>
#include <android/bitmap.h>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <android/log.h>

#define LOG_TAG "clog"
#define LOGI(...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,  __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG,  __VA_ARGS__)
#include <math.h>
#include <iostream>
using namespace cv;
using namespace std;
typedef unsigned char BYTE;


extern "C" {

Mat gray; // 当前图片
Mat gray_prev; // 预测图片
vector<Point2f> points[2]; // point0为特征点的原来位置，point1为特征点的新位置
vector<Point2f> initial; // 初始化跟踪点的位置
vector<Point2f> features; // 检测的特征
int maxCount = 500; // 检测的最大特征数
double qLevel = 0.01; // 特征检测的等级
double minDist = 10.0; // 两特征点之间的最小距离
vector<uchar> status; // 跟踪特征的状态，特征的流发现为1，否则为0
vector<float> err;



Mat gra; // 当前图片
Mat g_prev; // 预测图片
int counts = 0;
bool ready = true;
vector<Point2f> feats; // 检测的特征
vector<Point2f> init; // 初始化跟踪点的位置
vector<Point2f> pq[2]; // point0为特征点的原来位置，point1为特征点的新位置
vector<uchar> stat; // 跟踪特征的状态，特征的流发现为1，否则为0
vector<cv::Point2f> scene_cornerssss(4);
vector<Point2f> site[1];
vector<uchar> stat1; // 跟踪特征的状态，特征的流发现为1，否则为0
vector<Point2f> good[1];

int toGray(Mat img, Mat &gray) {
    cvtColor(img, gray, CV_RGBA2GRAY);
    if (gray.rows == img.rows && gray.cols == img.cols) {
        return 1;
    } else {
        return 0;
    }
}



int ORBDET(Mat src,Mat& dst){

    cvtColor(src, dst, CV_RGBA2GRAY);
    std::vector<KeyPoint> keypoints_1;
    Ptr<FeatureDetector> detector = ORB::create();
//    vector<KeyPoint> v;
//    v.clear();
//    Mat temp;
//    cvtColor(src,temp,COLOR_RGBA2GRAY);
    detector->detect(src,keypoints_1);
    drawKeypoints(src,keypoints_1,dst,Scalar::all(-1),DrawMatchesFlags::DRAW_OVER_OUTIMG);
    cvtColor(dst, src, CV_GRAY2RGB);
//    detector->detect (temp, v, cv::Mat());
//    for (int i = 0; i < v.size(); ++i) {
//        const KeyPoint& kp = v[i];
//        circle(src, Point(kp.pt.x, kp.pt.y), 5, Scalar(0,0,255,255));
//    }
    //dst = src;
    return 0;
}

//模板检测实现 画出检测物体部分所对应的框
void temtss(Mat &img1,Mat img2)
{
    cv::Mat dstImg;
    dstImg.create(img1.dims, img1.size, img1.type());
    cv::matchTemplate(img1, img2, dstImg, 0);
    LOGD("dstimg is converted");
    cv::normalize(dstImg, dstImg, 0, 1, 32);
    LOGD("normalize is finished");
    cv::Point minPoint;
    cv::Point maxPoint;
    double* minVal = 0;
    double* maxVal = 0;
    cv::minMaxLoc(dstImg, minVal, maxVal, &minPoint, &maxPoint);
    LOGD("minMaxLoc is finished");
    cv::rectangle(img1, minPoint, cv::Point(minPoint.x + img2.cols, minPoint.y + img2.rows), cv::Scalar(0,0,0), 5, 8);
    LOGD("paint rectangle is finished");
}

vector<Point2f>  match(Mat &img_scene,Mat &dst,Mat &img_object){    //两个图片的特征点提取和特征点匹配

    Ptr<ORB> detector = ORB::create();
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute( img_object, Mat(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, Mat(), keypoints_scene, descriptors_scene );
    BFMatcher matcher;
    std::vector< DMatch> matches;
    matcher.match(descriptors_object, descriptors_scene, matches);
    double max_dist = 0;
    double min_dist = 100;
    for( int i = 0; i < descriptors_object.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    std::vector<DMatch> good_matches;
    for( int i = 0; i < descriptors_object.rows; i++ )
    {
        if( matches[i].distance <= 3*min_dist )
        {
            good_matches.push_back( matches[i]);
        }
    }
    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
    Mat H = findHomography( obj, scene, RANSAC );
    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( img_object.cols, 0 );
    obj_corners[2] = cvPoint( img_object.cols, img_object.rows );
    obj_corners[3] = cvPoint( 0, img_object.rows );
    //std::vector<cv::Point2f> scene_corners(4);
    perspectiveTransform( obj_corners, scene_cornerssss, H);
    line(img_scene, scene_cornerssss[0] , scene_cornerssss[1] , Scalar(0, 255, 0), 4);
    line(img_scene, scene_cornerssss[1] , scene_cornerssss[2] , Scalar(0, 255, 0), 4);
    line(img_scene, scene_cornerssss[2] , scene_cornerssss[3] , Scalar(0, 255, 0), 4);
    line(img_scene, scene_cornerssss[3] , scene_cornerssss[0] , Scalar(0, 255, 0), 4);
    return scene;

}




//-------------------------------------------------------------------------------------------------
// function: addNewPoints
// brief: 检测新点是否应该被添加
// parameter:
// return: 是否被添加标志
//-------------------------------------------------------------------------------------------------
bool addNewPoints()
{
    return points[0].size() <= 10;
}

bool addNewPoint()
{
    return pq[0].size() <= 14;
}

//-------------------------------------------------------------------------------------------------
// function: acceptTrackedPoint
// brief: 决定哪些跟踪点被接受
// parameter:
// return:
//-------------------------------------------------------------------------------------------------
bool acceptTrackedPoint(int i)
{
    return status[i];
}

bool acceptTrackedPointss(int i)
{
    return stat[i];
}

// Color to gray
JNIEXPORT jint JNICALL
Java_com_example_bymyself_NDKInterface_colorToGray(JNIEnv *env, jobject, jlong addrSrc,
                                                    jlong addrDst) {
    Mat &src = *(Mat *) addrSrc;
    Mat &dst = *(Mat *) addrDst;

    int conv;
    jint retVal;
    conv = toGray(src, dst);
    retVal = (jint) conv;
    return retVal;
}

// Histogram
JNIEXPORT jint JNICALL
Java_com_example_bymyself_NDKInterface_histogram(JNIEnv *env, jobject, jlong addrSrc
                                                  ,jlong addrDst) {
    Mat &src = *(Mat *) addrSrc;
    Mat &dst = *(Mat *) addrDst;

    int conv;
    jint retVal;
    conv = ORBDET(src,dst);
    retVal = (jint) conv;
    return retVal;
}

//模板检测的接口函数
JNIEXPORT void JNICALL
Java_com_example_bymyself_NDKInterface_tem(JNIEnv *env, jobject thiz, jlong addrSrc
        ,jlong addrDst ) {
    Mat &src = *(Mat *) addrSrc;
    Mat &dst = *(Mat *) addrDst;
    LOGD("src and dst is ok");
    temtss(src,dst);
}


//ORB特征提取 物体检测
JNIEXPORT jint JNICALL
Java_com_example_bymyself_NDKInterface_OEBMatch(JNIEnv *env, jobject thiz, jlong src_add,
                                                jlong tem_add, jlong dst_add) {
    Mat &img_scene = *(Mat *) src_add;
    Mat &dst = *(Mat *) dst_add;
    Mat &img_object = *(Mat *) tem_add;
    Ptr<ORB> detector = ORB::create();
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute( img_object, Mat(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, Mat(), keypoints_scene, descriptors_scene );
    BFMatcher matcher;
    std::vector< DMatch> matches;
    matcher.match(descriptors_object, descriptors_scene, matches);
    double max_dist = 0;
    double min_dist = 100;
    for( int i = 0; i < descriptors_object.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );
    std::vector<DMatch> good_matches;
    for( int i = 0; i < descriptors_object.rows; i++ )
    {
        if( matches[i].distance <= 3*min_dist )
        {
        good_matches.push_back( matches[i]);
        }
    }
    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
    Mat H = findHomography( obj, scene, RANSAC );
    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( img_object.cols, 0 );
    obj_corners[2] = cvPoint( img_object.cols, img_object.rows );
    obj_corners[3] = cvPoint( 0, img_object.rows );
    std::vector<cv::Point2f> scene_corners(4);
    perspectiveTransform( obj_corners, scene_corners, H);
    line(img_scene, scene_corners[0] , scene_corners[1] , Scalar(0, 255, 0), 4);
    LOGD("0 is finished");
    line(img_scene, scene_corners[1] , scene_corners[2] , Scalar(0, 255, 0), 4);
    LOGD("1 is finished");
    line(img_scene, scene_corners[2] , scene_corners[3] , Scalar(0, 255, 0), 4);
    LOGD("2 is finished");
    line(img_scene, scene_corners[3] , scene_corners[0] , Scalar(0, 255, 0), 4);
    LOGD("3 is finished");
    dst = img_scene;
    return 0;

}



//光流法 实现跟踪
JNIEXPORT void JNICALL
Java_com_example_bymyself_NDKInterface_opticalflow(JNIEnv *env, jobject thiz, jlong src_add,
                                                   jlong dst_add) {
    // TODO: implement opticalflow()
    Mat &frame = *(Mat *) src_add;  //输入的帧frame(手机摄像头获得的图像信息)
    Mat &output = *(Mat *) dst_add; //输出的帧(处理以后画出跟踪的帧)
    //此句代码的OpenCV3版为：
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    frame.copyTo(output);
    // 添加特征点（得到可以跟踪的点）
    if (addNewPoints())
    {
        goodFeaturesToTrack(gray, features, maxCount, qLevel, minDist);      //得到可以跟踪的点,结果放在features之中
        points[0].insert(points[0].end(), features.begin(), features.end()); //points[0]是特征点原来的位置，将可以跟踪的点放入特征点原来的特征点的集合之中
        initial.insert(initial.end(), features.begin(), features.end());     //同理 对初始跟踪点也做相应操作
    }
    if (gray_prev.empty())                                                   //判断一下预测的图片是否为空
    {
        gray.copyTo(gray_prev);                                              //如果为空就将初始的图片赋值给预测的图片
    }
    // L-k光流法运动估计
    calcOpticalFlowPyrLK(gray_prev, gray, points[0], points[1], status, err);  //进行L-K光流法的运动估计
    // 去掉一些不好的特征点
    int k = 0;
    for (size_t i = 0; i < points[1].size(); i++)
    {
        if (acceptTrackedPoint(i))
        {
            initial[k] = initial[i];
            points[1][k++] = points[1][i];
        }
    }
    points[1].resize(k);
    initial.resize(k);
    // 显示特征点和运动轨迹
    for (size_t i = 0; i < points[1].size(); i++)
    {
        line(output, initial[i], points[1][i], Scalar(0, 0, 255));
        circle(output, points[1][i], 3, Scalar(0, 255, 0), -1);
    }

    // 把当前跟踪结果作为下一次的参考
    swap(points[1], points[0]);
    swap(gray_prev, gray);
}


//Canny算法 边缘检测
JNIEXPORT void JNICALL
Java_com_example_bymyself_NDKInterface_getEdge(JNIEnv *env, jobject, jobject bitmap) {
    AndroidBitmapInfo info;
    void *pixels;

    CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
    CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
              info.format == ANDROID_BITMAP_FORMAT_RGB_565);
    CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
    CV_Assert(pixels);
    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        Mat temp(info.height, info.width, CV_8UC4, pixels);
        Mat gray;
        cvtColor(temp, gray, COLOR_RGBA2GRAY);
        Canny(gray, gray, 100, 185);
        cvtColor(gray, temp, COLOR_GRAY2RGBA);
    } else {
        Mat temp(info.height, info.width, CV_8UC2, pixels);
        Mat gray;
        cvtColor(temp, gray, COLOR_RGB2GRAY);
        Canny(gray, gray, 100, 185);
        cvtColor(gray, temp, COLOR_GRAY2RGB);
    }
    AndroidBitmap_unlockPixels(env, bitmap);
}

//JNIEXPORT void JNICALL
//Java_com_example_bymyself_NDKInterface_trace(JNIEnv *env, jobject thiz, jlong src_add, jlong tem_add,
//                                             jlong dst_add) {
//    Mat &frame  = *(Mat *) src_add;
//    Mat &output = *(Mat *) dst_add;
//    Mat &img_object = *(Mat *) tem_add;
//    if(ready == true)
//    {
//        feats = match(frame,output,img_object);
//        ready = false;
//        LOGD("3 is finished");
//    }
//    else
//    {
//        for(int i = 0 ;i < 4 ;i++)
//        {
//            feats.push_back(scene_cornerssss[i]);
//        }
//
//        //此句代码的OpenCV3版为：
//        cvtColor(frame, gra, COLOR_BGR2GRAY);
//        frame.copyTo(output);
//        // 添加特征点（得到可以跟踪的点）
//        if (addNewPoint())
//        {
//            pq[0].insert(pq[0].end(), feats.begin(), feats.end()); //points[0]是特征点原来的位置，将可以跟踪的点放入特征点原来的特征点的集合之中
//            init.insert(init.end(), feats.begin(), feats.end());     //同理 对初始跟踪点也做相应操作
//            LOGD("999999999999999999999999999999999999999999999");
//        }
//        if (g_prev.empty())                                                   //判断一下预测的图片是否为空
//        {
//            gra.copyTo(g_prev);                                              //如果为空就将初始的图片赋值给预测的图片
//        }
//        calcOpticalFlowPyrLK(g_prev, gra, pq[0], pq[1], stat, err);  //进行L-K光流法的运动估计
//        calcOpticalFlowPyrLK(g_prev, gra, scene_cornerssss, site[0],stat1,err);
//        int k = 0;
//        for (size_t i = 0; i < pq[1].size(); i++)
//        {
//            if (acceptTrackedPointss(i))
//            {
//                init[k] = init[i];
//                pq[1][k++] = pq[1][i];
//            }
//        }
//        pq[1].resize(k);
//        init.resize(k);
//        // 显示特征点和运动轨迹
//        for (size_t i = 0; i < pq[1].size(); i++)
//        {
//            line(output, init[i], pq[1][i], Scalar(0, 0, 255));
//            circle(output, pq[1][i], 3, Scalar(0, 255, 0), -1);
//        }
//        if(counts == 80)
//        {
//            LOGD("22222222222222222222222222222222222222");
//            ready = true;
//            counts = 0;
//        } else
//        {
//            LOGD("2ooooooooooooooooooooooooooooooooooooo2");
//            line(output, site[0][0] , site[0][1], Scalar(0, 255, 0), 4);
//            line(output, site[0][1] , site[0][2], Scalar(0, 255, 0), 4);
//            line(output, site[0][2] , site[0][3], Scalar(0, 255, 0), 4);
//            line(output, site[0][3] , site[0][0], Scalar(0, 255, 0), 4);
//            counts++;
//        }
//
//        // 把当前跟踪结果作为下一次的参考
//        feats = pq[1];
//        scene_cornerssss = site[0];
//        swap(pq[1], pq[0]);
//        swap(g_prev, gra);
//    }
//
//}


bool panduan(vector<cv::Point2f>& pre,vector<cv::Point2f>& now)        //判断结果为false说明两者差距过大，需要重新
{
        double a[8];
        double min = 200;
       a[0] = abs(pre[0].x-now[0].x);
       a[1] = abs(pre[0].y-now[0].y);
       a[2] = abs(pre[1].x-now[1].x);
       a[3] = abs(pre[1].y-now[1].y);
       a[4] = abs(pre[2].x-now[2].x);
       a[5] = abs(pre[2].y-now[2].y);
       a[6] = abs(pre[3].x-now[3].x);
       a[7] = abs(pre[3].y-now[3].y);

    for(int i = 0 ;i < 8;i++)
       {
           if(a[i] > min)
               return false;
       }
      return true;

}

bool getN(Mat &img_scene,Mat &img_object,vector<Point2f>& cbb,vector<Point2f>& kkl){    //两个图像特征点比较  cbb为最好的结果 kk1为跟踪结果
    if(panduan(cbb, kkl) && kkl.size()==4)
    {
        LOGD("333333333333333333333333333333333333333 is finished");
        cbb = kkl;
        return false;
    }
    Ptr<ORB> detector = ORB::create();
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute( img_object, Mat(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, Mat(), keypoints_scene, descriptors_scene );
    BFMatcher matcher(NORM_HAMMING);
    std::vector< DMatch> matches;
    matcher.match(descriptors_object, descriptors_scene, matches);
    double max_dist = 0;
    double min_dist = 100;
    for( int i = 0; i < descriptors_object.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    std::vector<DMatch> good_matches;
    for( int i = 0; i < descriptors_object.rows; i++ )
    {
        if( matches[i].distance <= 2*min_dist )
        {
            good_matches.push_back( matches[i]);
        }
    }
    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
    Mat H = findHomography( obj, scene, RANSAC );
    if(H.empty())
         return true;
    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( img_object.cols, 0 );
    obj_corners[2] = cvPoint( img_object.cols, img_object.rows );
    obj_corners[3] = cvPoint( 0, img_object.rows );
    perspectiveTransform( obj_corners, scene_cornerssss, H);
    cbb = scene_cornerssss;
    feats = scene;
    return true;

}



bool getS(Mat &img_scene,Mat &img_object,vector<Point2f>& LKP){
    Ptr<ORB> detector = ORB::create();
    std::vector<KeyPoint> keypoints_object, keypoints_scene,lkp;
    Mat descriptors_object, descriptors_scene,descriptors_lkp;
    KeyPoint::convert(LKP, lkp, 1, 1, 0, -1);
    detector->detectAndCompute( img_object, Mat(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, Mat(), keypoints_scene, descriptors_scene );
    detector->compute(img_scene,lkp,descriptors_lkp);
    BFMatcher matcher(NORM_HAMMING);
    BFMatcher matcher1(NORM_HAMMING);
    std::vector< DMatch> matches;
    std::vector< DMatch> matches1;
    matcher.match(descriptors_object, descriptors_scene, matches);
    matcher1.match(descriptors_object, descriptors_lkp, matches1);
    double max_dist = 0;
    double min_dist = 100;
    int x = descriptors_object.rows;
    for( int i = 0; i < x; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    std::vector<DMatch> good_matches;
    for( int i = 0; i < x; i++ )
    {
        if( matches[i].distance <= 3*min_dist )
        {
            good_matches.push_back( matches[i]);
        }
    }

    double max_dist1 = 0;
    double min_dist1 = 100;
    for( int i = 0; i < x; i++ )
    { double dist = matches1[i].distance;
        if( dist < min_dist ) min_dist1= dist;
        if( dist > max_dist ) max_dist1 = dist;
    }
    std::vector<DMatch> good_matches1;
    for( int i = 0; i < x; i++ )
    {
        if( matches1[i].distance <= 3*min_dist1 )
        {
            good_matches1.push_back( matches1[i]);
        }
    }
    double a =(double) good_matches.size()/ keypoints_scene.size();
    double b =(double) good_matches1.size()/lkp.size();
    if(b >= a)
        return false;

    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
    Mat H = findHomography( obj, scene, RANSAC );
    if(H.empty())
    {
        LOGD("33333333333333333333333333333333333");
        return false;
    }

    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( img_object.cols, 0 );
    obj_corners[2] = cvPoint( img_object.cols, img_object.rows );
    obj_corners[3] = cvPoint( 0, img_object.rows );
    perspectiveTransform( obj_corners, scene_cornerssss, H);
    LKP = scene_cornerssss;
    feats = scene;
    return true;

}




bool flag = true;
bool flags = true;
JNIEXPORT void JNICALL
Java_com_example_bymyself_NDKInterface_trace(JNIEnv *env, jobject thiz, jlong src_add, jlong tem_add,
                                             jlong dst_add) {
    Mat &frame = *(Mat *) src_add;
    Mat &output = *(Mat *) dst_add;
    Mat &img_object = *(Mat *) tem_add;
    if (ready == true) {
        feats = match(frame, output, img_object);
        ready = false;
    } else {
        cvtColor(frame, gra, COLOR_BGR2GRAY);
        frame.copyTo(output);
        if (addNewPoint()) {
            pq[0].insert(pq[0].end(), feats.begin(),feats.end());    //points[0]是特征点原来的位置，将可以跟踪的点放入特征点原来的特征点的集合之中
            init.insert(init.end(), feats.begin(), feats.end());     //同理 对初始跟踪点也做相应操作
        }
        if (g_prev.empty())                                                   //判断一下预测的图片是否为空
        {
            gra.copyTo(g_prev);                                              //如果为空就将初始的图片赋值给预测的图片
        }
        calcOpticalFlowPyrLK(g_prev, gra, pq[0], pq[1], stat, err);  //进行L-K光流法的运动估计,跟踪内部的点
        calcOpticalFlowPyrLK(g_prev, gra, scene_cornerssss, site[0], stat1, err);   //跟踪模板的边界点
        if(flag)
        {
            good[0] = scene_cornerssss;
            flag = false;
            LOGD("4444444444444444444444444444444 is finished");
        }

        int k = 0;
        for (size_t i = 0; i < pq[1].size(); i++) {
            if (acceptTrackedPointss(i)) {
                init[k] = init[i];
                pq[1][k++] = pq[1][i];
            }
        }
        pq[1].resize(k);
        init.resize(k);
        //显示特征点和运动轨迹
        for (size_t i = 0; i < pq[1].size(); i++) {
            line(output, init[i], pq[1][i], Scalar(0, 0, 255));
            circle(output, pq[1][i], 3, Scalar(0, 255, 0), -1);
        }
        for(int i = 0;i < 4;i++)
        {
            if(stat1[i]== 0)
                flags = false;
        }
        if (!getN(frame,img_object,good[0],site[0])&&flags) {              //good[0]表示最好的匹配结果 site[0]表示跟踪的结果
            line(output, site[0][0], site[0][1], Scalar(0, 255, 0), 4);
            line(output, site[0][1], site[0][2], Scalar(0, 255, 0), 4);
            line(output, site[0][2], site[0][3], Scalar(0, 255, 0), 4);
            line(output, site[0][3], site[0][0], Scalar(0, 255, 0), 4);
            // 把当前跟踪结果作为下一次的参考
            feats = pq[1];
            scene_cornerssss = site[0];
            swap(pq[1], pq[0]);
            swap(g_prev, gra);
            LOGD("55555555555555555555555555555555555555555 is finished");
        } else
        {
            line(output, site[0][0], site[0][1], Scalar(0, 255, 0), 4);
            line(output, site[0][1], site[0][2], Scalar(0, 255, 0), 4);
            line(output, site[0][2], site[0][3], Scalar(0, 255, 0), 4);
            line(output, site[0][3], site[0][0], Scalar(0, 255, 0), 4);
            LOGD("6666666666666666666666666666666666666666666666 is finished");
        }
    }
}



bool flag1s = true;
JNIEXPORT void JNICALL
Java_com_example_bymyself_NDKInterface_trace11(JNIEnv *env, jobject thiz, jlong src_add, jlong tem_add,
                                             jlong dst_add) {
    Mat &frame = *(Mat *) src_add;
    Mat &output = *(Mat *) dst_add;
    Mat &img_object = *(Mat *) tem_add;
    if (ready == true) {
        feats = match(frame, output, img_object);
        ready = false;
    } else {
        cvtColor(frame, gra, COLOR_BGR2GRAY);
        frame.copyTo(output);
        if (addNewPoint()) {
            pq[0].insert(pq[0].end(), feats.begin(),feats.end());    //points[0]是特征点原来的位置，将可以跟踪的点放入特征点原来的特征点的集合之中
            init.insert(init.end(), feats.begin(), feats.end());     //同理 对初始跟踪点也做相应操作
        }
        if (g_prev.empty())                                                   //判断一下预测的图片是否为空
        {
            gra.copyTo(g_prev);                                              //如果为空就将初始的图片赋值给预测的图片
        }
        calcOpticalFlowPyrLK(g_prev, gra, pq[0], pq[1], stat, err);  //进行L-K光流法的运动估计,跟踪内部的点
        calcOpticalFlowPyrLK(g_prev, gra, scene_cornerssss, site[0], stat1, err);   //跟踪模板的边界点
        int k = 0;
        for (size_t i = 0; i < pq[1].size(); i++) {
            if (acceptTrackedPointss(i)) {
                init[k] = init[i];
                pq[1][k++] = pq[1][i];
            }
        }
        pq[1].resize(k);
        init.resize(k);
        //显示特征点和运动轨迹
        for (size_t i = 0; i < pq[1].size(); i++) {
            line(output, init[i], pq[1][i], Scalar(0, 0, 255));
            circle(output, pq[1][i], 3, Scalar(0, 255, 0), -1);
        }

        for(int i = 0;i < 4;i++)
        {
            if(stat1[i] == 0)
                flag1s = false;
        }
        if(flag1s == false)
            LOGD("llllllllllllllllllllllllllllllllllllllllllll is finished");
        if (!getS(frame,img_object,pq[1]) && flag1s) {              //good[0]表示最好的匹配结果 site[0]表示跟踪的结果
            line(output, site[0][0], site[0][1], Scalar(0, 255, 0), 4);
            line(output, site[0][1], site[0][2], Scalar(0, 255, 0), 4);
            line(output, site[0][2], site[0][3], Scalar(0, 255, 0), 4);
            line(output, site[0][3], site[0][0], Scalar(0, 255, 0), 4);
            // 把当前跟踪结果作为下一次的参考
            feats = pq[1];
            scene_cornerssss = site[0];
            swap(pq[1], pq[0]);
            swap(g_prev, gra);
            LOGD("55555555555555555555555555555555555555555 is finished");
        } else
        {
            line(output, site[0][0], site[0][1], Scalar(0, 255, 0), 4);
            line(output, site[0][1], site[0][2], Scalar(0, 255, 0), 4);
            line(output, site[0][2], site[0][3], Scalar(0, 255, 0), 4);
            line(output, site[0][3], site[0][0], Scalar(0, 255, 0), 4);
            if(!flag1s)
               flag1s = true;
//            ready = true;
            LOGD("6666666666666666666666666666666666666666666666 is finished");
        }
    }
}




}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_bymyself_NDKInterface_mtf(JNIEnv *env, jobject thiz, jlong native_obj_addr,
                                           jlong native_obj_addr1, jlong native_obj_addr2) {
    Mat &frame = *(Mat *) native_obj_addr;
    Mat &output = *(Mat *) native_obj_addr2;
    Mat &img_object = *(Mat *) native_obj_addr1;

}