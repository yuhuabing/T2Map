#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <iostream>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "T2Map/ORBextractor.h"
#include <cv_bridge/cv_bridge.h>
#include "T2Map/Converter.h"
#include "Thirdparty/DBoW2/include/DBoW2/DBoW2.h"
#include "Thirdparty/DBoW2/include/DBoW2/TemplatedDatabase.h"
#include "Thirdparty/DBoW2/include/DBoW2/TemplatedVocabulary.h"
using namespace std;
using namespace ORB_SLAM2;
using namespace cv;
std::string strSettingPath = "/home/yhb/T2Map_ws/src/T2Map/config/EuRoC.yaml";
cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
ORBextractor* mpORBextractorLeft;
int n=0;
std::vector<std::vector<cv::KeyPoint>> vvKeys; //用来存每一帧提取的特征点
std::vector<cv::Mat> vDescriptors; //用来存每一帧提取点的描述子
std::vector<nav_msgs::Odometry> vpose; //存下来每一帧的位姿
std::vector<std::vector<cv::Point3d>> vPoint3d; // 每一帧对应的3D点
std::vector<std::vector<int>> vPointID; // 每一帧3D点对应的2D点ID
OrbDatabase db;
OrbVocabulary* voc;
cv::Mat K = cv::Mat::eye(3,3,CV_32F);
cv::Mat DistCoef(4,1,CV_32F);

void triangulation(
  const vector<KeyPoint> &keypoint_1,
  const vector<KeyPoint> &keypoint_2,
  const std::vector<DMatch> &matches,
  const Mat &R, const Mat &t,
  vector<Point3d> &points
);
Point2f pixel2cam(const Point2d &p, const Mat &K);
void trans1toW(std::vector<cv::Point3d> &points_1,std::vector<cv::Point3d> &points_w,Eigen::Matrix4d T);
void callback(const sensor_msgs::Image::ConstPtr &img_msg,const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    cv_bridge::CvImageConstPtr cv_ptr_image;
    cv_ptr_image = cv_bridge::toCvShare(img_msg, "mono8");
    std::vector<cv::KeyPoint> mvKeys;
    cv::Mat mDescriptors;
    cv::Mat im= cv_ptr_image->image;
    (*mpORBextractorLeft)(im,				//待提取特征点的图像
						cv::Mat(),
						mvKeys,			//输出变量，用于保存提取后的特征点
						mDescriptors);
    vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
    if(n==0){
        vvKeys.push_back(mvKeys);
        vDescriptors.push_back(mDescriptors);
        n++;
        vpose.push_back(*pose_msg);
        std::vector<cv::Point3d> a;
        vPoint3d.push_back(a);
        std::vector<int> b;
        vPointID.push_back(b);
        return;
    }
    else if(n==1){
        std::vector<cv::DMatch> matches;
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
        vector<cv::DMatch> match;
        cv::Mat lastDescriptors=vDescriptors.back();// 上一帧特征点的描述子
        std::vector<cv::KeyPoint> lastKeys = vvKeys.back(); // 上一帧的特征点
        matcher->match(lastDescriptors, mDescriptors, match);
        double min_dist = 10000, max_dist = 0;
        for (int i = 0; i < lastDescriptors.rows; i++) {
            double dist = match[i].distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }
        for (int i = 0; i < lastDescriptors.rows; i++) {
            if (match[i].distance <= max(2 * min_dist, 30.0)) {
                matches.push_back(match[i]);
            }
        }
        // 根据匹配三角化点
        nav_msgs::Odometry lastpose = vpose.back();
        Eigen::Quaterniond q_l(lastpose.pose.pose.orientation.w,lastpose.pose.pose.orientation.x,lastpose.pose.pose.orientation.y,lastpose.pose.pose.orientation.z);
        Eigen::Matrix3d Rw1 = q_l.toRotationMatrix();
        Eigen::MatrixXd tw11(3,1);
        tw11 << lastpose.pose.pose.position.x,lastpose.pose.pose.position.y,lastpose.pose.pose.position.z;
        Eigen::MatrixXd tw12(1,4);
        tw12 << 0,0,0,1;
        Eigen::Matrix4d Tw1 ;
        Tw1.block(0,0,3,3) = Rw1;
        Tw1.block(0,3,3,1) = tw11;
        Tw1.block(3,0,1,4) = tw12;
        Eigen::Quaterniond q_m(pose_msg->pose.pose.orientation.w,pose_msg->pose.pose.orientation.x,pose_msg->pose.pose.orientation.y,pose_msg->pose.pose.orientation.z);
        Eigen::Matrix3d Rw2 = q_m.toRotationMatrix();
        Eigen::MatrixXd t21(3,1);
        t21 << pose_msg->pose.pose.position.x,pose_msg->pose.pose.position.y,pose_msg->pose.pose.position.z;
        Eigen::MatrixXd t22(1,4);
        t22 << 0,0,0,1;
        Eigen::Matrix4d Tw2;
        Tw2.block(0,0,3,3)= Rw2;
        Tw2.block(0,3,3,1) = t21;
        Tw2.block(3,0,1,4) = t22;
        Eigen::Matrix4d T21 = Tw2.inverse() * Tw1;
        double array_R[3][3] = { T21(0,0), T21(0,1), T21(0,2), T21(1,0), T21(1,1), T21(1,2),T21(2,0), T21(2,1),T21(2,2)};
        cv::Mat R(3,3,CV_32FC1,array_R);
        double array_t[3][1] = {T21(0,3) , T21(1,3),T21(2,3)};
        cv::Mat t(3,1,CV_32FC1,array_t);
        vector<Point3d> points_1;
        triangulation(lastKeys, mvKeys, matches, R, t, points_1);
        std::vector<int> matchedID;
        for (DMatch m:matches) 
            matchedID.push_back(m.trainIdx);
        vPointID.push_back(matchedID);
        vector<Point3d> points_w;
        trans1toW(points_1,points_w,Tw1);
        vPoint3d.push_back(points_w);
        vpose.push_back(*pose_msg);
        vvKeys.push_back(mvKeys);
        vDescriptors.push_back(mDescriptors);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "listener");
    ros::NodeHandle n;
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    //构造相机内参矩阵
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    // 图像矫正系数
    // [k1 k2 p1 p2 k3]
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    //有些相机的畸变系数中会没有k3项
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    // Load ORB parameters

    // 加载ORB特征点有关的参数,并新建特征点提取器

    // 每一帧提取的特征点数 1000
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    // 图像建立金字塔时的变化尺度 1.2
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    // 尺度金字塔的层数 8
    int nLevels = fSettings["ORBextractor.nLevels"];
    // 提取fast特征点的默认阈值 20
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    // 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];
    mpORBextractorLeft = new ORBextractor(
        nFeatures,      
        fScaleFactor,
        nLevels,
        fIniThFAST,
        fMinThFAST);
    // 消息同步回调
    message_filters::Subscriber<sensor_msgs::Image> image_sub(n, "image", 1);
    message_filters::Subscriber<nav_msgs::Odometry> pose_sub(n, "pose", 1);
    typedef  message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry> MySyncPolicy;
    // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_sub, pose_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2));

    ros::spin();
    return 0;
}
void triangulation(
  const vector<KeyPoint> &keypoint_1,
  const vector<KeyPoint> &keypoint_2,
  const std::vector<DMatch> &matches,
  const Mat &R, const Mat &t,
  vector<Point3d> &points) {
  Mat T1 = (Mat_<float>(3, 4) <<
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0);
  Mat T2 = (Mat_<float>(3, 4) <<
    R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
  );

  // Mat K = (Mat_<double>(3, 3) << 525.0, 0, 319.5, 0, 525.0 , 239.5, 0, 0, 1);
  vector<Point2f> pts_1, pts_2;
  for (DMatch m:matches) {
    // 将像素坐标转换至相机坐标
    pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
    pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
  }

  Mat pts_4d;
  cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

  // 转换成非齐次坐标
  for (int i = 0; i < pts_4d.cols; i++) {
    Mat x = pts_4d.col(i);
    x /= x.at<float>(3, 0); // 归一化
    Point3d p(
      x.at<float>(0, 0),
      x.at<float>(1, 0),
      x.at<float>(2, 0)
    );
    points.push_back(p);
  }
}
Point2f pixel2cam(const Point2d &p, const Mat &K) {
  return Point2f
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}
void trans1toW(std::vector<cv::Point3d> &points_1,std::vector<cv::Point3d> &points_w,Eigen::Matrix4d T){
    for(int i=0;i<points_1.size();i++){
      Eigen::MatrixXd point1(4,1);
      point1 << points_1[i].x,points_1[i].y,points_1[i].z,1;
      Eigen::MatrixXd pointW(4,1);
      pointW = T * point1;
      cv::Point3d tmp;
      tmp.x = pointW(0,0),tmp.y = pointW(1,0);tmp.z = pointW(2,0);
      points_w.push_back(tmp);
    }
    return;
}