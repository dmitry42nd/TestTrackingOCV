#include "stdafx.h"

#include "TrackedPoint.h"


TrackedPoint::TrackedPoint(cv::Point2f location, int frameId, cv::KeyPoint keyPt, cv::Mat desc, double matchScore, double depth) :
  loc(location),
  frameId(frameId),
  depth(depth),
  keyPt(keyPt),
  desc(desc),
  matchScore(matchScore)
{}

TrackedPoint::TrackedPoint(cv::Point2f location, int frameId, double depth) :
  loc(location),
  frameId(frameId),
  depth(depth)
{}

cv::Point2d TrackedPoint::undist(cv::Mat const& K, cv::Mat const& dist)
{
  cv::Mat projPoint(1, 1, CV_64FC2);
  projPoint.at<cv::Vec2d>(0, 0)[0] = loc.x;
  projPoint.at<cv::Vec2d>(0, 0)[1] = loc.y;

  cv::Mat undistPoint = cv::Mat(1,1,CV_64FC2);
  cv::undistortPoints(projPoint, undistPoint, K, dist);

  return cv::Point2d(undistPoint.at<cv::Vec2d>(0,0)[0], undistPoint.at<cv::Vec2d>(0,0)[1]);
}