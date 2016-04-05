#pragma once

#define MAX_DISTANCE 20

class TrackedPoint
{
public:
	TrackedPoint(cv::Point2f location, int frameId, double depth = 0);
	TrackedPoint(cv::Point2f location, int frameId, cv::KeyPoint keyPt, cv::Mat desc, double matchScore = MAX_DISTANCE,
               double depth = 0);

  cv::Point2d undist(cv::Mat const& K, cv::Mat const& dist);
public:
	int frameId;
	cv::Point2f loc;
	cv::KeyPoint keyPt;
	cv::Mat desc;
  double matchScore;
  double depth;

};

