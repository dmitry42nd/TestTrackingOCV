#pragma once

#define MAX_DISTANCE 20

class TrackedPoint
{
public:
  TrackedPoint() :
    frameId(0), 
    matchScore(MAX_DISTANCE)
  { }

	TrackedPoint(cv::Point2f location, int frameId, double score, cv::KeyPoint keyPt, cv::Mat desc, double depth = 0);
	TrackedPoint(cv::Point2f location, int frameId);


	double matchScore;
	double depth;

	cv::Point2f location;
	cv::KeyPoint keyPt;
	cv::Mat desc;

	int frameId;
};

