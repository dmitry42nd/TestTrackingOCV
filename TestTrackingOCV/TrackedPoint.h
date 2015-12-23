#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
class TrackedPoint
{
public:
	TrackedPoint(cv::Point2f location, int frameId, double score, cv::KeyPoint keyPt, cv::Mat desc, double depth = 0);
	~TrackedPoint();

	cv::Point2f location;
	int frameId;
	double matchScore;
	cv::KeyPoint keyPt;
	cv::Mat desc;
	double depth;
};

