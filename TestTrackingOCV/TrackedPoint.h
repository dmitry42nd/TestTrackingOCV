#pragma once

class TrackedPoint
{
public:
	TrackedPoint(cv::Point2f location, int frameId, double score, cv::KeyPoint keyPt, cv::Mat desc, double depth = 0);

	double matchScore;
  double depth;

  cv::Point2f location;
	cv::KeyPoint keyPt;
	cv::Mat desc;

  int frameId;
};

