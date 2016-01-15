#include "stdafx.h"

#include "TrackedPoint.h"


TrackedPoint::TrackedPoint(cv::Point2f location, int frameId, double score, cv::KeyPoint keyPt, cv::Mat desc, double depth) : 
location(location), frameId(frameId), matchScore(score), keyPt(keyPt), desc(desc), depth(depth)
{
}