#include "stdafx.h"
#include "CameraPoseProviderTXT.h"
#include "boost/filesystem.hpp"
#include <map>
#include <utility>
namespace fs = boost::filesystem;

CameraPoseProviderTXT::CameraPoseProviderTXT(std::string& pathToCameraPoses)
{
  poses.clear();
  std::ifstream cameraPosesData(pathToCameraPoses);
  int frameId;

  double RData[9];
  double tData[3];
  while (cameraPosesData >> frameId) {
    for (auto i = 0; i < 9; i++) {
      cameraPosesData >> RData[i];
    }
    for (auto i = 0; i < 3; i++) {
      cameraPosesData >> tData[i];
    }
    cv::Mat R = cv::Mat(3, 3, CV_64F, RData);
    cv::Mat t = cv::Mat(1, 3, CV_64F, tData);
    cv::Mat P;
    cv::hconcat(R.t(), -R.t()*t.t(), P);
    //P = K*P;
    poses.insert(std::make_pair(frameId, P));
  }
  frameNum = 0;
}

void CameraPoseProviderTXT::setCurrentFrameNumber(int frameInd)
{
	frameNum = frameInd;
}

CameraPoseProviderTXT::~CameraPoseProviderTXT()
{
}

void CameraPoseProviderTXT::getCurrentPose(CameraPose& cameraPose)
{	
	getPoseForFrame(cameraPose, frameNum);
}

//better to return projection matrix
void CameraPoseProviderTXT::getPoseForFrame(CameraPose& cameraPose, int qFrameNum)
{
	cameraPose.R = cv::Mat::zeros(0, 0, CV_64F);
	cameraPose.t = cv::Mat::zeros(0, 0, CV_64F);
	if (poses.count(qFrameNum) > 0)
	{
		//std::cout << "pose given " << std::endl;
		cameraPose.R = poses[qFrameNum](cv::Rect(0, 0, 3, 3));
		cameraPose.t = poses[qFrameNum](cv::Rect(3, 0, 1, 3));
	}
}