#include "stdafx.h"
#include "CameraPoseProviderTXT.h"
#include "boost/filesystem.hpp"
#include <map>
#include <utility>
namespace fs = boost::filesystem;

//kinect
static double k_data[9] = {522.97697, 0.0,       318.47217,
                           0.0,       522.58746, 256.49968,
                           0.0,       0.0,       1.0};
static double dist_data[4] = {0.18962, -0.38214, 0, 0};

CameraPoseProviderTXT::CameraPoseProviderTXT(std::string& pathToCameraPoses)
{
  K = cv::Mat(3, 3, CV_64F, k_data);
  dist = cv::Mat(1,4, CV_64F, dist_data);

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
    cv::Mat t = cv::Mat(3, 1, CV_64F, tData);

    CameraPose cp(R.t(), -R.t()*t);
    poses.insert(std::make_pair(frameId, cp));
  }
}

int CameraPoseProviderTXT::getCameraPoseForFrame(CameraPose &cameraPose, int frameId)
{
	if (poses.count(frameId) == 0)
    return 1; //empty

  cameraPose = poses[frameId];
  return 0; //success
}

int CameraPoseProviderTXT::getProjMatrForFrame(cv::Mat & projMatr, int frameId)
{
  CameraPose cp;
  if(getCameraPoseForFrame(cp, frameId)) {
    return 1;
  }

  cv::hconcat(cp.R, cp.t, projMatr);
  return 0;
}