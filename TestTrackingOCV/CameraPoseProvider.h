#pragma once
#include <memory>
#include "opencv2/core/core.hpp"

struct CameraPose {
  cv::Mat R;
  cv::Mat t;
};

class CameraPoseProvider
{
public:
  CameraPoseProvider();
  ~CameraPoseProvider();

  void getCurrentPose(CameraPose& cameraPose); //last from history?

protected:
  CameraPose curCameraPose; // or CameraPose*
  std::vector<std::shared_ptr<CameraPose>> history;

};
