#pragma once
#include <unordered_map>

struct CameraPose {
  CameraPose() {}

  CameraPose(cv::Mat const& RData, cv::Mat const& tData)
  {
    RData.copyTo(R);
    tData.copyTo(t);
  }

  cv::Mat R;
  cv::Mat t;
};

class CameraPoseProvider
{
public:
  virtual int getCameraPoseForFrame(CameraPose &cameraPose, int frameId);
  virtual int getCeresCameraForFrame(double * camera, int frameId);
  virtual int getProjMatrForFrame(cv::Mat & projMatr, int frameId);

protected:
  //key: frameId
  std::unordered_map<int, CameraPose> poses;
};
