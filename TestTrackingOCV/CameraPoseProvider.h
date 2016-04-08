#pragma once
#include <unordered_map>

struct CameraPose {
  CameraPose() : R(), t() {}

  CameraPose(cv::Mat const& RData, cv::Mat const& tData) : CameraPose()
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
  virtual ~CameraPoseProvider() {};

  virtual int getCameraPoseForFrame(CameraPose &cameraPose, int frameId);
  virtual int getCeresCameraForFrame(double * camera, int frameId);
  virtual int getProjMatrForFrame(cv::Mat & projMatr, int frameId);

  cv::Mat K, dist;
protected:
  //key: frameId
  std::unordered_map<int, CameraPose> poses;
};
