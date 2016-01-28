#pragma once

struct CameraPose {
  cv::Mat R;
  cv::Mat t;
};

class CameraPoseProvider
{
public:
  CameraPoseProvider();
  ~CameraPoseProvider();

  virtual void getCurrentPose(CameraPose& cameraPose); //last from history?
  virtual void getPoseForFrame(CameraPose& cameraPose, int frameNum);

protected:
  CameraPose curCameraPose; // or CameraPose*
  std::vector<std::shared_ptr<CameraPose>> history;

};
