#pragma once

struct CameraPose {
  CameraPose() { }

  CameraPose(int frameId, cv::Mat const& RData, cv::Mat const& tData) :
  frameId(frameId) {
    RData.copyTo(R);
    tData.copyTo(t);

    /*std::cerr << "frame id " << frameId << std::endl;
    std::cerr << "R " << R << std::endl;
    std::cerr << "t " << t << std::endl;*/
  }

  int frameId;
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
