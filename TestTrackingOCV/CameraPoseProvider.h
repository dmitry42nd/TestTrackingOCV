#pragma once
class CameraPoseProvider
{
public:
  CameraPoseProvider();
  ~CameraPoseProvider();

  void getCurrentPose(cv::Mat &R, cv::Mat &t);
};

