#include "stdafx.h"

#include "CameraPoseProvider.h"
#include <string>

int CameraPoseProvider::getCameraPoseForFrame(CameraPose &cameraPose, int frameId)
{
  return 1;
}

int CameraPoseProvider::getCeresCameraForFrame(double * camera, int frameId)
{
  return 1;
}

int CameraPoseProvider::getProjMatrForFrame(cv::Mat & projMatr, int frameId)
{
  return 1;
}