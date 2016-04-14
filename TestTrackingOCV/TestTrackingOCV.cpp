// TestTrackingOCV.cpp : Defines the entry point for the console application.

#include "stdafx.h"

#include <regex>
#include <time.h>

#include "Tracker.h"
#include "CameraPoseProviderTXT.h"
#include "TrajectoryArchiver.h"
#include "DynamicTrajectoryEstimator.h"

#define ID_SHIFT 601

typedef boost::filesystem::path ImgPath;

static const std::regex e("^[0-9]+/.[0-9]+/.(bmp|png)$");

void extractNumbers(std::string fOnly, int &prefInt, int& sufInt)
{
  auto sepInd = fOnly.find(".");
  std::string pref = fOnly.substr(0, sepInd);
  std::string suf = fOnly.substr(sepInd + 1, fOnly.size() - sepInd - 1);
  prefInt = std::stoi(pref);
  sufInt = std::stoi(suf);
}


void getDepthImg(cv::Mat &depthImg, std::vector<ImgPath> const &depthImgsPaths, ImgPath const &rgbImgPath)
{
  std::string fName = rgbImgPath.filename().string();
  int prefInt, sufInt;
  int minInd = -1; //depth img index
  long double minPref = 1e100;

  if (depthImgsPaths.size() > 0)
  {
    //TODO: usual 000.png case
    //000.000.png
    if (std::regex_match(fName.c_str(), e))
    {
      extractNumbers(fName, prefInt, sufInt);
      for (decltype(depthImgsPaths.size()) cInd = 0; cInd < depthImgsPaths.size(); cInd++)
      {
        int dPrefInt, dSufInt;
        extractNumbers(depthImgsPaths[cInd].filename().string(), dPrefInt, dSufInt);
        long double val = abs(dPrefInt - prefInt)*1e8 + abs(dSufInt - sufInt);
        if (val < minPref)
        {
          minInd = cInd;
          minPref = val;
        }
      }
      depthImg = cv::imread(depthImgsPaths[minInd].string(), CV_LOAD_IMAGE_ANYDEPTH);
    }
  }
}

int main()
{
#if 0
  int size = 13;
  int sample_size = 2;
  int step = 13 / 2;

  int i;
  for(i = 3; i < size; i+=step)
  {
    std::cerr << i << std::endl;

  }
  //last frame must have?
  if(i != size-1) std::cerr << size-1 << std::endl;
#else
  cv::FileStorage fs("settings.yaml", cv::FileStorage::READ);

  std::string rootFld            = fs["root"];
  std::string inFld              = fs["inFld"];
  std::string outFld             = fs["outFld"];
  std::string depthFld           = fs["depthFld"];
  std::string trackTypesInfoFld  = fs["trackTypesInfoFld"];
  //std::string lostTracksFld      = fs["lostTracksFld"];
  std::string finalTrackTypesFld = fs["finalTrackTypesFld"];

  std::string pathToCameraPoses = fs["pathToCameraPoses"];
  std::string pathToSavedTracks = fs["pathToSavedTracks"];

  boost::filesystem::current_path(rootFld);

  boost::filesystem::path pathToDepthFld(depthFld);
  boost::filesystem::path pathToInFld(inFld);

  std::vector<ImgPath> depthImgsPaths, rgbImgsPaths;
  copy(boost::filesystem::directory_iterator(pathToDepthFld), boost::filesystem::directory_iterator(), std::back_inserter(depthImgsPaths));
  copy(boost::filesystem::directory_iterator(pathToInFld), boost::filesystem::directory_iterator(), std::back_inserter(rgbImgsPaths));
  sort(rgbImgsPaths.begin(), rgbImgsPaths.end());
  sort(depthImgsPaths.begin(), depthImgsPaths.end());

  CameraPoseProviderTXT poseProvider(pathToCameraPoses);

#if 0
  cv::Size imgSize = cv::imread(rgbImgsPaths.front().string()).size();

  TrajectoryArchiver trajArchiver(pathToSavedTracks);
  Tracker tracker(trajArchiver, poseProvider, imgSize);

  clock_t tStart = clock();

  for (decltype(rgbImgsPaths.size())  imgId = 0; imgId < rgbImgsPaths.size(); imgId++)
  {
    cv::Mat depthImg = cv::Mat::zeros(imgSize, CV_16S);
    getDepthImg(depthImg, depthImgsPaths, rgbImgsPaths[imgId]);

    std::string rgbImgName = rgbImgsPaths[imgId].string();
    if (boost::filesystem::exists(rgbImgName))
    {
      cv::Mat img = cv::imread(rgbImgName, 0);
      cv::Mat outImg;
      cv::cvtColor(img, outImg, CV_GRAY2BGR);

      tracker.trackWithKLT(ID_SHIFT + imgId, img, outImg, depthImg);

      std::string outImgName = outFld + std::to_string(ID_SHIFT + imgId) + ".bmp";
      cv::imwrite(outImgName, outImg);
    }
    std::cerr << ID_SHIFT + imgId << std::endl;
  }

  double totalTime = (double)(clock() - tStart) / CLOCKS_PER_SEC;
  fprintf(stderr, "Total time taken: %.2fs\n", totalTime);
  fprintf(stderr, "Average time per frame taken: %.4fs\n", totalTime / rgbImgsPaths.size());
  fprintf(stderr, "Average fps: %.2fs\n", rgbImgsPaths.size() / totalTime);


  std::cerr << "final tracks types\n";
  for (decltype(rgbImgsPaths.size()) imgId = 0; imgId < rgbImgsPaths.size(); imgId++)
  {
    std::string rgbImgName = rgbImgsPaths[imgId].string();
    if (boost::filesystem::exists(rgbImgName))
    {
      cv::Mat img = cv::imread(rgbImgName, 0);
      cv::Mat outImg;
      cv::cvtColor(img, outImg, CV_GRAY2BGR);

      tracker.drawFinalPointsTypes(ID_SHIFT + imgId, img, outImg);

      std::string outFTTImgName = finalTrackTypesFld + std::to_string(ID_SHIFT + imgId) + ".bmp";
      cv::imwrite(outFTTImgName, outImg);
    }
    std::cerr << ID_SHIFT + imgId << std::endl;
  }
#endif

#if 1
  std::cerr << "build dynamic tracks\n";

  std::vector<ImgPath> fImgsPaths;
  copy(boost::filesystem::directory_iterator(finalTrackTypesFld), boost::filesystem::directory_iterator(), std::back_inserter(fImgsPaths));
  sort(fImgsPaths.begin(), fImgsPaths.end());

  DynamicTrajectoryEstimator DTE(poseProvider);
  DTE.loadOnlyDynamicsTracksFromFile(pathToSavedTracks);
  //DTE.buildTrack(620, 639);
  //DTE.buildTrack(660, 711);
  //DTE.buildTrack(725, 765);
  //DTE.buildTrack(788, 830);
  DTE.buildTrack(863, 895);

  /*for (int imgId = 0; ID_SHIFT + imgId < 654; imgId++)
  {
    std::string rgbImgName = fImgsPaths[imgId].string();
    if (boost::filesystem::exists(rgbImgName))
    {
      cv::Mat img = cv::imread(rgbImgName, 0);
      cv::Mat outImg;
      cv::cvtColor(img, outImg, CV_GRAY2BGR);

      std::string outFTTImgName = finalTrackTypesFld + std::to_string(ID_SHIFT + imgId) + ".bmp";
      cv::imwrite(outFTTImgName, outImg);
    }
    std::cout << ID_SHIFT + imgId << std::endl;
  }*/
#endif


#endif

  std::cerr << "Done" << std::endl;

  return 0;
}
