// TestTrackingOCV.cpp : Defines the entry point for the console application.

#include "stdafx.h"

#include <regex>
#include <time.h>

#include "Tracker.h"
#include "CameraPoseProviderTXT.h"
#include "TrajectoryArchiver.h"

#define ID_SHIFT 601

typedef std::vector<boost::filesystem::path> vec; // store paths, so we can sort them later
static const std::regex e("^[0-9]+/.[0-9]+/.(bmp|png)$");

void extractNumbers(std::string fOnly, int &prefInt, int& sufInt)
{
  auto sepInd = fOnly.find(".");
  std::string pref = fOnly.substr(0, sepInd);
  std::string suf = fOnly.substr(sepInd + 1, fOnly.size() - sepInd - 1);
  prefInt = std::stoi(pref);
  sufInt = std::stoi(suf);
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
  clock_t tStart = clock();

  boost::filesystem::current_path("/home/dmitry/projects/DynTrack/TestTrackingOCV/bin");

  std::string inFld  = "../../fullTrack/rgb/";
  std::string outFld = "../../debug_tracking/out/";
  std::string outCleanFld = "../../outClean/";
  std::string depthFld    = "../../depth/";
  std::string depthDebFld = "../../depthDebug/";
  std::string trackTypesInfoFld  = "../../tracktypes/";
  //storing tracks online
  std::string lostTracksFld = "../../TD_Data/";
  std::string finalTrackTypesFld = "../../outProc/";

  std::string pathToCameraPoses = "../../cameraPoses";
  std::string pathToSavedTracks = "../../savedTracks";

  CameraPoseProviderTXT poseProvider(pathToCameraPoses);
  TrajectoryArchiver trajArchiver(poseProvider, lostTracksFld);

  boost::filesystem::path p(depthFld);
  boost::filesystem::path p2(inFld);

  vec v, vRgb;
  copy(boost::filesystem::directory_iterator(p), boost::filesystem::directory_iterator(), std::back_inserter(v));
  copy(boost::filesystem::directory_iterator(p2), boost::filesystem::directory_iterator(), std::back_inserter(vRgb));
  sort(v.begin(), v.end());
  sort(vRgb.begin(), vRgb.end());

  int dInd, imgsStartId = 0;
  while (!boost::filesystem::is_regular_file(vRgb[imgsStartId]))
    imgsStartId++;

  dInd = imgsStartId;
  cv::Size imgsSize = cv::imread(vRgb[dInd].string()).size();
  Tracker tracker(trajArchiver, imgsSize, trackTypesInfoFld);

#if 0
  while (dInd < vRgb.size())
  {
    poseProvider.setCurrentFrameNumber(dInd);

    cv::Mat depthImg;
    std::string fName = vRgb[dInd].string();
    std::string fNameOnly = vRgb[dInd].filename().string();
    int prefInt, sufInt;
    int minInd = -1; //depth img index
    long double minPref = 1e100;

    depthImg = cv::Mat::zeros(imgsSize, CV_16S);

    if (v.size() > 0) {
      //TODO: usual 000.png case
      //000.000.png
      if (std::regex_match(fNameOnly.c_str(), e)) {
        extractNumbers(fNameOnly, prefInt, sufInt);
        for (int cInd = 0; cInd < v.size(); cInd++)
        {
          int dPrefInt, dSufInt;
          extractNumbers(v[cInd].filename().string(), dPrefInt, dSufInt);
          long double val = abs(dPrefInt - prefInt)*1e8 + abs(dSufInt - sufInt);
          if (val < minPref)
          {
            minInd = cInd;
            minPref = val;
          }
        }

        depthImg = cv::imread(v[minInd].string(), CV_LOAD_IMAGE_ANYDEPTH);
        cv::imwrite(depthDebFld + "f" + std::to_string(dInd) + ".bmp", 255 / 10 * depthImg / 5000);
      }
    }

    std::cout << fName << std::endl;
    if (boost::filesystem::exists(fName))
    {
      cv::Mat img = cv::imread(fName, 0);
      cv::Mat outputImg;
      cv::cvtColor(img, outputImg, CV_GRAY2BGR);

      std::string fNameOutClean = outCleanFld + std::to_string(dInd) + ".bmp";
      cv::imwrite(fNameOutClean, outputImg);

      tracker.trackWithKLT(img, outputImg, ID_SHIFT + dInd, depthImg);
      //tracker.trackWithOrb(img, outputImg, dInd, depthImg);
      std::string fNameOut = outFld + std::to_string(ID_SHIFT + dInd) + ".bmp";
      cv::imwrite(fNameOut, outputImg);
    }
    std::cout << ID_SHIFT + dInd << " " << tracker.lostTracks.size() << std::endl;
    dInd++;
  }
#endif
  double totalTime = (double)(clock() - tStart) / CLOCKS_PER_SEC;
  //tracker.saveAllTracks(pathToSavedTracks);

#if 0
  std::cerr << "postprocessing stuff\n";

  dInd = imgsStartId;
  while (dInd < vRgb.size())
  {
    poseProvider.setCurrentFrameNumber(dInd);

    cv::Mat depthImg;
    std::string fName = vRgb[dInd].string();
    std::string fNameOnly = vRgb[dInd].filename().string();

    std::cout << fName << std::endl;
    if (boost::filesystem::exists(fName))
    {
      cv::Mat img = cv::imread(fName, 0);
      cv::Mat outputImg;
      cv::cvtColor(img, outputImg, CV_GRAY2BGR);

      tracker.drawFinalPointsTypes(img, outputImg, ID_SHIFT + dInd, depthImg);

      std::string fNameOut = finalTrackTypesFld + std::to_string(ID_SHIFT + dInd) + ".bmp";
      cv::imwrite(fNameOut, outputImg);
    }
    std::cout << ID_SHIFT + dInd << " " << tracker.lostTracks.size() << std::endl;
    dInd++;
  }
#endif

#if 1
  std::cerr << "build tracks\n";
  tracker.loadAllTracks(pathToSavedTracks);

  dInd = imgsStartId;
  while (dInd < vRgb.size())
  {
    poseProvider.setCurrentFrameNumber(dInd);

    std::string fName = vRgb[dInd].string();
    std::cout << fName << std::endl;
    if (boost::filesystem::exists(fName))
    {
      cv::Mat img = cv::imread(fName, 0);
      cv::Mat outputImg;
      cv::cvtColor(img, outputImg, CV_GRAY2BGR);

      tracker.buildTracks(img, outputImg, ID_SHIFT + dInd);

    }
    std::cout << ID_SHIFT + dInd << " " << tracker.lostTracks.size() << std::endl;
    dInd++;
  }
#endif

  fprintf(stderr, "Total time taken: %.2fs\n", totalTime);
  fprintf(stderr, "Average time per frame taken: %.4fs\n", totalTime / vRgb.size());
  fprintf(stderr, "Average fps: %.2fs\n", vRgb.size() / totalTime);
#endif

  std::cout << "Done" << std::endl;

  return 0;
}
