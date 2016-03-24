#include "stdafx.h"

#include "Tracker.h"
#include "TriangulateError.h"

#include "opencv2/video/tracking.hpp"

Tracker::Tracker(TrajectoryArchiver &trajArchiver, cv::Size imSize) :
  trajArchiver(trajArchiver),
  imSize(imSize)
{
  mcnt = 0;
  m_tracksFrame = cv::Mat::zeros(imSize, CV_8UC3);
  orb = new cv::ORB(1000, 1.2, 8, 31, 0, 2, cv::ORB::FAST_SCORE, 31);
  m_orbMatcher = new cv::BFMatcher(cv::NORM_HAMMING);

  fastDetector = new cv::FastFeatureDetector(10);
  wx = 2;
  wy = 2;

  for (int j = 0; j < wy; j++)
  {
    detMasks.push_back(std::vector<cv::Mat>());
    for (int i = 0; i < wx; i++)
    {
      detMasks[j].push_back(cv::Mat());
    }
  }
  int sx = imSize.width / wx;
  int sy = imSize.height / wy;
  for (int i = 0; i < wx; i++)
  {
    for (int j = 0; j < wy; j++)
    {
      cv::Mat mask = cv::Mat::zeros(imSize, CV_8U);
      for (int px = 0; px < imSize.width; px++)
      {
        for (int py = 0; py < imSize.height; py++)
        {
          if (px / sx == i && py / sy == j)
          {
            mask.at<uchar>(py, px) = 255;
          }
        }
      }
      detMasks[j][i] = mask;
      //cv::imwrite("C:\\projects\\debug_tracking\\mask.bmp", mask);
    }
  }
}

Tracker::Tracker(TrajectoryArchiver & trajArchiver, cv::Size imSize, std::string pathToTrackTypes) : Tracker(trajArchiver, imSize)
{
  Tracker::pathToTrackTypes = pathToTrackTypes;
}

void Tracker::createNewTrack(cv::Point2f point, int frameInd, cv::KeyPoint const & keyPt, cv::Mat const & desc, double depth)
{
  std::shared_ptr<Track> newTrack(std::make_shared<Track>());
  newTrack->bestCandidate = std::make_shared<TrackedPoint>(point, frameInd, MAX_DISTANCE, keyPt, desc, depth);
  newTrack->history.push_back(newTrack->bestCandidate);
  curTracks.push_back(newTrack);
}

void roundCoords(int &px, int &py, cv::Point2f const& pt, cv::Mat const& img)
{
  px = round(pt.x);
  px = std::min({ img.cols - 1, px });
  px = std::max({ 0, px });
  py = round(pt.y);
  py = std::min({ img.rows - 1, py });
  py = std::max({ 0, py });
}

cv::Point2f fillDepthPt(cv::Point2f& ptIm, double ccx, double ccy, double cfx, double cfy, double dcx, double dcy, double dfx, double dfy)
{
  cv::Point2f pt;
  pt.x = (ptIm.x - ccx) / cfx;
  pt.y = (ptIm.y - ccy) / cfy;
  pt.x = dfx*pt.x + dcx;
  pt.y = dfy*pt.y + dcy;
  return pt;
}

cv::Mat Tracker::calcStatsByQuadrant(int wx, int wy, int ptNum, std::vector<std::shared_ptr<Track>> const& curTracks)
{
  //num of points in each quadrant
  cv::Mat res = cv::Mat::zeros(wy, wx, CV_32F);
  int sx = imSize.width / wx + 1;
  int sy = imSize.height / wy + 1;
  //count points in each quadrant
  for (auto const& c : curTracks)
  {
    int resX = c->bestCandidate->location.x / sx;
    int resY = c->bestCandidate->location.y / sy;
    res.at<float>(resY, resX) += 1;
  }
  cv::Mat boolRes = cv::Mat::zeros(wy, wx, CV_8U);
  //check if less than threshold ptNum
  for (int i = 0; i < wx; i++)
  {
    for (int j = 0; j < wy; j++)
    {
      if (res.at<float>(j, i) < ptNum)
      {
        boolRes.at<uchar>(j, i) = 1;
      }
    }
  }
  return boolRes;
}

std::vector<cv::KeyPoint> Tracker::filterPoints(int wx, int wy, std::vector<cv::KeyPoint>& keyPts) {

  typedef std::pair<int, int> Coords;

  std::map<Coords, cv::KeyPoint> curKeyPtsMap;
  for (auto const& track : curTracks)
  {
    int cx = track->bestCandidate->location.x / wx;
    int cy = track->bestCandidate->location.y / wy;

    curKeyPtsMap[Coords(cx, cy)] = track->bestCandidate->keyPt;
  }

  std::map<Coords, cv::KeyPoint> keyPtsMap;
  for (auto const& keyPt : keyPts)
  {
    int cx = keyPt.pt.x / wx;
    int cy = keyPt.pt.y / wy;

    if (!curKeyPtsMap.count(Coords(cx, cy)) && keyPt.response > keyPtsMap[Coords(cx, cy)].response)
    {
      keyPtsMap[Coords(cx, cy)] = keyPt;
    }
  }

  //copy map to vector as needed
  std::vector<cv::KeyPoint> keyPtsNew;
  for (auto const& keyPt : keyPtsMap) {
    keyPtsNew.push_back(keyPt.second);
  }

  return keyPtsNew;
}

void Tracker::detectPoints(int indX, int indY, cv::Mat& m_nextImg, cv::Mat& depthImg, cv::Mat& outputFrame, int frameInd) {

  std::vector<cv::KeyPoint> keyPts;
  std::cout << " detecting.. " << indX << " " << indY << std::endl;
  fastDetector->detect(m_nextImg, keyPts, detMasks[indY][indX]);
  int keyPtsSize = keyPts.size();

  if (keyPtsSize > 0)
  {
    //draw all FAST points (blue)
    /*
    for (auto const& kp : keyPts)
    {
      cv::circle(outputFrame, kp.pt, 3, cv::Scalar(255, 0, 0));
    }*/

    //get 80% of best by reduction points
    cv::KeyPointsFilter::retainBest(keyPts, keyPts.size() * 6 / 10);
    //draw all filtered points (yellow)
    /*for (auto const& kp : keyPts)
    {
      cv::circle(outputFrame, kp.pt, 3, cv::Scalar(0, 255, 255));
    }*/

    //TODO: magic numbers 16, 16
    std::vector<cv::KeyPoint> keyPtsFiltered = filterPoints(16, 16, keyPts);

    auto repProc = keyPtsFiltered.size() * 100 / keyPtsSize;
    std::cout << "key point reduction: " << repProc << " % " << keyPtsSize << ":" << keyPtsFiltered.size() << "\n";

    //draw final filtered points (red) and some stuff
    for (int i = 0; i < keyPtsFiltered.size(); i++)
    {
      int px = 0, py = 0;

      cv::Point2f pt = fillDepthPt(keyPtsFiltered[i].pt, ccx, ccy, cfx, cfy, dcx, dcy, dfx, dfy);
      roundCoords(px, py, pt, m_nextImg);
      double depthVal = (int)(depthImg.at<ushort>(py, px) / 5000.0);
      createNewTrack(keyPtsFiltered[i].pt, frameInd, keyPtsFiltered[i], cv::Mat(), depthVal);

      cv::circle(outputFrame, keyPtsFiltered[i].pt, 3, cv::Scalar(0, 0, 255));
    }
  }
}

#if 1
void undistPoint(cv::Point2f const& point, cv::Mat const& K, cv::Mat const& dist, cv::Point2d & undist) {
  cv::Mat projPoint(1, 1, CV_64FC2);
  projPoint.at<cv::Vec2d>(0, 0)[0] = point.x;
  projPoint.at<cv::Vec2d>(0, 0)[1] = point.y;

  cv::Mat undistPoint = cv::Mat(1,1,CV_64FC2);
  cv::undistortPoints(projPoint, undistPoint, K, dist);

  undist.x = undistPoint.at<cv::Vec2d>(0,0)[0];
  undist.y = undistPoint.at<cv::Vec2d>(0,0)[1];
}

void printCamera(double * camera, int id) {
  std::cout << "camera" << id << std::endl;
  for(int i = 0; i < 6; i++) {
    std::cout << camera[i] << " ";
  }
  std::cout << std::endl;
}

//ceres-solver
void Tracker::defineTrackType(std::shared_ptr<Track> track, double errThr) {
  //kinect
  cv::Mat K = cv::Mat::zeros(3, 3, CV_64F);
  K.at<double>(0, 0) = 522.97697;
  K.at<double>(0, 2) = 318.47217;
  K.at<double>(1, 1) = 522.58746;
  K.at<double>(1, 2) = 256.49968;
  K.at<double>(2, 2) = 1.0;
  cv::Mat_<double> dist(1,4);
  dist << 0.18962,-0.38214, 0, 0;

  double *point = new double[3];
  point[0] = 0;
  point[1] = 0;
  point[2] = 0;

  if (track->history.size() > 3) {
    ceres::Problem problem;

    std::shared_ptr<TrackedPoint> firstPoint, lastPoint;
    CameraPose cp1, cp2;
    int pfId;
    for (pfId = 0; pfId < track->history.size(); pfId++)
    {
      auto pfFrameId = track->history[pfId]->frameId;
      trajArchiver.poseProvider.getPoseForFrame(cp1, pfFrameId);
      if (cv::countNonZero(cp1.R) && cv::countNonZero(cp1.t))
        break;
    }
    firstPoint  = track->history[pfId];
    lastPoint   = track->history.back();
    trajArchiver.poseProvider.getPoseForFrame(cp2, lastPoint->frameId);


    double *camera1 = new double[6];
    //get R t
    cv::Mat_<double> Rv1;
    cv::Rodrigues(cp1.R, Rv1);

    for (auto i = 0; i < 3; i++) {
      camera1[i] = Rv1.at<double>(i, 0);
    }
    for (auto i = 0; i < 3; i++) {
      camera1[i+3] = cp1.t.at<double>(i, 0);
    }
    printCamera(camera1, 1);
    cv::Point2d unPoint1;
    undistPoint(firstPoint->location, K, dist, unPoint1);


    ceres::CostFunction* cost_function1 = TriangulateError::Create(unPoint1.x, unPoint1.y, camera1);
    problem.AddResidualBlock(cost_function1, NULL, point);


    double *camera2 = new double[6];
    //get R t
    cv::Mat_<double> Rv2;
    cv::Rodrigues(cp2.R, Rv2);

    for (auto i = 0; i < 3; i++) {
      camera2[i] = Rv2.at<double>(i, 0);
    }
    for (auto i = 0; i < 3; i++) {
      camera2[i+3] = cp2.t.at<double>(i, 0);
    }
    printCamera(camera2, 2);

    cv::Point2d unPoint2;
    undistPoint(lastPoint->location, K, dist, unPoint2);

    ceres::CostFunction* cost_function2 = TriangulateError::Create(unPoint2.x, unPoint2.y, camera2);
    problem.AddResidualBlock(cost_function2, NULL, point);


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    std::cout << "final p: " <<  point[0] << " " <<  point[1] << " " <<  point[2]<< "\n";
    printCamera(camera1, 1);
    printCamera(camera2, 2);

    double p[3];
    p[0] = 0; p[1] = 0; p[2] = 0;
    ceres::AngleAxisRotatePoint(camera1, point, p);
    p[0] += camera1[3]; p[1] += camera1[4]; p[2] += camera1[5];
    double xp = - p[0] / p[2];
    double yp = - p[1] / p[2];

    std::cout << "observed_norm1: " <<  unPoint1.x << " " <<  unPoint1.y << "\n";
    std::cout << "predicted_norm: " <<  xp << " " <<  yp << "\n";


    //p[0] = 0; p[1] = 0; p[2] = 0;
    ceres::AngleAxisRotatePoint(camera2, point, p);
    p[0] += camera2[3]; p[1] += camera2[4]; p[2] += camera2[5];
    xp = - p[0] / p[2];
    yp = - p[1] / p[2];
    std::cout << "observed_norm2: " <<  unPoint2.x << " " <<  unPoint2.y << "\n";
    std::cout << "predicted_norm: " <<  xp << " " <<  yp << "\n";

    delete[] point;
    delete[] camera1;
    delete[] camera2;
    /*
    std::cerr << "actual0: " <<  double(track->bestCandidate->location.x) << " " <<  double(track->bestCandidate->location.y) << "\n";
    std::cerr << "predict: " <<  predicted_x << " " <<  predicted_y << "\n";
    std::cerr << "res    : " <<  residuals[0] << " " <<  residuals[1] << "\n";
    */

  }
}
#endif


void checkError(std::ofstream& trackOut, std::vector<std::shared_ptr<Track>> curTracks,
                int const frameInd, cv::Mat const & mask, int errThr)
{
  int  FN = 0; int  TN = 0; int  TP = 0; int  FP = 0; int  U = 0;
  long double FPErr = 0;
  long double FNErr = 0;
  long double TPErr = 0;
  long double TNErr = 0;

  for (auto t : curTracks) {
    cv::Point2f p = t->bestCandidate->location;
    //dyn if 255
    bool dyn = mask.at<uchar>(trunc(p.y), trunc(p.x));

    if (!dyn && t->type == Static)  {
      TP++;
      TPErr += (t->err[0] + t->err[1])/2;
    } else if ( dyn && t->type == Dynamic) {
      TN++;
      TNErr += (t->err[0] + t->err[1])/2;
    } else if ( dyn && t->type == Static)  {
      FP++;
      FPErr += (t->err[0] + t->err[1])/2;
    } else if (!dyn && t->type == Dynamic) {
      FN++;
      FNErr += (t->err[0] + t->err[1])/2;
    } else if (t->type == Undef) {
      U++;
    }

    /*trackOut << "x\t y\t type(algorithm)\t type(mask) err1\t err2\t ang c median/10\n";
    trackOut << t->bestCandidate->location.x << "\t ";
    trackOut << t->bestCandidate->location.y << "\t ";
    trackOut << (t->type == Static ? "Static" : (t->type == Dynamic ? "Dynamic" : "Undef")) << "\t ";
    trackOut << (!dyn ? "Static" : "Dynamic") << "\t ";
    trackOut << t->err1 << "\t ";
    trackOut << t->err2 << "\t ";
    trackOut << t->angle << "\t ";
    trackOut << t->c << "\t ";
    trackOut << t->median;
    trackOut << std::endl;*/
  }
  /*trackOut << "errThr: " << i << std::endl;
  trackOut << "mask static  \ algo static  (TP): " << TP << std::endl;
  trackOut << "mask dynamic \ algo dynamic (TN): " << TN << std::endl;
  trackOut << "mask dynamic \ algo static  (FP): " << FP << std::endl;
  trackOut << "mask static  \ algo dynamic (FN): " << FN << std::endl;
  trackOut << "Undefinded                   (U): " << U << std::endl;*/

  std::cerr << "errThr: " << errThr << std::endl;
  std::cerr << "mask static  / algo static  (TP): " << TP << std::endl;
  std::cerr << "mask dynamic / algo dynamic (TN): " << TN << std::endl;
  std::cerr << "mask dynamic / algo static  (FP): " << FP << std::endl;
  std::cerr << "mask static  / algo dynamic (FN): " << FN << std::endl;
  std::cerr << "Undefinded                   (U): " << U << std::endl;
  std::cerr << "TP projective error             : " << TPErr/TP << std::endl;
  std::cerr << "TN projective error             : " << TNErr/TN << std::endl;
  std::cerr << "FP projective error             : " << FPErr/FP << std::endl;
  std::cerr << "FN projective error             : " << FNErr/FN << std::endl;
  std::cerr << std::endl;

  float TPR = TP / (float)(TP + FN);
  float FPR = FP / (float)(TN + FP);
  trackOut << FPR << ", " << TPR  << std::endl;
}
char imgn[100];

void Tracker::trackWithKLT(cv::Mat& m_nextImg, cv::Mat& outputFrame, int frameInd, cv::Mat& depthImg) {
  curTracks.clear();

  cv::cvtColor(m_nextImg, outputFrame, CV_GRAY2BGR);

  if (prevTracks.size() > 0)
  {
    sprintf(imgn, "../../dynmasks/%06d.png", frameInd);
    cv::Mat mask = cv::imread(imgn, CV_8U);

    std::vector<cv::Point2f> prevCorners;
    for (auto &p : prevTracks)
    {
      prevCorners.push_back(p->history.back()->location);
    }

    std::vector<cv::Point2f> nextCorners;
    std::vector<uchar> status;
    std::vector<float> err;
    double minEigThreshold = 1e-2;
    cv::calcOpticalFlowPyrLK(prevImg, m_nextImg, prevCorners, nextCorners, status, err,
      cv::Size(11, 11), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
      cv::OPTFLOW_LK_GET_MIN_EIGENVALS, minEigThreshold);

    for (size_t i = 0; i < prevTracks.size(); i++)
    {
      cv::Mat err = cv::Mat(nextCorners[i] - prevCorners[i]);
      double trackDist = norm(err);

      if (trackDist < trackThr && status[i] && nextCorners[i].x >= 0 && nextCorners[i].x < m_nextImg.cols &&
        nextCorners[i].y >= 0 && nextCorners[i].y < m_nextImg.rows)
      {
        //std::cout << "track depth read " << round(nextCorners[i].y) << " " << round(nextCorners[i].x) << std::endl;
        //std::cout << nextCorners[i] << std::endl;
        cv::Point2f pt = fillDepthPt(nextCorners[i], ccx, ccy, cfx, cfy, ccx, ccy, cfx, cfy);
        int px, py;

        //std::cout << pt<< std::endl;
        roundCoords(px, py, pt, m_nextImg);
        double depthVal = 0;
        if (pt.x > 0 && pt.y > 0 && pt.x < depthImg.cols && pt.y < depthImg.rows)
        {
          depthVal = (int)depthImg.at<ushort>(py, px);
          depthVal /= 5000.0;
        }

        prevTracks[i]->history.push_back(std::make_shared<TrackedPoint>(nextCorners[i], frameInd, 0, cv::KeyPoint(), cv::Mat(), depthVal));
        prevTracks[i]->bestCandidate = prevTracks[i]->history.back();

#if 1
        cv::circle(outputFrame, prevCorners[i], 5, cv::Scalar(250, 0, 250), -1);
        cv::line(outputFrame,   prevCorners[i], nextCorners[i], cv::Scalar(0, 250, 0));
        cv::circle(outputFrame, nextCorners[i], 2, cv::Scalar(0, 250, 0), -1);
#endif

        curTracks.push_back(prevTracks[i]);
      }
      else
      {
        if (prevTracks[i]->history.size() > 3)
        {
          lostTracks.push_back(prevTracks[i]);
          trajArchiver.archiveTrajectorySimple(prevTracks[i]);
        }
      }
    }

    if (!mask.empty()) {
      for (auto angFact = 10; angFact < 11; angFact += 4) {
        sprintf(imgn, "%stt%d-max-angFact%d.txt", pathToTrackTypes.c_str(), frameInd, 10);
        std::ofstream trackOut(imgn);
        for (auto errThr = 0; errThr < 200; errThr += 5) {
          for (auto track : curTracks) {
            defineTrackType(track, errThr);
          }
          checkError(trackOut, curTracks, frameInd, mask, errThr);
        }
      }

#if 1
      for (size_t i = 0; i < curTracks.size(); i++)
      {
        cv::Scalar color;
        if (curTracks[i]->type == Static)
          color = cv::Scalar(200, 200, 200);
        else if (curTracks[i]->type == Dynamic)
          color = cv::Scalar(0, 200, 0);
        else //undef
          color = cv::Scalar(200, 0, 200);

        cv::circle(outputFrame, prevCorners[i], 5, color, -1);
        cv::line(outputFrame, prevCorners[i], nextCorners[i], cv::Scalar(0, 250, 0));
        cv::circle(outputFrame, nextCorners[i], 2, cv::Scalar(0, 250, 0), -1);
      }
#endif
    }

  }

  cv::Mat boolStats = calcStatsByQuadrant(wx, wy, kltPtThr / (wx *wy), curTracks);

  for (int i = 0; i < wx; i++)
  {
    for (int j = 0; j < wy; j++)
    {
      if (boolStats.at<uchar>(j, i) > 0)
      {
        detectPoints(i, j, m_nextImg, depthImg, outputFrame, frameInd);
      }
    }
  }

  if (curTracks.size() < kltPtThr) {}

  prevTracks = curTracks;
  prevImg = m_nextImg;
}

#if 0
void Tracker::trackWithOrb(cv::Mat& m_nextImg, cv::Mat& outputFrame, int frameInd, cv::Mat& depthImg)
{
  //orb->detect(m_nextImg, m_nextKeypoints);
  fastDetector->detect(m_nextImg, m_nextKeypoints);
  int keyPtsSize = m_nextKeypoints.size();

  if (keyPtsSize > 0)
  {
    //get 80% of best by reduction points
    cv::KeyPointsFilter::retainBest(m_nextKeypoints, m_nextKeypoints.size() * 8 / 10);

    //TODO: magic numbers 16, 16
    std::vector<cv::KeyPoint> keyPtsFiltered = filterPoints(8, 8, m_nextKeypoints);

    auto repProc = keyPtsFiltered.size() * 100 / keyPtsSize;
    std::cout << "key point reduction: " << repProc << " % " << keyPtsSize << ":" << keyPtsFiltered.size() << "\n";
    keyPtsFiltered.swap(m_nextKeypoints);
  }

  orb->compute(m_nextImg, m_nextKeypoints, m_nextDescriptors);

  if (m_prevKeypoints.size() > 0)
  {
    curTracks.swap(prevTracks);
    curTracks.clear();

    std::vector<std::vector<cv::DMatch>> matches;
    m_orbMatcher->radiusMatch(m_nextDescriptors, m_prevDescriptors, matches, MAX_DISTANCE);
    for (size_t i = 0; i < matches.size(); i++)
    {
      if (matches[i].size() == 0)
      {
        createNewTrack(m_nextKeypoints[i].pt, frameInd, m_nextKeypoints[i], m_nextDescriptors.row(i));
        continue;
      }

      int prevTrackId = matches[i][0].trainIdx;
      int nextTrackId = matches[i][0].queryIdx;

      cv::Point prevPt = m_prevKeypoints[prevTrackId].pt;
      cv::Point nextPt = m_nextKeypoints[nextTrackId].pt;

      cv::Mat err = cv::Mat(nextPt - prevPt);
      double trackDist = norm(err);
      if (trackDist < trackThr && matches[i][0].distance < prevTracks[prevTrackId]->bestCandidate->matchScore)
      {
        prevTracks[prevTrackId]->bestCandidate = std::make_shared<TrackedPoint>(nextPt, frameInd,
          matches[i][0].distance, m_nextKeypoints[nextTrackId], m_nextDescriptors.row(nextTrackId));
      }
    }

    for (int i = 0; i < prevTracks.size(); i++)
    {
      if (prevTracks[i]->bestCandidate->frameId == frameInd)
      {
        prevTracks[i]->history.push_back(prevTracks[i]->bestCandidate);
        curTracks.push_back(prevTracks[i]);

        cv::circle(outputFrame, prevTracks[i]->history[prevTracks[i]->history.size() - 2]->keyPt.pt, 3, cv::Scalar(250, 0, 250), -1);
       // cv::line(outputFrame, prevTracks[i]->history[prevTracks[i]->history.size() - 2]->keyPt.pt,
       //   prevTracks[i]->history.back()->keyPt.pt, cv::Scalar(0, 250, 0));

        cv::line(m_tracksFrame, prevTracks[i]->history[prevTracks[i]->history.size() - 2]->keyPt.pt,
          prevTracks[i]->history.back()->keyPt.pt, cv::Scalar(0, 250, 0));
        outputFrame += m_tracksFrame;

        cv::circle(outputFrame, prevTracks[i]->history.back()->keyPt.pt, 1, cv::Scalar(0, 250, 0), -1);
      }
      else if (frameInd - prevTracks[i]->bestCandidate->frameId < 4)
      {
        curTracks.push_back(prevTracks[i]);
      }
      else //totaly lost
      {
        if (prevTracks[i]->history.size() > 3)
        {
          lostTracks.push_back(prevTracks[i]);
          trajArchiver.archiveTrajectorySimple(prevTracks[i]);
        }
      }
    }
  }
  else
  {
    for (int i = 0; i < m_nextKeypoints.size(); i++)
    {
      createNewTrack(m_nextKeypoints[i].pt, frameInd, m_nextKeypoints[i], m_nextDescriptors.row(i));
    }
  }

  m_prevDescriptors = cv::Mat(curTracks.size(), m_nextDescriptors.cols, m_nextDescriptors.type());
  m_prevKeypoints.clear();
  for (int i = 0; i < curTracks.size(); i++)
  {
    curTracks[i]->bestCandidate->desc.copyTo(m_prevDescriptors.row(i));
    m_prevKeypoints.push_back(curTracks[i]->bestCandidate->keyPt);
  }
}
#endif

void Tracker::saveAllTracks(std::string& pathToSaveFolder)
{
  for (int i = 0; i < prevTracks.size(); i++)
  {
    lostTracks.push_back(prevTracks[i]);
  }
  std::cout << "final size " << lostTracks.size() << std::endl;
  for (int i = 0; i < lostTracks.size(); i++)
  {
    std::ofstream outTrackSave(pathToSaveFolder + std::to_string(i) + ".txt");
    std::shared_ptr<Track> curTrack = lostTracks[i];
    for (int hInd = 0; hInd < curTrack->history.size(); hInd++)
    {
      outTrackSave << curTrack->history[hInd]->frameId << " " << curTrack->history[hInd]->location.x << " " << curTrack->history[hInd]->location.y << " " << curTrack->history[hInd]->depth << std::endl;
    }
  }
}