#include "stdafx.h"

#include "Tracker.h"

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

void Tracker::createNewTrack(cv::Point2f point, int frameInd, cv::KeyPoint const & keyPt, cv::Mat const & desc, double depth)
{
  std::shared_ptr<Track> newTrack(std::make_shared<Track>());
  newTrack->bestCandidate = std::make_shared<TrackedPoint>(point, frameInd, MAX_DISTANCE, keyPt, desc, depth);
  newTrack->history.push_back(newTrack->bestCandidate);
  curTracks.push_back(newTrack);
  curKeyPts.push_back(newTrack->bestCandidate->keyPt);
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
  std::map<Coords, cv::KeyPoint> keyPtsMap;

  for (auto const& keyPt : keyPts)
  {
    int cx = keyPt.pt.x / wx;
    int cy = keyPt.pt.y / wy;

    if (keyPt.response > keyPtsMap[Coords(cx, cy)].response)
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
    keyPts.insert(keyPts.begin(), curKeyPts.begin(), curKeyPts.end());
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

#define PI 3.14159265

void Tracker::defineTrackType(std::shared_ptr<Track> & track) {
  //const intrinsic camera matrix
  cv::Mat K = cv::Mat::zeros(3, 3, CV_64F);
  K.at<double>(0, 0) = 457.62253;
  K.at<double>(0, 2) = 326.11843;
  K.at<double>(1, 1) = 457.62253;
  K.at<double>(1, 2) = 177.17820;
  K.at<double>(2, 2) = 1.0;

  if (!curFrameProjMatr.empty() && track->history.size() > 3) {
    std::shared_ptr<TrackedPoint> firstPoint, lastPoint;

    //search first track point with defined frame's R, t 
    int pfId;
    for (pfId = 0; pfId < track->history.size(); pfId++) {
      auto pfFrameId = track->history[pfId]->frameId;
      if (trajArchiver.poseProvider.poses.count(pfFrameId))
        break;
    }

    firstPoint = track->history[pfId];
    lastPoint  = track->history.back();

    if (lastPoint->frameId > firstPoint->frameId) {
      cv::Mat projMatr1 = trajArchiver.poseProvider.poses[firstPoint->frameId];
      cv::Mat projPoint1(1, 1, CV_64FC2);
      projPoint1.at<cv::Vec3d>(0, 0)[0] = firstPoint->location.x;
      projPoint1.at<cv::Vec3d>(0, 0)[1] = firstPoint->location.y;
      //std::cerr << "First track point. FrameId " << firstPoint->frameId << ":" << firstPoint->location << std::endl;

      cv::Mat projPoint2(1, 1, CV_64FC2);
      projPoint2.at<cv::Vec3d>(0, 0)[0] = lastPoint->location.x;
      projPoint2.at<cv::Vec3d>(0, 0)[1] = lastPoint->location.y;
      //std::cerr << "Last track point. FrameId " << lastPoint->frameId << ":" << lastPoint->location << std::endl;

      cv::Mat point4D;
      cv::triangulatePoints(K*projMatr1, K*curFrameProjMatr, projPoint1, projPoint2, point4D);

      //std::cerr << "4D " << point4D << std::endl;
      //std::cerr << "projMatr1 " << projMatr1 << std::endl;
      cv::Mat pr1 = projMatr1*point4D;
      cv::Mat pr2 = curFrameProjMatr*point4D;

      double cosa = norm(pr1.t()*pr2) / (norm(pr1)*norm(pr2));
      double ang = acos(cosa) * 180.0 / PI;
//      std::cerr << "angle " << ang << std::endl;

      double a = norm(pr1);
      double b = norm(pr2);
      double c = norm(pr1 - pr2);
      double median = pow(2 * pow(a, 2) + 2 * pow(b, 2) - pow(c, 2), 0.5) / 2;
      //std::cerr << "median " << median << std::endl;
      //std::cerr << "c " << c  << " " << 10*c << std::endl;

      if (ang > 2 && 10*c > median) {
        const cv::Rect roi = cv::Rect(0, 0, 1, 2);
        cv::Mat pr1_ = K*projMatr1*point4D;
        cv::Mat pr2_ = K*curFrameProjMatr*point4D;
        //std::cerr << "projected " << pr1_  << " " << pr2_ << std::endl;
        //std::cerr << "projected " << pr1_ / pr1_.at<double>(2, 0) << " " << pr2_ / pr2_.at<double>(2, 0) << std::endl;

        cv::Mat cpr1 = (pr1_ / pr1_.at<double>(2, 0))(roi);
        cv::Mat cpr2 = (pr2_ / pr2_.at<double>(2, 0))(roi);
        cv::Mat_<double> q1 = cv::Mat(firstPoint->location);
        cv::Mat_<double> q2 = cv::Mat(lastPoint->location);

        double pointErr1 = norm(q1 - cpr1);
        double pointErr2 = norm(q2 - cpr2);
        //std::cerr << "norms12 " << pointErr1 << " " << pointErr2 << std::endl;
        if (pointErr1 + pointErr2 < 30)
        {
          track->type = Static;
        }
        else
        {
          track->type = Undef;
        }
      }
    }
  }
}

void Tracker::trackWithKLT(cv::Mat& m_nextImg, cv::Mat& outputFrame, int frameInd, cv::Mat& depthImg) {
  curTracks.clear();
  curKeyPts.clear();
  curFrameProjMatr.release();

  cv::cvtColor(m_nextImg, outputFrame, CV_GRAY2BGR);

  if (prevTracks.size() > 0)
  {
    //get curr frame camera pose
    if (trajArchiver.poseProvider.poses.count(frameInd))
    {
      curFrameProjMatr = trajArchiver.poseProvider.poses[frameInd];
    }

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
        defineTrackType(prevTracks[i]);

#if 1
        cv::Scalar color;
        if (prevTracks[i]->type == Static)
          color = cv::Scalar(100, 100, 100);
        else
          color = cv::Scalar(250, 0, 250);
        cv::circle(outputFrame, prevCorners[i], 5, color, -1);
        cv::line(outputFrame, prevCorners[i], nextCorners[i], cv::Scalar(0, 250, 0));
//        cv::line(m_tracksFrame, prevCorners[i], nextCorners[i], cv::Scalar(0, 250, 0));
//        outputFrame += m_tracksFrame;
        cv::circle(outputFrame, nextCorners[i], 2, cv::Scalar(0, 250, 0), -1);
#endif
        curTracks.push_back(prevTracks[i]);
        curKeyPts.push_back(prevTracks[i]->bestCandidate->keyPt);
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