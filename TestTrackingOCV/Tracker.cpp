#include "stdafx.h"

#include "Tracker.h"
#include "TriangulateError.h"

#include "opencv2/video/tracking.hpp"

cv::Mat K, dist;
Tracker::Tracker(TrajectoryArchiver &trajArchiver, cv::Size imSize) :
  trajArchiver(trajArchiver),
  imSize(imSize)
{
  m_tracksFrame = cv::Mat::zeros(imSize, CV_8UC3);
  //orb = new cv::ORB(1000, 1.2, 8, 31, 0, 2, cv::ORB::FAST_SCORE, 31);
  //m_orbMatcher = new cv::BFMatcher(cv::NORM_HAMMING);

  fastDetector = cv::FastFeatureDetector::create(10);
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

  //kinect
  K = cv::Mat::zeros(3, 3, CV_64F);
  K.at<double>(0, 0) = 522.97697;
  K.at<double>(0, 2) = 318.47217;
  K.at<double>(1, 1) = 522.58746;
  K.at<double>(1, 2) = 256.49968;
  K.at<double>(2, 2) = 1.0;
  double dist_data[4] = {0.18962, -0.38214, 0, 0};
  dist = cv::Mat(1,4, CV_64F, dist_data);

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

void makeCamera(CameraPose const& cp, double* camera) {
  cv::Mat_<double> Rv;
  cv::Rodrigues(cp.R, Rv);

  for (auto i = 0; i < 3; i++) {
    camera[i] = Rv.at<double>(i, 0);
  }
  for (auto i = 0; i < 3; i++) {
    camera[i+3] = cp.t.at<double>(i, 0);
  }
  //printCamera(camera, 1);
}


void getProjectionAndNorm(double *camera, double *point, cv::Point2f & pp, cv::Point3f & np) {
  double p[3], xp, yp;
  ceres::AngleAxisRotatePoint(camera, point, p);
  p[0] += camera[3];
  p[1] += camera[4];
  p[2] += camera[5];
  xp = -p[0] / p[2];
  yp = -p[1] / p[2];

  np = cv::Point3f(xp, yp, 1);
  std::vector<cv::Point3f> vnp;
  vnp.push_back(np);
  std::vector<cv::Point2f> vpp;

  cv::projectPoints(vnp, cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 0), K, dist, vpp);
  pp = vpp[0];
}


void Tracker::defineTrackType(std::shared_ptr<Track> track, double errThr) {
  double *point = new double[3];

  if (track->history.size() > 10) {
    #define SAMPLE_SIZE 5
    #define FIRST_FRAME 5
    ceres::Problem problem;

    std::vector<double *> cameras;
    std::vector<std::shared_ptr<TrackedPoint>> oPoints;
    std::vector<cv::Point2d> unPoints;
    std::vector<CameraPose> cameraPoses;

    int trackSize = track->history.size();
    int sample_size = SAMPLE_SIZE;
    int step = std::ceil(trackSize / sample_size);

    int pId;
    for(pId = FIRST_FRAME; pId < trackSize; pId+=step)
    {
      for (; pId < pId + step; pId++)
      {
        CameraPose cp;
        auto pFrameId = track->history[pId]->frameId;
        trajArchiver.poseProvider.getPoseForFrame(cp, pFrameId);
        if (cv::countNonZero(cp.R) && cv::countNonZero(cp.t)) {
          cameraPoses.push_back(cp);

          std::shared_ptr<TrackedPoint> op = track->history[pId];
          oPoints.push_back(op);

          double *camera = new double[6];
          makeCamera(cp, camera);
          cameras.push_back(camera);

          cv::Point2d unp;
          undistPoint(op->location, K, dist, unp);
          unPoints.push_back(unp);

          ceres::CostFunction* cost_function = TriangulateError::Create(unp.x, unp.y, camera);
          problem.AddResidualBlock(cost_function, NULL, point);

          break;
        }
      }
    }

    //last frame must have
    if(pId != trackSize-1)
    {
      pId = trackSize-1;
      CameraPose cp;
      auto pFrameId = track->history[pId]->frameId;
      trajArchiver.poseProvider.getPoseForFrame(cp, pFrameId);
      if (cv::countNonZero(cp.R) && cv::countNonZero(cp.t)) {
        cameraPoses.push_back(cp);
        std::shared_ptr<TrackedPoint> op = track->history[pId];
        oPoints.push_back(op);
        double *camera = new double[6];
        cameras.push_back(camera);
        makeCamera(cp, camera);
        cv::Point2d unp;
        undistPoint(op->location, K, dist, unp);
        unPoints.push_back(unp);
        ceres::CostFunction *cost_function = TriangulateError::Create(unp.x, unp.y, camera);
        problem.AddResidualBlock(cost_function, NULL, point);
      }
    }

    //get first approach of 3d point
    cv::Mat projMatrF = trajArchiver.poseProvider.poses[oPoints.front()->frameId];
    cv::Mat projMatrL = trajArchiver.poseProvider.poses[oPoints.back()->frameId];

    cv::Mat undistProjPoint1 = cv::Mat(1,1,CV_64FC2);
    std::vector<cv::Vec2d> vunpF;
    vunpF.push_back(unPoints.front());
    std::vector<cv::Vec2d> vunpL;
    vunpL.push_back(unPoints.back());

    cv::Vec4d point4D;
    cv::triangulatePoints(projMatrF, projMatrL, vunpF, vunpL, point4D);

    point[0] = point4D[0]/point4D[3];
    point[1] = point4D[1]/point4D[3];
    point[2] = point4D[2]/point4D[3];

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //std::cout << "final p: " <<  point[0] << " " <<  point[1] << " " <<  point[2]<< "\n";

    cv::Point2f ppF;
    cv::Point3f npF;
    getProjectionAndNorm(cameras.front(), point, ppF, npF);

    cv::Point2f ppL;
    cv::Point3f npL;
    getProjectionAndNorm(cameras.back(), point, ppL, npL);

    CameraPose cpM;
    int pIdM = track->history.size()/2;
    trajArchiver.poseProvider.getPoseForFrame(cpM, track->history[pIdM]->frameId);
    double *cameraM = new double[6];
    makeCamera(cpM, cameraM);
    cv::Point2f ppM;
    cv::Point3f npM;//useles
    getProjectionAndNorm(cameraM, point, ppM, npM);

    double a = cv::norm(cv::Mat(npF));
    double b = cv::norm(cv::Mat(npL));
    double c = cv::norm(cv::Mat(npL - npF));
    double median = pow(2 * pow(a, 2) + 2 * pow(b, 2) - pow(c, 2), 0.5) / 2;

    if(20*c > median)
    {
      double pointErr1 = cv::norm(cv::Mat(ppF - oPoints.front()->location));
      double pointErr2 = cv::norm(cv::Mat(ppM - track->history[pIdM]->location));
      double pointErr3 = cv::norm(cv::Mat(ppL - oPoints.back()->location));
      //std::cerr << "norms: " << pointErr1 << " " << pointErr2 << std::endl;

      double mean2Error = (pointErr1 + pointErr2) / 2;
      double maxError = std::max(pointErr1, pointErr2);
      double mean3Error = (pointErr1 + pointErr2 + pointErr3) / 3;

      if (mean3Error < errThr) {
        track->type = Static;
      }
      else {
        track->type = Dynamic;
      }

      track->err[0] = mean2Error;
      track->err[1] = maxError;
      track->err[2] = mean3Error;
    }

    delete[] point;
    for(auto camera : cameras) {
      delete[] camera;
    }
  }
}
#endif

bool ifTracksEnd(int frameId)
{
  const int ends[5] = {653, 720, 782, 857, 917};
  for(auto i : ends)
  {
    if(frameId == i) return true;
  }
  return false;
}

void Tracker::trackWithKLT(cv::Mat& m_nextImg, cv::Mat& outputFrame, int frameInd, cv::Mat& depthImg) {
  curTracks.clear();

  cv::cvtColor(m_nextImg, outputFrame, CV_GRAY2BGR);

  if (prevTracks.size() > 0)
  {
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

      if (!ifTracksEnd(frameInd) && trackDist < trackThr && status[i] && nextCorners[i].x >= 0 && nextCorners[i].x < m_nextImg.cols &&
        nextCorners[i].y >= 0 && nextCorners[i].y < m_nextImg.rows)
      {
        cv::Point2f pt = fillDepthPt(nextCorners[i], ccx, ccy, cfx, cfy, ccx, ccy, cfx, cfy);
        int px, py;

        roundCoords(px, py, pt, m_nextImg);
        double depthVal = 0;
        if (pt.x > 0 && pt.y > 0 && pt.x < depthImg.cols && pt.y < depthImg.rows)
        {
          depthVal = (int)depthImg.at<ushort>(py, px);
          depthVal /= 5000.0;
        }

        prevTracks[i]->history.push_back(std::make_shared<TrackedPoint>(nextCorners[i], frameInd, 0, cv::KeyPoint(), cv::Mat(), depthVal));
        prevTracks[i]->bestCandidate = prevTracks[i]->history.back();

        cv::circle(outputFrame, prevCorners[i], 5, cv::Scalar(250, 0, 250), -1);
        cv::line(outputFrame, prevCorners[i], nextCorners[i], cv::Scalar(0, 250, 0));
        cv::circle(outputFrame, nextCorners[i], 2, cv::Scalar(0, 250, 0), -1);

        curTracks.push_back(prevTracks[i]);
      }
      else
      {
        if (prevTracks[i]->history.size() > 10)
        {
          defineTrackType(prevTracks[i], 80);
          lostTracks.push_back(prevTracks[i]);
          trajArchiver.archiveTrajectorySimple(prevTracks[i]);
        }
      }
    }
  }

  if(!ifTracksEnd(frameInd)) {
    cv::Mat boolStats = calcStatsByQuadrant(wx, wy, kltPtThr / (wx * wy), curTracks);

    for (int i = 0; i < wx; i++) {
      for (int j = 0; j < wy; j++) {
        if (boolStats.at<uchar>(j, i) > 0) {
          detectPoints(i, j, m_nextImg, depthImg, outputFrame, frameInd);
        }
      }
    }
  }

  prevTracks = curTracks;
  prevImg = m_nextImg;
}

void Tracker::buildTracks(cv::Mat &m_nextImg, cv::Mat &outputFrame, int frameInd)
{
  cv::cvtColor(m_nextImg, outputFrame, CV_GRAY2BGR);

  std::vector<cv::Point2d> pointsL, pointsR;
  //getPointsForFrameInd()
  for(auto track : lostTracks)
  {
    if (track->type == Dynamic)
    {
      for(int p = 0; p < track->history.size() - 1; p++)
      {
        if(track->history[p]->frameId == frameInd &&
           track->history[p + 1]->frameId == frameInd + 1)
        {
          cv::Point2d unp1, unp2;
          undistPoint(track->history[p]->location, K, dist, unp1);
          pointsL.push_back(unp1);
          undistPoint(track->history[p + 1]->location, K, dist, unp2);
          pointsR.push_back(unp2);
          break;
        }
      }
    }
  }


  if(pointsL.size() >= 5 /*&& pointsR.size() >= 5*/)
  {
    std::cerr << "got " << pointsL.size() << " points for frame pair " << frameInd << " - " << frameInd + 1 << std::endl;

    //get R, t from frameInd to frameInd + 1
    cv::Mat E = cv::Mat::zeros(3, 3, CV_64F);
    E = cv::findEssentialMat(pointsL, pointsR);
    std::cerr << E << std::endl;
    cv::Mat R, t;
    if (E.rows == 3 && E.cols == 3)
    {
      int res = cv::recoverPose(E, pointsL, pointsR, R, t);
      std::cerr << res << std::endl;
      std::cerr << R << std::endl;
      std::cerr << t << std::endl;
    }
    else
    {
      std::cerr << "five point failed\n";
    }

    //get first approach of 3d point

    cv::Mat projMatrF;
    cv::hconcat(cv::Mat::ones(3,3,CV_64F), cv::Mat::zeros(1,3,CV_64F), projMatrF);
    cv::Mat projMatrL = trajArchiver.poseProvider.poses[oPoints.back()->frameId];

    cv::Mat undistProjPoint1 = cv::Mat(1,1,CV_64FC2);
    std::vector<cv::Vec2d> vunpF;
    vunpF.push_back(unPoints.front());
    std::vector<cv::Vec2d> vunpL;
    vunpL.push_back(unPoints.back());

    cv::Vec4d point4D;
    cv::triangulatePoints(projMatrF, projMatrL, vunpF, vunpL, point4D);

    point[0] = point4D[0]/point4D[3];
    point[1] = point4D[1]/point4D[3];
    point[2] = point4D[2]/point4D[3];

  }


}


void Tracker::drawFinalPointsTypes(cv::Mat &m_nextImg, cv::Mat &outputFrame, int frameInd, cv::Mat &depthImg)
{
  cv::cvtColor(m_nextImg, outputFrame, CV_GRAY2BGR);
  std::cerr << frameInd << std::endl;

  for(auto track : lostTracks) {
    for(auto p : track->history) {
      if(p->frameId == frameInd) {
        cv::Scalar color;
        if (track->type == Static) {
          color = cv::Scalar(200, 200, 200);
          cv::circle(outputFrame, p->location, 2, color, -1);
        }
        else if (track->type == Dynamic)
        {
          color = cv::Scalar(0, 200, 0);
          cv::circle(outputFrame, p->location, 2, color, -1);
        }
        break;
      }
    }
  }
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
  std::cout << "Total tracks: " << lostTracks.size() << std::endl;

  int id = 0;
  for (auto track : lostTracks)
  {
    std::ofstream outTrackSave(pathToSaveFolder + std::to_string(id++) + ".txt");
    outTrackSave << track->type << " ";
    for (auto p : track->history)
    {
      outTrackSave << p->frameId << " " << p->location.x << " " << p->location.y << std::endl;
    }
  }
}