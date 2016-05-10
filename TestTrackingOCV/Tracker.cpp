#include "stdafx.h"

#include "Tracker.h"
#include "TriangulateError.h"

#include "opencv2/video/tracking.hpp"

Tracker::Tracker(TrajectoryArchiver & trajArchiver, CameraPoseProvider & poseProvider, cv::Size imgSize) :
  trajArchiver(trajArchiver),
  poseProvider(poseProvider),
  imgSize(imgSize),
  K(poseProvider.K),
  dist(poseProvider.dist)
{
  fastDetector = cv::FastFeatureDetector::create(10);

  for (int j = 0; j < wy; j++)
  {
    detMasks.push_back(std::vector<cv::Mat>());
    for (int i = 0; i < wx; i++)
    {
      detMasks[j].push_back(cv::Mat());
    }
  }
  int sx = imgSize.width / wx;
  int sy = imgSize.height / wy;
  for (int i = 0; i < wx; i++)
  {
    for (int j = 0; j < wy; j++)
    {
      cv::Mat mask = cv::Mat::zeros(imgSize, CV_8U);
      for (int px = 0; px < imgSize.width; px++)
      {
        for (int py = 0; py < imgSize.height; py++)
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


void Tracker::createNewTrack(cv::Point2f point, int frameId, cv::KeyPoint const & keyPt, cv::Mat const & desc, double depth)
{
  std::shared_ptr<Track> newTrack(std::make_shared<Track>());
  std::shared_ptr<TrackedPoint> t = std::make_shared<TrackedPoint>(point, frameId, depth);
  newTrack->history.push_back(t);
  curTracks.push_back(newTrack);
}


int sat(int min, int val, int max)
{
  return min < val ? (val < max ? val : max) : min;
}


void roundCoords(cv::Point2i & ipt, cv::Point2f const& pt, cv::Size const& size)
{
  ipt.x = sat(0, static_cast<int>(round(pt.x)), size.width - 1);
  ipt.y = sat(0, static_cast<int>(round(pt.y)), size.height - 1);
}


cv::Mat Tracker::calcStatsByQuadrant(int wx, int wy, int ptNum, std::vector<std::shared_ptr<Track>> const& curTracks)
{
  //num of points in each quadrant
  cv::Mat res = cv::Mat::zeros(wy, wx, CV_32F);
  int sx = static_cast<int>(std::ceil(imgSize.width / wx));
  int sy = static_cast<int>(std::ceil(imgSize.height / wy));
  //count points in each quadrant
  for (auto const& track : curTracks)
  {
    auto const& hb = track->history.back();
    int resX = static_cast<int>(hb->loc.x / sx);
    int resY = static_cast<int>(hb->loc.y / sy);
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
    auto const & hb = track->history.back();
    int cx = hb->loc.x / wx;
    int cy = hb->loc.y / wy;

    curKeyPtsMap[Coords(cx, cy)] = hb->keyPt;
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


void Tracker::detectPoints(int indX, int indY, cv::Mat const& img, cv::Mat& depthImg, cv::Mat& outImg, int frameId) {

  std::vector<cv::KeyPoint> keyPts;
  std::cout << " detecting.. " << indX << " " << indY << std::endl;
  fastDetector->detect(img, keyPts, detMasks[indY][indX]);
  auto keyPtsSize = keyPts.size();

  //TODO: > kltPointsMin?
  if (keyPtsSize > 0)
  {
    //draw all FAST points (blue)
    /*
    for (auto const& kp : keyPts)
    {
      cv::circle(outImg, kp.pt, 3, cv::Scalar(255, 0, 0));
    }*/
    //get 80% of best by reduction points
    cv::KeyPointsFilter::retainBest(keyPts, static_cast<int>(keyPts.size() * 0.6));
    //draw all filtered points (yellow)
    /*for (auto const& kp : keyPts)
    {
      cv::circle(outImg, kp.pt, 3, cv::Scalar(0, 255, 255));
    }*/

    //TODO: magic numbers 16, 16
    std::vector<cv::KeyPoint> keyPtsFiltered = filterPoints(4, 4, keyPts);

    auto reduction = keyPtsFiltered.size() * 100 / keyPtsSize;
    std::cout << "key point reduction: " << reduction << " % " << keyPtsSize << ":" << keyPtsFiltered.size() << "\n";

    //draw final filtered points (red) and some stuff
    for (int i = 0; i < keyPtsFiltered.size(); i++)
    {
      cv::Point2i pt;
      roundCoords(pt, keyPtsFiltered[i].pt, imgSize);
      double depth = (double)(depthImg.at<ushort>(pt) / 5000.0);
      createNewTrack(keyPtsFiltered[i].pt, frameId, keyPtsFiltered[i], cv::Mat(), depth);
      cv::circle(outImg, keyPtsFiltered[i].pt, 3, cv::Scalar(0, 0, 255));
    }
  }
}


void printCamera(double * camera) {
  std::cout << "camera" << std::endl;
  for(int i = 0; i < 6; i++) {
    std::cout << camera[i] << " ";
  }
  std::cout << std::endl;
}


void makeCeresCamera(double* camera, CameraPose const& cp) {
  cv::Mat_<double> Rv;
  cv::Rodrigues(cp.R, Rv);

  for (auto i = 0; i < 3; i++) {
    camera[i] = Rv.at<double>(i, 0);
  }
  for (auto i = 0; i < 3; i++) {
    camera[i+3] = cp.t.at<double>(i, 0);
  }
  //printCamera(camera);
}


void Tracker::getProjectionAndNormCeres(double *camera, double *point, cv::Point2f &pp, cv::Point3f &np) {
  double p[3], xp, yp;
  ceres::AngleAxisRotatePoint(camera, point, p);
  p[0] += camera[3];
  p[1] += camera[4];
  p[2] += camera[5];
  xp = p[0] / p[2];
  yp = p[1] / p[2];

  np = cv::Point3f(xp, yp, 1);
  std::vector<cv::Point3f> vnp;
  vnp.push_back(np);
  std::vector<cv::Point2f> vpp;

  cv::projectPoints(vnp, cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 0), K, dist, vpp);
  pp = vpp[0];
}


void Tracker::defineTrackType(std::shared_ptr<Track> track) {
  double *point = new double[3];

  if (track->history.size() > 10) {
    #define SAMPLE_SIZE 5
    #define FIRST_FRAME 3
    ceres::Problem problem;

    std::vector<double *> cameras;
    std::vector<std::shared_ptr<TrackedPoint>> oPoints;
    std::vector<cv::Point2d> unPoints;
    auto trackSize = track->history.size();
    int  sample_size = SAMPLE_SIZE;
    int  step = static_cast<int>(std::ceil((trackSize - FIRST_FRAME)/ sample_size));

    decltype(trackSize) pId;
    for(pId = FIRST_FRAME; pId < trackSize; pId += step) {
      int tpid = pId + step;
        for (; pId < tpid; pId++) {
          auto pFrameId = track->history[pId]->frameId;
          CameraPose cp;
          if (!poseProvider.getCameraPoseForFrame(cp, pFrameId)) {
            oPoints.push_back(track->history[pId]);

            double *camera = new double[6];
            makeCeresCamera(camera, cp);
            cameras.push_back(camera);

            unPoints.push_back(oPoints.back()->undist(K, dist));
            ceres::CostFunction *cost_function = TriangulateError::Create(unPoints.back().x, unPoints.back().y, camera);
            problem.AddResidualBlock(cost_function, NULL, point);
            break;
          }
        }
    }

    //last frame must have
    if(pId != trackSize - 1)
    {
      pId = trackSize - 1;
      auto pFrameId = track->history[pId]->frameId;
      CameraPose cp;
      if (!poseProvider.getCameraPoseForFrame(cp, pFrameId)) { //what if not&
        oPoints.push_back(track->history[pId]);

        double *camera = new double[6];
        makeCeresCamera(camera, cp);
        cameras.push_back(camera);

        unPoints.push_back(oPoints.back()->undist(K, dist));
        ceres::CostFunction* cost_function = TriangulateError::Create(unPoints.back().x, unPoints.back().y, camera);
        problem.AddResidualBlock(cost_function, NULL, point);
      }
    }

    //get first approach of 3d point
    cv::Mat projMatrF, projMatrL;
    poseProvider.getProjMatrForFrame(projMatrF, oPoints.front()->frameId);
    poseProvider.getProjMatrForFrame(projMatrL, oPoints.back()->frameId);

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

    cv::Point2f ppF, ppL, ppM;
    cv::Point3f npF, npL, npM;
    getProjectionAndNormCeres(cameras.front(), point, ppF, npF);
    getProjectionAndNormCeres(cameras.back(), point, ppL, npL);

    CameraPose cpM;
    int pIdM = static_cast<int>(std::round(track->history.size()/2));
    poseProvider.getCameraPoseForFrame(cpM, track->history[pIdM]->frameId);
    double *cameraM = new double[6];
    makeCeresCamera(cameraM, cpM);

    getProjectionAndNormCeres(cameraM, point, ppM, npM);

    double a = cv::norm(cv::Mat(npF));
    double b = cv::norm(cv::Mat(npL));
    double c = cv::norm(cv::Mat(npL - npF));
    double median = pow(2 * pow(a, 2) + 2 * pow(b, 2) - pow(c, 2), 0.5) / 2;

    if(30*c > median)
    {
      double pointErr1 = cv::norm(cv::Mat(ppF - oPoints.front()->loc));
      double pointErr2 = cv::norm(cv::Mat(ppM - track->history[pIdM]->loc));
      double pointErr3 = cv::norm(cv::Mat(ppL - oPoints.back()->loc));
      //std::cerr << "norms: " << pointErr1 << " " << pointErr2 << std::endl;

      double mean2Error = (pointErr1 + pointErr2) / 2;
      double maxError = std::max(std::max(pointErr1, pointErr2), pointErr3);
      double mean3Error = (pointErr1 + pointErr2 + pointErr3) / 3;

      if (mean3Error < backProjThr) {
        track->type = Static;
      }
      else {
        track->type = Dynamic;
      }

      track->err[0] = mean2Error;
      track->err[1] = maxError;
      track->err[2] = mean3Error;
      track->defineTypeFrameId = track->history.back()->frameId;
    }

    delete[] point;
    for(auto camera : cameras) {
      delete[] camera;
    }
    delete[] cameraM;
  }
}

void Tracker::trackWithKLT(int frameId, cv::Mat const& img, cv::Mat& outputFrame, cv::Mat& depthImg) {

  if (ifTracksEnd(frameId)) {
    for (size_t i = 0; i < prevTracks.size(); i++) {
      if (prevTracks[i]->history.size() > 10) {
        defineTrackType(prevTracks[i]);
        lostTracks.push_back(prevTracks[i]);
        trajArchiver.archiveTrajectorySimple(prevTracks[i]);
      }
    }

    prevTracks.clear();
    prevImg.release();
  }
  else
  {
    curTracks.clear();
    cv::cvtColor(img, outputFrame, CV_GRAY2BGR);
    if (prevTracks.size() > 0) {
      std::vector<cv::Point2f> prevCorners;
      //TODO: optimize loc extraction
      for (auto &p : prevTracks)
        prevCorners.push_back(p->history.back()->loc);

      std::vector<cv::Point2f> nextCorners;
      std::vector<uchar> status;
      std::vector<float> err_;
      cv::calcOpticalFlowPyrLK(prevImg, img, prevCorners, nextCorners, status, err_,
                               cv::Size(11, 11), 3,
                               cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                               cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 1e-2);

      for (size_t i = 0; i < prevTracks.size(); i++) {
        cv::Mat err = cv::Mat(nextCorners[i] - prevCorners[i]);
        double trackDist = norm(err);
        if (trackDist < optFlowThr && status[i] && err_[i] < KLTErrThr &&
            nextCorners[i].x >= 0 && nextCorners[i].x < img.cols &&
            nextCorners[i].y >= 0 && nextCorners[i].y < img.rows) {
          cv::Point2i pt;
          roundCoords(pt, nextCorners[i], imgSize);
          double depth = (double) (depthImg.at<ushort>(pt)/5000.0);
          prevTracks[i]->history.push_back(std::make_shared<TrackedPoint>(nextCorners[i], frameId, depth));

          cv::circle(outputFrame, prevCorners[i], 5, cv::Scalar(250, 0, 250), -1);
          cv::line(outputFrame, prevCorners[i], nextCorners[i], cv::Scalar(0, 250, 0));
          cv::circle(outputFrame, nextCorners[i], 2, cv::Scalar(0, 250, 0), -1);

          /*if (prevTracks[i]->history.size() > 10 && (prevTracks[i]->type == Undef || frameId - prevTracks[i]->defineTypeFrameId > 10))
            defineTrackType(prevTracks[i]);*/

          curTracks.push_back(prevTracks[i]);
        }
        else
        {
          if (prevTracks[i]->history.size() > 10) {
            //if (prevTracks[i]->type == Undef)
            defineTrackType(prevTracks[i]);
            lostTracks.push_back(prevTracks[i]);
            trajArchiver.archiveTrajectorySimple(prevTracks[i]);
          }
        }
      }
    }

    cv::Mat boolStats = calcStatsByQuadrant(wx, wy, kltPointsMin / (wx * wy), curTracks);
    for (int i = 0; i < wx; i++) {
      for (int j = 0; j < wy; j++) {
        if (boolStats.at<uchar>(j, i) > 0) {
          detectPoints(i, j, img, depthImg, outputFrame, frameId);
        }
      }
    }

    prevTracks = curTracks;
    prevImg = img;
  }
}


void Tracker::drawFinalPointsTypes(int frameId, cv::Mat const& img, cv::Mat &outImg)
{
  cv::cvtColor(img, outImg, CV_GRAY2BGR);

  for(auto track : lostTracks) {
    for(auto p : track->history) {
      if(p->frameId == frameId) {
        cv::Scalar color;
        if (track->type == Static) {
          color = cv::Scalar(200, 200, 200);
          cv::circle(outImg, p->loc, 4, color, -1);
        }
        else if (track->type == Dynamic)
        {
          color = cv::Scalar(0, 200, 0);
          cv::circle(outImg, p->loc, 4, color, -1);
        }
        break;
      }
    }
  }
}

#if 0
void Tracker::saveAllTracks(std::string& pathToAllTracks)
{
  std::cout << "Total tracks: " << lostTracks.size() << std::endl;
  std::ofstream allTracksData(pathToAllTracks);

  for (auto track : lostTracks)
  {
    allTracksData << track->type << " " << track->history.size() << " ";
    for (auto p : track->history)
    {
      allTracksData << p->frameId << " " << p->loc.x << " " << p->loc.y << std::endl;
    }
  }
}
#endif