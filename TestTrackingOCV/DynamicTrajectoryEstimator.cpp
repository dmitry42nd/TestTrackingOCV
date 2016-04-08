#include "stdafx.h"

#include "DynamicTrajectoryEstimator.h"

int color;
DynamicTrajectoryEstimator::DynamicTrajectoryEstimator(CameraPoseProvider& poseProvider) :
    poseProvider(poseProvider),
    K(poseProvider.K),
    dist(poseProvider.dist),
    dynamicTracks()
{
  color = 0;

}


void DynamicTrajectoryEstimator::loadOnlyDynamicsTracksFromFile(std::string &pathToAllTracks)
{
  dynamicTracks.clear();
  std::ifstream allTracksData(pathToAllTracks);

  int trackType;
  int trackSize;
  while (allTracksData >> trackType) {
    std::shared_ptr<Track> newTrack(std::make_shared<Track>());
    newTrack->type = static_cast<PointType>(trackType);

    allTracksData >> trackSize;
    int frameId;
    float x,y;
    double depth;
    for(int i = 0; i < trackSize; i++)
    {
      allTracksData >> frameId >> x >> y >> depth;
      newTrack->history.push_back(std::make_shared<TrackedPoint>(cv::Point2f(x,y), frameId));
    }
    if(newTrack->type == Dynamic)
      dynamicTracks.push_back(newTrack);
  }
}



cv::Mat sup(cv::Mat const& projMatr, cv::Mat const& R, cv::Mat const& t)
{
  cv::Mat res;
  cv::Mat T1, T2;
  cv::Mat line = cv::Mat::zeros(1, 4, CV_64F);
  line.at<double>(0,3) = 1;

  cv::vconcat(projMatr, line, T1);

  cv::hconcat(R, t, T2);
  cv::vconcat(T2, line, T2);

  const cv::Rect roi = cv::Rect(0, 0, 4, 3);
  return (T2*T1)(roi);
}


double getProjErr(cv::Point2d p, cv::Mat const& projMatr, cv::Mat const& p4D)
{
  const cv::Rect roi = cv::Rect(0, 0, 1, 2);
  cv::Mat pr1_ = projMatr*p4D;
  cv::Mat pr1 = (pr1_ / pr1_.at<double>(2, 0))(roi);
  return cv::norm(cv::Vec2d(p.x - pr1.at<double>(0,0), p.y - pr1.at<double>(0,1)));
}


cv::Point3f getPoint3d(cv::Mat const& p4D)
{
  const cv::Rect roi = cv::Rect(0, 0, 1, 3);
  cv::Mat p3D = (p4D / p4D.at<double>(0, 3))(roi);
  return cv::Point3d(p3D);
}


void DynamicTrajectoryEstimator::setObjectWorldCoordsOnFrame(cv::Mat const& rvec, cv::Mat const& t, int frameId)
{
  cv::Mat R;
  cv::Rodrigues(rvec, R);
  cv::Mat line = cv::Mat::zeros(1, 4, CV_64F);
  line.at<double>(0,3) = 1;

  cv::Mat ocT;
  cv::hconcat(R, t, ocT);
  cv::vconcat(ocT, line, ocT);

  cv::Mat wcT;
  poseProvider.getProjMatrForFrame(wcT, frameId);
  cv::vconcat(wcT, line, wcT);
  //std::cerr << wcT << std::endl;

  cv::Mat owT = wcT.inv()*ocT;
  owTs.push_back(owT);
  //std::cerr << owT << std::endl;

  std::vector<cv::Mat> oX;

  for(int i = 0; i < objectPoints.size(); i++) {
    cv::Mat point = cv::Mat(objectPoints[i]);
    cv::vconcat(point, cv::Mat::ones(1,1,CV_64F),point);
    //std::cout << point << std::endl;

    cv::Mat wp = owT*point;
    //std::cout << wp.t() << std::endl;
    std::cout << color<< ", " << wp.at<double>(0,0) << ", " << wp.at<double>(0,1) << ", " << wp.at<double>(0,2) << std::endl;

    cv::Mat cp = ocT*point;
    //std::cout << cp << std::endl;
    //std::cout << cp.at<double>(0,0)/cp.at<double>(0,2) << ", " << cp.at<double>(0,1)/cp.at<double>(0,2)  << std::endl;
    //std::cout << imagePoints[i].x << ", " << imagePoints[i].y << std::endl << std::endl;


    oX.push_back(wp);
  }
  color+=SOME_STEP;
  //std::cerr << oX.size() << std::endl;
  oXs.push_back(oX);

  cv::Mat xs;
}


void updateByMask(std::vector<cv::Point2d> & vF, std::vector<cv::Point2d> & vL, cv::Mat const& mask) {
  std::vector<cv::Point2d>vF_, vL_;
  for(decltype(vF.size()) i = 0; i < vF.size(); i++) {
    if (mask.at<char>(0, i)) {
      vF_.push_back(vF[i]);
      vL_.push_back(vL[i]);;
    }
  }
  vF.swap(vF_);
  vL.swap(vL_);
}


void DynamicTrajectoryEstimator::buildTracks(int frameId, cv::Mat const& img, cv::Mat& outputFrame)
{
  objectPoints.clear();
  imagePoints.clear();
  unPointsF.clear();
  unPointsL.clear();
  for (auto track : dynamicTracks) {
    if (track->history.back()->frameId >= frameId) {
      auto p = std::find_if(track->history.begin(), track->history.end(),
                            [frameId](const std::shared_ptr<TrackedPoint> obj) { return obj->frameId == frameId; });
      if (p < track->history.end() - SOME_STEP) {
        unPointsF.push_back((*p)->undist(K, dist));
        unPointsL.push_back((*(p+SOME_STEP))->undist(K, dist));
      }
    }
  }

  //std::cout << "got " << unPointsF.size() << " points for frame pair " << frameId << " - " << frameId + SOME_STEP << std::endl;
  if (unPointsF.size() >= 5 /*&& unPointsL.size() >= 5*/) {

    //get R, t from frameId to frameId + SOME_STEP
    cv::Mat E = cv::Mat::zeros(3, 3, CV_64F);
    cv::Mat mask;
    E = cv::findEssentialMat(unPointsF, unPointsL,  1.0, cv::Point2d(0, 0), cv::RANSAC, 0.999, 0.001, mask);
    //std::cerr << mask.type() << ": " << mask.t() << std::endl;
    //std::cerr << E << std::endl;
    cv::Mat R, t;
    if (E.rows == 3 && E.cols == 3) {
      updateByMask(unPointsF, unPointsL, mask);
      if(unPointsF.size() < 5) {
        std::cerr << "algo failed 1\n";
        return;
      }

      cv::Mat mask2;
      cv::recoverPose(E, unPointsF, unPointsL, R, t, 1.0, cv::Point2d(0, 0), mask2);
      updateByMask(unPointsF, unPointsL, mask2);

      //cv::Point2d TrackedPoint::dist(cv::Mat const& K, cv::Mat const& dist)
      if(unPointsF.size() < 5) {
        std::cerr << "algo failed 2\n";
        return;
      }

      /*std::cerr << R << std::endl;
      std::cerr << t << std::endl;*/


      cv::Mat projMatrF, projMatrL;
      projMatrF = cv::Mat::eye(3,4,CV_64F);
      cv::hconcat(R, t, projMatrL);

      cv::Mat points4D;
      cv::triangulatePoints(projMatrF, projMatrL, unPointsF, unPointsL, points4D);

#if 0
      //debug info?
      std::vector<std::pair<double, int>> projErrs;
      for (int i = 0; i < points4D.cols /*== unPointsF.size()*/; i++) {
        double projErr1 = getProjErr(unPointsF[i], projMatrF, points4D.col(i));
        double projErr2 = getProjErr(unPointsL[i], projMatrL, points4D.col(i));

        double mean2Err = (projErr1 + projErr2) / 2;
        //std::cout << mean2Err << std::endl;
        projErrs.push_back(std::make_pair(mean2Err, i));
      }
#endif

      //http://stackoverflow.com/questions/19842035/stdmap-how-to-sort-by-value-then-by-key
      /*std::sort(projErrs.begin(), projErrs.end());

      f
       or (auto i = 0; i < 5; i++) {
        //std::cerr << projErrs[i].first << " ";
        int pId = projErrs[i].second;
        objectPoints.push_back(getPoint3d(points4D.col(pId)));
        imagePoints.push_back(unPointsL[pId]);
      }*/

      for (auto i = 0; i < points4D.cols; i++) {
        objectPoints.push_back(getPoint3d(points4D.col(i)));
        imagePoints.push_back(unPointsL[i]);
      }

      cv::Mat rvec, tvec, inliers;
      if(imagePoints.size() == 5)
        cv::solvePnPRansac(objectPoints, imagePoints, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(1, 4, CV_64F), rvec, tvec,
                           false, 100, 0.1, 0.99, inliers, cv::SOLVEPNP_EPNP);
      else if(unPointsL.size() > 3)
        cv::solvePnPRansac(objectPoints, imagePoints, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(1, 4, CV_64F), rvec, tvec,
                           false, 100, 0.1, 0.99, inliers, cv::SOLVEPNP_DLS);
      else
        std::cerr << "algo failed 3\n";

      std::cout << inliers.t() << std::endl;

      setObjectWorldCoordsOnFrame(rvec, tvec, frameId+SOME_STEP);
    }
    else {
      std::cerr << "five point failed\n";
    }
  }
}