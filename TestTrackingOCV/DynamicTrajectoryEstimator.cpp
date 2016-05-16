#include "stdafx.h"

#include "DynamicTrajectoryEstimator.h"
#include "TriangulateError.h"
#include <boost/ref.hpp>
#include <boost/type_traits/remove_reference.hpp>

/*typedef std::vector<std::vector<std::shared_ptr<TrackedPoint>>> histVector;
typedef std::vector<std::vector<std::shared_ptr<TrackedPoint>>::const_iterator> histIterVector;*/

int color;
DynamicTrajectoryEstimator::DynamicTrajectoryEstimator(CameraPoseProvider& poseProvider) :
    poseProvider(poseProvider),
    K(poseProvider.K),
    dist(poseProvider.dist),
    dynamicTracks()
{
  color = 0;

  dataOut.open("../data4");
  dataOut_gt.open("../data4_gt");
  errOut1.open("../data");
  errOut2.open("../data_");

}

static const int HIST_LENGTH = 20;
static const int MIN_POINTS  = 5;

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
      newTrack->history.push_back(std::make_shared<TrackedPoint>(cv::Point2f(x,y), frameId, depth));
    }
    if(newTrack->type == Dynamic)
      dynamicTracks.push_back(newTrack);
  }
}


/*double getProjErr(cv::Point2d p, cv::Mat const& projMatr, cv::Mat const& p4D)
{
  static const cv::Rect roi = cv::Rect(0, 0, 1, 2);
  cv::Mat pr1_ = projMatr*p4D;
  cv::Mat pr1 = (pr1_ / pr1_.at<double>(2, 0))(roi);
  return cv::norm(cv::Vec2d(p.x - pr1.at<double>(0,0), p.y - pr1.at<double>(0,1)));
}*/


double getProjErr(cv::Point2d p, cv::Mat const& projMatr, double * point)
{
  double data[4] = {point[0], point[1],point[2],1.0};
  cv::Mat p4D = cv::Mat(4, 1, CV_64F, data);
  static const cv::Rect roi = cv::Rect(0, 0, 1, 2);
  cv::Mat pr1_ = projMatr*p4D;
  cv::Mat pr1 = (pr1_ / pr1_.at<double>(2, 0))(roi);
  return cv::norm(cv::Vec2d(p.x - pr1.at<double>(0,0), p.y - pr1.at<double>(0,1)));
}


double getProjErrCeres(double *camera, double *point, cv::Point2d p2d)
{
  double p[3], xp, yp;
  ceres::AngleAxisRotatePoint(camera, point, p);
  p[0] += camera[3];
  p[1] += camera[4];
  p[2] += camera[5];
  xp = p[0] / p[2];
  yp = p[1] / p[2];

  cv::Point2d p_ = cv::Point2d(xp, yp);
  //std::cout << xp - p2d.x << " " << yp - p2d.y << std::endl;
  return  cv::norm(cv::Vec2d(p2d.x - p_.x, p2d.y - p_.y));
}


void DynamicTrajectoryEstimator::getProjectionAndNormCeres(double *camera, double *point, cv::Point2f &pp, cv::Point3f &np) {
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

cv::Point3f getPoint3d(cv::Mat const& p4D)
{
  const cv::Rect roi = cv::Rect(0, 0, 1, 3);
  cv::Mat p3D = (p4D / p4D.at<double>(0, 3))(roi);
  return cv::Point3d(p3D);
}

cv::Point3d getPoint3dCeres(double *p3D)
{
  return cv::Point3d(p3D[0], p3D[1], p3D[2]);
}


double DynamicTrajectoryEstimator::getScale(cv::Mat const& rvec, cv::Mat const& t, int frameId, cv::Mat const& inliers)
{
  cv::Mat line = cv::Mat::zeros(1, 4, CV_64F);
  line.at<double>(0,3) = 1;

  cv::Mat R;
  cv::Rodrigues(rvec, R);

  cv::Mat ocT;
  cv::hconcat(R, t, ocT);
  cv::vconcat(ocT, line, ocT);

  cv::Mat wcT;
  poseProvider.getProjMatrForFrame(wcT, frameId);
  cv::vconcat(wcT, line, wcT);
  //std::cerr << wcT << std::endl;

  cv::Mat owT = wcT.inv()*ocT;
  //std::cerr << owT << std::endl;

  std::vector<double> depths;
  for(auto o : hists_) {
    auto p = std::find_if(o.cbegin(), o.cend(),
                          [frameId](const std::shared_ptr<TrackedPoint> obj) { return obj->frameId == frameId; });
    depths.push_back((*p)->depth);
  }

  //scale stuff
  double scale = 0;
  double cnt = 0;
  for(int j = 0; j < inliers.rows; j++) {
    int i = inliers.row(j).at<int>(0, 0);
    cv::Mat point = cv::Mat(objectPoints[i]);
    point.convertTo(point, CV_64F);
    //std::cout << point.type() << std::endl;
    cv::vconcat(point, cv::Mat::ones(1, 1, CV_64F), point);
    cv::Mat wp = owT * point;

    double s0 = wp.at<double>(0, 2);
    double s1 = depths[i];
    if(s1 != 0 && s0 != 0) {
      cnt++;
      scale += std::fabs(s1 - s0);
    }
  }
  if(cnt != 0 )
    scale /= cnt;
  return scale;
}


void DynamicTrajectoryEstimator::setObjectWorldCoordsOnFrame(cv::Mat const& rvec, cv::Mat const& t, int frameId, cv::Mat const& inliers,
  std::vector<cv::Point3d> &Xs,
  std::vector<cv::Point2d> &projXs_debug)
{
  cv::Mat line = cv::Mat::zeros(1, 4, CV_64F);
  line.at<double>(0,3) = 1;

  cv::Mat R;
  cv::Rodrigues(rvec, R);

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

  std::vector<double> depths;
  imagePoints.clear();
  for(auto o : hists_) {
    auto p = std::find_if(o.cbegin(), o.cend(),
                          [frameId](const std::shared_ptr<TrackedPoint> obj) { return obj->frameId == frameId; });
    imagePoints.push_back((*p)->undist(K,dist));
    depths.push_back((*p)->depth);
  }

  double xa,ya,za;
  double xd,yd,zd;
  xa=ya=za=xd=yd=zd=0;
  double cnt = 0;

  std::vector<double> err;
  for(int j = 0; j < inliers.rows; j++) {
  //for(int j = 0; j < objectPoints.size(); j++) {
    int i = inliers.row(j).at<int>(0,0);
    //int i = j;

    cv::Mat point = cv::Mat(objectPoints[i]);
    point.convertTo(point, CV_64F);
    cv::vconcat(point, cv::Mat::ones(1,1,CV_64F),point);
    //std::cout << point << std::endl;

    cv::Mat wp = owT*point;
    //std::cout << wp.t() << std::endl;

    double x = wp.at<double>(0,0);
    double y = wp.at<double>(0,1);
    double z = wp.at<double>(0,2);

    //dpeth
    cv::Mat pointd = cv::Mat(cv::Point3d(objectPoints[i].x, objectPoints[i].y, depths[i]));
    pointd.convertTo(pointd, CV_64F);
    cv::vconcat(pointd, cv::Mat::ones(1,1,CV_64F),pointd);
    cv::Mat wpd = owT*pointd;
    xd = x;
    yd = y;
    //zd = depths[i];

    if(depths[i] > 0) {
      xa += wp.at<double>(0,0);
      ya += wp.at<double>(0,1);
      za += wp.at<double>(0,2);
      zd += depths[i];
      cnt++;

      dataOut << i << ", " << x << ", " << y << ", " << z-scale_z << std::endl;
      dataOut_gt << color << ", " << xd << ", " << yd << ", " << depths[i] << std::endl;

      //errOut2 << ((z-scale_z) - depths[i]) << std::endl;
    }

    Xs.push_back(cv::Point3d(x,y,z));

    //reprojection error
    cv::Mat cp = ocT*point;
    projXs_debug.push_back(cv::Point2d(cp.at<double>(0,0)/cp.at<double>(0,2), cp.at<double>(0,1)/cp.at<double>(0,2)));
    //std::cout << cp << std::endl;
    cv::Mat a = cv::Mat(cv::Point2d(cp.at<double>(0,0)/cp.at<double>(0,2), cp.at<double>(0,1)/cp.at<double>(0,2)));
    cv::Mat b = cv::Mat(cv::Point2d(imagePoints[i].x, imagePoints[i].y));
    double projErr = cv::norm(a-b);
    //std::cout << projErr << std::endl;
    errOut1 << projErr << std::endl;
    err.push_back(projErr);
  }

  if(cnt > 0) {
    xa /= cnt;
    ya /= cnt;
    za /= cnt;
    zd /= cnt;

    //dataOut << 0 << ", " << xa << ", " << ya << ", " << za*scale_z << std::endl;
    //dataOut_gt << color << ", " << xa << ", " << ya << ", " << zt << std::endl;
    errOut2 << ((za - scale_z) - zd) << std::endl;
  }

  color++;
}

void filterByMask(cv::Mat const &mask, std::vector<cv::Point2d> &vF, std::vector<cv::Point2d> &vL,
                  histVector &its) {

  histVector its__;
  std::vector<cv::Point2d> vF_, vL_;

  for(decltype(vF.size()) i = 0; i < vF.size(); i++) {
    if (mask.at<char>(0, i)) {
      vF_.push_back(vF[i]);
      vL_.push_back(vL[i]);
      its__.push_back(its[i]);
    }
  }

  vF_.swap(vF);
  vL_.swap(vL);
  its__.swap(its);
}


void DynamicTrajectoryEstimator::filterByMaskDebug(cv::Mat const &mask, std::vector<cv::Point2d> &vF, std::vector<cv::Point2d> &vL,
                                                   std::vector<cv::Point2d> &v, histVector &its, std::vector<int> &trackIds, int i_) {

  histVector its__;
  std::vector<cv::Point2d> vF_, vL_, v_;
  std::vector<int> trackIds_;

  for(decltype(vF.size()) i = 0; i < vF.size(); i++) {
    if (mask.at<char>(0, i)) {
      vF_.push_back(vF[i]);
      vL_.push_back(vL[i]);
      v_.push_back(v[i]);
      its__.push_back(its[i]);
      trackIds_.push_back(trackIds[i]);
    }
  }

  vF_.swap(vF);
  vL_.swap(vL);
  v_.swap(v);
  its__.swap(its);
  trackIds_.swap(trackIds);

  cv::Mat outImg;
  img.copyTo(outImg);

  for(auto p : v){
    cv::circle(outImg, p, 3, cv::Scalar(0, 0, 200), -1);
  }

  std::string outImgName = std::to_string(i_) + ".bmp";
  cv::imwrite(outImgName, outImg);

}

void makeCeresCamera(double* camera, cv::Mat const& R, cv::Mat const& t) {
  cv::Mat_<double> Rv;
  cv::Rodrigues(R, Rv);

  for (auto i = 0; i < 3; i++) {
    camera[i] = Rv.at<double>(i, 0);
  }
  for (auto i = 0; i < 3; i++) {
    camera[i+3] = t.at<double>(i, 0);
  }

  /*std::cout << "camera" << std::endl;
  for(int i = 0; i < 6; i++) {
    std::cout << camera[i] << " ";
  }
  std::cout << std::endl;*/

}

void DynamicTrajectoryEstimator::reset() {
  owTs.clear();

  oldrvec.clear();
  oldtvec.clear();
  hists_.clear();
  scaleObs.clear();
  scaleCameras.clear();
  scaleInliers.clear();
  scaleXsF.clear();
  scaleXsL.clear();
  projXsL_debug.clear();

}

void DynamicTrajectoryEstimator::block1(int frameIdF, int frameIdL) {

  sprintf(imgn, "../essmasks/%03d.png", frameIdF);
  std::cout << imgn << std::endl;
  cv::Mat essmask = cv::imread(imgn, CV_8U);

  std::vector<cv::Point2d>  unPointsF, unPointsL;
  std::vector<cv::Point2d>  pointsF_debug;
  std::vector<int> trackIds;
  histVector hists;

  //sprintf(imgn, "../outProc/%06d.png", frameIdF);
  std::string ImgName = "../outProc/" + std::to_string(frameIdF) + ".bmp";
  img = cv::imread(ImgName, 1);
  cv::Mat outImg;
  img.copyTo(outImg);

  int id = 0;
  for (auto track : dynamicTracks) {
    if (track->history.front()->frameId <= frameIdF && track->history.back()->frameId >= frameIdL) {
      auto pF = std::find_if(track->history.cbegin(), track->history.cend(),
                             [frameIdF](const std::shared_ptr<TrackedPoint> obj) { return obj->frameId == frameIdF; });

      //std::cout << trunc((*pF)->loc.x) << " " <<  trunc((*pF)->loc.y) << std::endl;
      bool en = int(essmask.at<uchar>(trunc((*pF)->loc.y), trunc((*pF)->loc.x)));
      auto pL = std::find_if(pF, track->history.cend(),
                             [frameIdL](const std::shared_ptr<TrackedPoint> obj) { return obj->frameId == frameIdL; });


      if (en && pF != track->history.end() && pL != track->history.end() && std::distance(pF, pL) > HIST_LENGTH) {
        unPointsF.push_back((*pF)->undist(K, dist));
        unPointsL.push_back((*(pF+HIST_LENGTH))->undist(K, dist));
        hists.push_back(track->history);

        cv::circle(outImg, (*pF)->loc, 3, cv::Scalar(0, 0, 200), -1);
        pointsF_debug.push_back((*pF)->loc);
        trackIds.push_back(id);
        std::cout << "added: " << unPointsF.back() << " " << unPointsF.back() << std::endl;      }
    }

    std::string outImgName = std::to_string(frameIdF) + ".bmp";
    cv::imwrite(outImgName, outImg);
    ++id;
  }

  //std::cout << "got " << unPointsF.size() << " points for frame pair " << frameId << " - " << frameId + SOME_STEP << std::endl;
  if (hists.size() >= MIN_POINTS) {
    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(unPointsF, unPointsL, 1.0, cv::Point2d(0, 0), cv::RANSAC, 0.99, essThr, mask);
    //filterByMaskDebug(mask, unPointsF, unPointsL, pointsF_debug, hists, trackIds, 1);
    //std::cerr << E << std::endl;

    cv::Mat R, t, R1, R2;
    if (E.rows == 3 && E.cols == 3) {
      cv::recoverPose(E, unPointsF, unPointsL, R, t, 1.0, cv::Point2d(0, 0), mask);
      /*cv::decomposeEssentialMat(E, R1, R2, t);
      R2.copyTo(R);
      t = t;*/

      //t*=10;
      //std::cerr << mask.type() << ": " << mask.t() << std::endl;
      filterByMaskDebug(mask, unPointsF, unPointsL, pointsF_debug, hists, trackIds, 2);
      if(hists.size() < MIN_POINTS) {
        std::cerr << "algo failed 2\n";
        return;
      }

      //for(int i = trackIds.size()-1; i >= 0; --i)
      //  dynamicTracks.erase(dynamicTracks.begin() + trackIds[i]);

      cv::Mat projMatrF = cv::Mat::eye(3,4,CV_64F);
      cv::Mat projMatrL;
      cv::hconcat(R, t, projMatrL);

      cv::Mat points4D;
      cv::triangulatePoints(projMatrF, projMatrL, unPointsF, unPointsL, points4D);

      std::vector<double *> points;
      ceres::Problem problem;

      double *cameraF = new double[6];
      makeCeresCamera(cameraF, cv::Mat::eye(3,3,CV_64F), cv::Mat::zeros(1,3,CV_64F));
      double *cameraL = new double[6];
      makeCeresCamera(cameraL, R, t);

      for(auto i = 0; i < points4D.cols; i++) {
        double *point = new double[3];

        cv::Mat p4D;
        points4D.col(i).copyTo(p4D);
        point[0] = p4D.at<double>(0,0)/p4D.at<double>(0,3);
        point[1] = p4D.at<double>(0,1)/p4D.at<double>(0,3);
        point[2] = p4D.at<double>(0,2)/p4D.at<double>(0,3);
        //std::cout << point[0] << " " << point[1] << " " << point[2] << std::endl;

        ceres::CostFunction *cost_function1 = TriangulateError2::Create(unPointsF[i].x, unPointsF[i].y, cameraF);
        ceres::CostFunction *cost_function2 = TriangulateError2::Create(unPointsL[i].x, unPointsL[i].y, cameraL);
        problem.AddResidualBlock(cost_function1, NULL, point);
        problem.AddResidualBlock(cost_function2, NULL, point);
        points.push_back(point);
      }
      //std::cout << std::endl;

      ceres::Solver::Options options;
      options.minimizer_progress_to_stdout = true;
      options.linear_solver_type = ceres::DENSE_SCHUR;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);

      /*for(auto p : points) {
        std::cout << p[0] << " " << p[1] << " " << p[2] << std::endl;
      }*/

      std::vector<std::pair<double, int>> projErrs;
      for (int i = 0; i < points4D.cols ; i++) {
        double reprojErr1 = getProjErrCeres(cameraF, points[i], unPointsF[i]);
        double reprojErr2 = getProjErrCeres(cameraL, points[i], unPointsL[i]);
        double mean2Err = (reprojErr1 + reprojErr2) / 2;
        std::cout << mean2Err << std::endl;
        projErrs.push_back(std::make_pair(mean2Err , i));
      }

      /*std::cout << "pew" <<  std::endl;
      for(int i = 0; i < points.size(); ++i) {
        std::cout << "[" <<points[i][0] << ", " << points[i][1] << ", " << points[i][2] << "] "<< unPointsF[i] << std::endl;
      }*/

      //http://stackoverflow.com/questions/19842035/stdmap-how-to-sort-by-value-then-by-key
      //std::sort(projErrs.begin(), projErrs.end());

      objectPoints.clear();
      std::cout << projErrs.size() << std::endl;
      //for(auto i = 0; i < MIN_POINTS; i++) {
      for(auto i = 0; i < points.size(); i++) {
        //int pId = projErrs[i].second;
        int pId = i;
        hists_.push_back(hists[pId]);
        objectPoints.push_back(getPoint3dCeres(points[pId]));

        /*double *objP = new double[3];
        objP[0] = points[pId][0];
        objP[1] = points[pId][1];
        objP[2] = points[pId][2];*/
        objectPoints_ceres.push_back(points[pId]);
      }

      std::cout << "wew!" <<std::endl;
      projErrs.clear();
      for (int i = 0; i < points4D.cols ; i++) {
        double reprojErr1 = getProjErr(unPointsF[i], projMatrF, points[i]);
        double reprojErr2 = getProjErr(unPointsL[i], projMatrL, points[i]);
        double mean2Err = (reprojErr1 + reprojErr2) / 2;
        std::cout << mean2Err << std::endl;
        projErrs.push_back(std::make_pair(mean2Err , i));
      }

    } else {
      std::cerr << "five point failed\n";
    }
  }
}

void DynamicTrajectoryEstimator::renewObjectPoints(std::vector<double *> objectPoints_ceres,
                                                   std::vector<cv::Point3d>  & objectPoints) {
  for(auto i = 0; i < objectPoints_ceres.size(); ++i) {
    objectPoints[i] = cv::Point3d(objectPoints_ceres[i][0],objectPoints_ceres[i][1],objectPoints_ceres[i][2]);
    std::cout << objectPoints[i] << std::endl;
  }
  std::cout << "pew!" << std::endl;

}


void DynamicTrajectoryEstimator::buildTrack(int frameIdF, int frameIdL) {
  reset();

  block1(frameIdF, frameIdL);

  ceres::Solver::Options mainOptions;
  mainOptions.minimizer_progress_to_stdout = true;
  mainOptions.linear_solver_type = ceres::DENSE_SCHUR;
  ceres::Solver::Summary mainSummary;

  std::vector<int> fids_;
  ceres::Problem mainProblem;
  cv::Mat outImg;
  int cnt0 = 0;
  for(int fid = frameIdF; fid < frameIdL; fid++) {

    //sprintf(imgn, "../outProc/%06d.png", fid);
    std::string ImgName = "../outProc/" + std::to_string(fid) + ".bmp";
    img = cv::imread(ImgName, 1);
    img.copyTo(outImg);

    std::vector<cv::Point2f> coords;
    imagePoints.clear();
    for (auto o : hists_) {
      auto p = std::find_if(o.begin(), o.end(),
                            [fid](const std::shared_ptr<TrackedPoint> obj) { return obj->frameId == fid; });
      coords.push_back((*p)->loc);
      if (p != o.end())
        imagePoints.push_back((*p)->undist(K, dist));
      else
        std::cout << "Alarm!" <<std::endl;
    }

    /*if (fid == frameIdF) {
      std::cout << "pew2" << std::endl;
      for (int i = 0; i < objectPoints.size(); ++i) {
        std::cout << objectPoints[i] << " " << imagePoints[i] << std::endl;
      }
    }*/

    cv::Mat rvec, tvec, inliers;
    if(imagePoints.size() >= 4)
      cv::solvePnPRansac(objectPoints, imagePoints, cv::Mat::eye(3, 3, CV_32F), cv::Mat::zeros(1, 4, CV_32F),
                         rvec, tvec, false, 100, pnpThr, 0.99, inliers, cv::SOLVEPNP_EPNP);
      //cv::solvePnP(objectPoints, imagePoints, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(1, 4, CV_64F), rvec, tvec, false, cv::SOLVEPNP_EPNP);


    /*else if(imagePoints.size() >= 3)
      cv::solvePnPRansac(objectPoints, imagePoints, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(1, 4, CV_64F), rvec, tvec,
                         false, 200, 0.03, 0.98, inliers, cv::SOLVEPNP_DLS);*/
    else
      std::cerr << "algo failed 3\n";

    if(true || inliers.rows > 0) {
      oldrvec.push_back(rvec); //just for debug
      oldtvec.push_back(tvec);

      std::vector<double> camera;
      for (auto i = 0; i < 3; i++) {
        camera.push_back(rvec.at<double>(i, 0));
      }
      for (auto i = 0; i < 3; i++) {
        camera.push_back(tvec.at<double>(i, 0));
      }

      scaleObs.push_back(imagePoints);
      scaleCameras.push_back(camera);

      fids_.push_back(fid);
      scaleInliers.push_back(inliers);

      std::cout << inliers.type() << " " << inliers.t() << std::endl;
      for (int j = 0; j < inliers.rows; j++) {
      //for (int j = 0; j < objectPoints_ceres.size(); j++) {
        int i = inliers.row(j).at<int>(0, 0);
        //int i = j;
        ceres::CostFunction *cost_function = TriangulateError3::Create(scaleObs.back()[i].x, scaleObs.back()[i].y);
        mainProblem.AddResidualBlock(cost_function, NULL, scaleCameras.back().data(), objectPoints_ceres[i]);

        if(i == 7)
          cv::circle(outImg, coords[i], 3, cv::Scalar(200, 0, 0), -1);
        else
          cv::circle(outImg, coords[i], 3, cv::Scalar(0, 0, 200), -1);
      }
      /*if(cnt0 == HIST_LENGTH) {
        cnt0 = 0;
        ceres::Solve(mainOptions, &mainProblem, &mainSummary);
        renewObjectPoints(objectPoints_ceres, objectPoints);
      }
      cnt0++;*/
    } else {
      std::cout << "no inliers in solveRansac " << fid << "\n";
    }

    std::string outImgName =  "dout/" + std::to_string(fid) + ".bmp";
    cv::imwrite(outImgName, outImg);
  }

  ceres::Solve(mainOptions, &mainProblem, &mainSummary);
  /*std::cout << "frames chosed: " << fids_.size() << std::endl;
  std::cout << "prev obj points: " << std::endl;
  for(auto p : objectPoints) {
    std::cout << p << std::endl;
  }*/

  //std::cout << "actual obj points: " << std::endl;
  objectPoints.clear();
  for(auto i = 0; i < objectPoints_ceres.size(); i++) {
    cv::Point3d p = cv::Point3d(objectPoints_ceres[i][0], objectPoints_ceres[i][1], objectPoints_ceres[i][2]);
    //std::cout << p << std::endl;
    objectPoints.push_back(p);
  }
  //std::cout << std::endl;

  //scale stuff
  scale_z = 0;
  double cnt = 0;
  for(auto i = 0; i < scaleCameras.size(); i++) {
    cv::Mat rvec = cv::Mat(3, 1, CV_64F);
    rvec.at<double>(0, 0) = scaleCameras[i][0];
    rvec.at<double>(1, 0) = scaleCameras[i][1];
    rvec.at<double>(2, 0) = scaleCameras[i][2];

    //std::cout << oldrvec[i].t() << std::endl;
    //std::cout << rvec.t() << std::endl;

    cv::Mat tvec = cv::Mat(3, 1, CV_64F);
    tvec.at<double>(0, 0) = scaleCameras[i][3];
    tvec.at<double>(1, 0) = scaleCameras[i][4];
    tvec.at<double>(2, 0) = scaleCameras[i][5];

    double s = getScale(rvec, tvec, fids_[i], scaleInliers[i]);
    if(s > 0) {
      scale_z+=s;
      cnt++;
    }
  }
  scale_z/=cnt;
  std::cout << "sale_: " << scale_z << std::endl;

  for(auto i = 0; i < scaleCameras.size(); i++) {
    cv::Mat rvec = cv::Mat(3, 1, CV_64F);
    rvec.at<double>(0, 0) = scaleCameras[i][0];
    rvec.at<double>(1, 0) = scaleCameras[i][1];
    rvec.at<double>(2, 0) = scaleCameras[i][2];

    //std::cout << oldrvec[i].t() << std::endl;
    //std::cout << rvec.t() << std::endl;

    cv::Mat tvec = cv::Mat(3, 1, CV_64F);
    tvec.at<double>(0, 0) = scaleCameras[i][3];
    tvec.at<double>(1, 0) = scaleCameras[i][4];
    tvec.at<double>(2, 0) = scaleCameras[i][5];
    //tvec = tvec*2;

    //std::cout << oldtvec[i].t() <<std::endl;
    //std::cout << tvec.t() << std::endl;

    std::vector<cv::Point3d> Xs;
    std::vector<cv::Point2d> projXs;
    if (i == 0) {
      Fdebug_ = fids_[i];
      std::cout << fids_[i] << std::endl;
      setObjectWorldCoordsOnFrame(rvec, tvec, fids_[i], scaleInliers[i], scaleXsF, projXs);
    }
    else if (i == scaleCameras.size() - 1) {
      Ldebug_ = fids_[i];
      std::cout << fids_[i] << std::endl;
      setObjectWorldCoordsOnFrame(rvec, tvec, fids_[i], scaleInliers[i], scaleXsL, projXsL_debug);
    }
    else
      setObjectWorldCoordsOnFrame(rvec, tvec, fids_[i], scaleInliers[i], Xs, projXs);
  }

#if 0
  cv::Point3d meanXF(0,0,0);
  for(auto p : scaleXsF)
    meanXF += p;
  meanXF /= (double)scaleXsF.size();

  cv::Point3d meanXL(0,0,0);
  for(auto p : scaleXsL)
    meanXL += p;
  meanXL /= (double)scaleXsL.size();

  std::cout << meanXF << std::endl;
  std::cout << meanXL << std::endl;
  //std::cout << (double)obs.size() << " : " << (Ldebug_ - Fdebug_) << std::endl;
  cv::Point3d Vest = (meanXL - meanXF) / (Ldebug_ - Fdebug_);
  //std::cout << V << std::endl;

  scaleSolver(scaleObs, scaleCameras, scaleInliers, scaleXsF, Vest);
#endif
}


#if 0
static void DynamicTrajectoryEstimator::scaleSolver(std::vector<std::vector<cv::Point2d>> obs,
                                             std::vector<std::vector<double>> cameras,
                                             std::vector<cv::Mat> inliers,
                                             std::vector<cv::Point3d> XsF,
                                             cv::Point3d V) {

  ceres::Problem scaleProblem;
  ceres::Solver::Options scaleOptions;
  ceres::Solver::Summary scaleSummary;

  std::vector<double> s;
  s.push_back(1.0);

  /*std::vector<double> rvec;
  rvec.push_back(0);
  rvec.push_back(0);
  rvec.push_back(0);*/

  std::vector<double> v;
  v.push_back(V.x);
  v.push_back(V.y);
  v.push_back(V.z);

  /*std::cout << "pew1\n";
  for(auto p : projXsL_debug)
    std::cout << p << " ";
  std::cout << std::endl;

  std::cout << "pew2\n";
  for(auto p : obs.back())
    std::cout << p << " ";
  std::cout << std::endl;*/

  for(int k = 0; k < obs.size(); k++) {
    for(int j = 0; j < inliers[k].rows; j++) {
      int pid = inliers[k].row(j).at<int>(0,0);
      cv::Point2d const& obs_ = obs[k][pid];
      cv::Point3d const& X_   = XsF[pid];
      std::vector<double> const& camera_ = cameras[k];

      ceres::CostFunction *cost_function = ScaleError::Create(obs_.x, obs_.y, X_.x, X_.y, X_.z, camera_.data(), k);
      scaleProblem.AddResidualBlock(cost_function, NULL, s.data(), /*&rvec[0],*/ v.data());
    }
  }

  /*std::cout << "rvec: " << std::endl;
  for(auto r : rvec)
    std::cout << r << " ";
  std::cout << std::endl;*/

  std::cout << "v: " << std::endl;
  for(auto v_ : v)
    std::cout << v_ << " ";
  std::cout << std::endl;

  scaleOptions.minimizer_progress_to_stdout = true;
  scaleOptions.linear_solver_type = ceres::DENSE_SCHUR;
  ceres::Solve(scaleOptions, &scaleProblem, &scaleSummary);
  std::cout << "scale Solver finished" << std::endl;

  /*std::cout << "final rvec: " << std::endl;
  for(auto r : rvec)
    std::cout << r << " ";
  std::cout << std::endl;*/

  std::cout << "final v: " << std::endl;
  for(auto v_ : v)
    std::cout << v_ << " ";
  std::cout << std::endl;

  std::cout << "scale: " << s[0] << std::endl;
}

#else

static void DynamicTrajectoryEstimator::scaleSolver(std::vector<std::vector<cv::Point2d>> obs,
                                                    std::vector<std::vector<double>> cameras,
                                                    std::vector<cv::Mat> inliers,
                                                    std::vector<cv::Point3d> XsF,
                                                    cv::Point3d V) {

  ceres::Problem scaleProblem;
  ceres::Solver::Options scaleOptions;
  ceres::Solver::Summary scaleSummary;

  std::vector<double> s;
  s.push_back(1/*4.22329*/);

  std::vector<double> rvec;
  rvec.push_back(0.00);
  rvec.push_back(0.00);
  rvec.push_back(0.00);

  std::vector<double> v;
  v.push_back(V.x);
  v.push_back(V.y);
  v.push_back(V.z);

  /*v.push_back(0.0774386);
  v.push_back(-0.0465671);
  v.push_back(-0.0540333);*/


  for(int k = 0; k < obs.size(); k++) {
    for(int j = 0; j < inliers[k].rows /*XsF.size()*/; j++) {
      int pId = inliers[k].row(j).at<int>(0,0);
      cv::Point2d const& obs_ = obs[k][pId];
      cv::Point3d const& X_   = XsF[pId];
      std::vector<double> const& camera_ = cameras[k];

      ceres::CostFunction *cost_function = ScaleError::Create(obs_.x, obs_.y, X_.x, X_.y, X_.z, camera_.data(), k);
      scaleProblem.AddResidualBlock(cost_function, NULL, s.data(), /*rvec.data(),*/ v.data());
    }
  }

  std::cout << "rvec: " << std::endl;
  for(auto r : rvec)
    std::cout << r << " ";
  std::cout << std::endl;

  std::cout << "v: " << std::endl;
  for(auto v_ : v)
    std::cout << v_ << " ";
  std::cout << std::endl;
  std::cout << "scale: " << s[0] << std::endl;

  scaleOptions.minimizer_progress_to_stdout = true;
  scaleOptions.linear_solver_type = ceres::DENSE_QR;
  ceres::Solve(scaleOptions, &scaleProblem, &scaleSummary);
  std::cout << "scale Solver finished" << std::endl;

  std::cout << "final rvec: " << std::endl;
  for(auto r : rvec)
    std::cout << r << " ";
  std::cout << std::endl;

  std::cout << "final v: " << std::endl;
  for(auto v_ : v)
    std::cout << v_ << " ";
  std::cout << std::endl;
  std::cout << "final scale: " << s[0] << std::endl;
}

#endif