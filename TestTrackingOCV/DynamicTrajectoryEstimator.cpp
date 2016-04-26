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
  errOut.open("../data");

}

static const int HIST_LENGTH = 10;
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
  static const cv::Rect roi = cv::Rect(0, 0, 1, 2);
  cv::Mat pr1_ = projMatr*p4D;
  cv::Mat pr1 = (pr1_ / pr1_.at<double>(2, 0))(roi);
  return cv::norm(cv::Vec2d(p.x - pr1.at<double>(0,0), p.y - pr1.at<double>(0,1)));
}


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
  xp = -p[0] / p[2];
  yp = -p[1] / p[2];

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

cv::Point3f getPoint3dCeres(double * p3D)
{
  return cv::Point3d(p3D[0],p3D[1],p3D[2]);
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

  imagePoints.clear();
  for(auto o : hists_) {
    auto p = std::find_if(o.cbegin(), o.cend(),
                          [frameId](const std::shared_ptr<TrackedPoint> obj) { return obj->frameId == frameId; });
    imagePoints.push_back((*p)->undist(K,dist));
  }

  std::vector<double> err;
  for(int j = 0; j < inliers.rows; j++) {
    int i = inliers.row(j).at<int>(0,0);

    cv::Mat point = cv::Mat(objectPoints[i]);
    cv::vconcat(point, cv::Mat::ones(1,1,CV_64F),point);
    //std::cout << point << std::endl;

    cv::Mat wp = owT*point;
    //std::cout << wp.t() << std::endl;
    double x = wp.at<double>(0,0);
    double y = wp.at<double>(0,1);
    double z = wp.at<double>(0,2);
    dataOut << color<< ", " << x << ", " << y << ", " << z << std::endl;
    Xs.push_back(cv::Point3d(x,y,z));

    //reprojection error
    cv::Mat cp = ocT*point;
    projXs_debug.push_back(cv::Point2d(cp.at<double>(0,0)/cp.at<double>(0,2), cp.at<double>(0,1)/cp.at<double>(0,2)));
    //std::cout << cp << std::endl;
    cv::Mat a = cv::Mat(cv::Point2d(cp.at<double>(0,0)/cp.at<double>(0,2), cp.at<double>(0,1)/cp.at<double>(0,2)));
    cv::Mat b = cv::Mat(cv::Point2d(imagePoints[i].x, imagePoints[i].y));
    double projErr = cv::norm(a-b);
    //std::cout << projErr << std::endl;
    errOut << projErr << std::endl;
    err.push_back(projErr);
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
                                                   std::vector<cv::Point2d> &v, histVector &its, int i) {

  histVector its__;
  std::vector<cv::Point2d> vF_, vL_, v_;

  for(decltype(vF.size()) i = 0; i < vF.size(); i++) {
    if (mask.at<char>(0, i)) {
      vF_.push_back(vF[i]);
      vL_.push_back(vL[i]);
      v_.push_back(v[i]);
      its__.push_back(its[i]);
    }
  }

  vF_.swap(vF);
  vL_.swap(vL);
  v_.swap(v);
  its__.swap(its);

  cv::Mat outImg;
  cv::cvtColor(img, outImg, CV_GRAY2BGR);

  for(auto p : v){
    cv::circle(outImg, p, 3, cv::Scalar(0, 0, 200), -1);
  }

  std::string outImgName = std::to_string(i) + ".bmp";
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


void DynamicTrajectoryEstimator::block1(int frameIdF, int frameIdL) {
  std::vector<cv::Point2d>  unPointsF, unPointsL;
  std::vector<cv::Point2d>  pointsF_debug;
  histVector hists;

  std::string ImgName = "../outProc/" + std::to_string(frameIdF) + ".bmp";
  img = cv::imread(ImgName,0);
  cv::Mat outImg;
  cv::cvtColor(img, outImg, CV_GRAY2BGR);

  for (auto track : dynamicTracks) {
    if (track->history.front()->frameId <= frameIdF && track->history.back()->frameId >= frameIdL) {
      auto pF = std::find_if(track->history.cbegin(), track->history.cend(),
                             [frameIdF](const std::shared_ptr<TrackedPoint> obj) { return obj->frameId == frameIdF; });

      auto pL = std::find_if(pF, track->history.cend(),
                             [frameIdL](const std::shared_ptr<TrackedPoint> obj) { return obj->frameId == frameIdL; });

      if (pF != track->history.end() && pL != track->history.end() && std::distance(pF, pL) > HIST_LENGTH) {
        unPointsF.push_back((*pF)->undist(K, dist));
        unPointsL.push_back((*(pF+HIST_LENGTH) )->undist(K, dist));
        hists.push_back(track->history);
        cv::circle(outImg, (*pF)->loc, 3, cv::Scalar(0, 0, 200), -1);
        pointsF_debug.push_back((*pF)->loc);
      }
    }
    std::string outImgName = std::to_string(frameIdF) + ".bmp";
    cv::imwrite(outImgName, outImg);
  }

  //std::cout << "got " << unPointsF.size() << " points for frame pair " << frameId << " - " << frameId + SOME_STEP << std::endl;
  if (hists.size() >= MIN_POINTS) {
    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(unPointsF, unPointsL, 1.0, cv::Point2d(0, 0), cv::RANSAC, 0.99, 0.0005, mask);
    //std::cerr << E << std::endl;

    cv::Mat R, t;
    if (E.rows == 3 && E.cols == 3) {
      cv::recoverPose(E, unPointsF, unPointsL, R, t, 1.0, cv::Point2d(0, 0), mask);
      //t*=10;
      //std::cerr << mask.type() << ": " << mask.t() << std::endl;
      filterByMaskDebug(mask, unPointsF, unPointsL, pointsF_debug, hists, 2);
      if(hists.size() < MIN_POINTS) {
        std::cerr << "algo failed 2\n";
        return;
      }

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

        ceres::CostFunction *cost_function = TriangulateError2::Create(unPointsF[i].x, unPointsF[i].y, cameraF,
                                                                       unPointsL[i].x, unPointsL[i].y, cameraL);
        problem.AddResidualBlock(cost_function, NULL, point);
        points.push_back(point);
      }
      //std::cout << std::endl;

      ceres::Solver::Options options;
      options.linear_solver_type = ceres::DENSE_QR;
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
        //std::cout << mean2Err << std::endl;
        projErrs.push_back(std::make_pair(mean2Err , i));
      }

      //http://stackoverflow.com/questions/19842035/stdmap-how-to-sort-by-value-then-by-key
      std::sort(projErrs.begin(), projErrs.end());

      objectPoints.clear();
      //std::cout << projErrs.size() << std::endl;
      for(auto i = 0; i < projErrs.size(); i++) {
        int pId = projErrs[i].second;
        hists_.push_back(hists[pId]);
        objectPoints.push_back(getPoint3dCeres(points[pId]));

        double *objP = new double[3];
        objP[0] = objectPoints.back().x;
        objP[1] = objectPoints.back().y;
        objP[2] = objectPoints.back().z;
        objectPoints_ceres.push_back(objP);
      }

    } else {
      std::cerr << "five point failed\n";
    }
  }
}


void DynamicTrajectoryEstimator::buildTrack(int frameIdF, int frameIdL) {
      block1(frameIdF, frameIdL);

      std::vector<int> fids_;
      ceres::Problem mainProblem;
      for(int fid = frameIdF; fid < frameIdL; fid++) {

        /*std::string ImgName = "../outProc/" + std::to_string(fid) + ".bmp";
        img = cv::imread(ImgName,1);
        cv::Mat outImg;
        img.copyTo(outImg);*/


        imagePoints.clear();
        for(auto o : hists_) {
          auto p = std::find_if(o.cbegin(), o.cend(),
                                 [fid](const std::shared_ptr<TrackedPoint> obj) { return obj->frameId == fid; });
          //cv::circle(outImg, (*p)->loc, 3, cv::Scalar(0, 0, 200), -1);
          imagePoints.push_back((*p)->undist(K,dist));
        }

        /*std::string outImgName =  "dout/" + std::to_string(fid) + ".bmp";
        cv::imwrite(outImgName, outImg);*/

        cv::Mat rvec, tvec;
        if(imagePoints.size() >= 5)
          cv::solvePnPRansac(objectPoints, imagePoints, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(1, 4, CV_64F), rvec, tvec,
                             false, 100, 0.03, 0.98, inliers, cv::SOLVEPNP_EPNP);
        else if(imagePoints.size() > 3)
          cv::solvePnPRansac(objectPoints, imagePoints, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(1, 4, CV_64F), rvec, tvec,
                             false, 100, 0.03, 0.98, inliers, cv::SOLVEPNP_DLS);
        else
          std::cerr << "algo failed 3\n";

        if(inliers.rows > 0) {
          oldrvec.push_back(rvec); //just for debug
          oldtvec.push_back(tvec);

          std::vector<double> camera;
          //double * camera = new double[6];
          for (auto i = 0; i < 3; i++) {
            //camera[i] = rvec.at<double>(i, 0);
            camera.push_back(rvec.at<double>(i, 0));
          }
          for (auto i = 0; i < 3; i++) {
            //camera[3 + i] = rvec.at<double>(i, 0);
            camera.push_back(tvec.at<double>(i, 0));
          }

          scaleObs.push_back(imagePoints);
          scaleCameras.push_back(camera);

          fids_.push_back(fid);
          scaleInliers.push_back(inliers);

          //std::cout << inliers.type() << " " << inliers.t() << std::endl;
          for (int j = 0; j < inliers.rows; j++) {
            int i = inliers.row(j).at<int>(0, 0);
            ceres::CostFunction *cost_function = TriangulateError3::Create(imagePoints[i].x, imagePoints[i].y);
            mainProblem.AddResidualBlock(cost_function, NULL, &scaleCameras.back()[0], objectPoints_ceres[i]);
          }
        } else {
          std::cout << "no inliers in solveRansac " << fid << "\n";
        }
      }

      ceres::Solver::Options mainOptions;
      mainOptions.minimizer_progress_to_stdout = true;
      mainOptions.linear_solver_type = ceres::DENSE_SCHUR;
      ceres::Solver::Summary mainSummary;
      ceres::Solve(mainOptions, &mainProblem, &mainSummary);

      /*std::cout << "frames chosed: " << fids_.size() << std::endl;
      std::cout << "prev obj points: " << std::endl;
      for(auto p : objectPoints) {
        std::cout << p << std::endl;
      }*/

      //std::cout << "actual obj points: " << std::endl;
      objectPoints.clear();
      for(auto i = 0; i < objectPoints_ceres.size(); i++) {
        cv::Point3d p = cv::Point3d(objectPoints_ceres[i][0],objectPoints_ceres[i][1],objectPoints_ceres[i][2]);
        //std::cout << p << std::endl;
        objectPoints.push_back(p);
      }
      //std::cout << std::endl;

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

        //std::cout << oldtvec[i].t() << std::endl;
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

scaleSolver(scaleObs, scaleCameras, scaleInliers, scaleXsF , scaleXsL);

}

/*
struct Cube{
  cv::Point3d points[8];
}*/

//simple test. outlier test,
void testProgram(){
  return;
}


void DynamicTrajectoryEstimator::scaleSolver(std::vector<std::vector<cv::Point2d>> obs,
                                             std::vector<std::vector<double>> cameras,
                                             std::vector<cv::Mat> inliers,
                                             std::vector<cv::Point3d> XsF , std::vector<cv::Point3d> XsL) {

  ceres::Problem scaleProblem;
  ceres::Solver::Options scaleOptions;
  ceres::Solver::Summary scaleSummary;

  double s[1] = {1.0};
  std::vector<double> rvec;
  rvec.push_back(0);
  rvec.push_back(0);
  rvec.push_back(0);

  std::vector<double> v;

  cv::Point3d meanXF(0,0,0);
  for(auto p : XsF)
    meanXF += p;
  meanXF /= (double)XsF.size();

  cv::Point3d meanXL(0,0,0);
  for(auto p : XsL)
    meanXL += p;
  meanXL /= (double)XsL.size();

  std::cout << meanXF << std::endl;
  std::cout << meanXL << std::endl;
  std::cout << (double)obs.size() << " : " << (Fdebug_ - Ldebug_) << std::endl;
  cv::Point3d V = (meanXL - meanXF) / (Fdebug_ - Ldebug_);
  //std::cout << V << std::endl;

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

  for(int k = 0; k < obs.size() - 1; k++) {
    for(int j = 0; j < inliers[k].rows; j++) {
      int pid = inliers[k].row(j).at<int>(0,0);
      cv::Point2d const& obs_ = obs[k+1][pid];
      cv::Point3d const& X_   = XsF[pid];
      std::vector<double> const& camera_ = cameras[k+1];

      ceres::CostFunction *cost_function = ScaleError::Create(obs_.x, obs_.y, X_.x, X_.y, X_.z, camera_.data(), k+1);
      scaleProblem.AddResidualBlock(cost_function, NULL, s, /*&rvec[0],*/ v.data());
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

  scaleOptions.minimizer_progress_to_stdout = true;
  scaleOptions.linear_solver_type = ceres::DENSE_SCHUR;
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

  std::cout << "scale: " << s[0] << std::endl;
}