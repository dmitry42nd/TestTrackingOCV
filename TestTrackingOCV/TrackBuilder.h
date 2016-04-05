//
// Created by dmitry on 4/5/16.
//
#ifndef TESTTRACKINGOCV_TRACKBUILDER_H
#define TESTTRACKINGOCV_TRACKBUILDER_H

#include "Track.h"
#include "TrajectoryArchiver.h"

class TrackBuilder {
public:
  void TrackBuilder::loadAllTracks(std::string &pathToAllTracks);

protected:
  std::vector<std::shared_ptr<Track>> tracks;

};


#endif //TESTTRACKINGOCV_TRACKBUILDER_H
