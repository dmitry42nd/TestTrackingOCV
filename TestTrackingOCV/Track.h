#pragma once

#include "stdafx.h"

#include "TrackedPoint.h"

enum PointType {Undef, Static, Dynamic};

class Track
{
public:
  Track();

  PointType type;
  std::vector<std::shared_ptr<TrackedPoint>> history;
  std::shared_ptr<TrackedPoint> bestCandidate;
};

