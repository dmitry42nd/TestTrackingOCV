#pragma once

#include "stdafx.h"

#include "TrackedPoint.h"

class Track
{
public:
  Track();

  std::vector<std::shared_ptr<TrackedPoint>> history;
  std::size_t bestCandidate;
};

