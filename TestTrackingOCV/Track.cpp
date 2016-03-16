#include "stdafx.h"

#include "Track.h"

Track::Track() : 
  type(Undef),
  bestCandidate(std::make_shared<TrackedPoint>())
{	
}