#include "stdafx.h"

#include "Track.h"

Track::Track() : 
  bestCandidate(std::make_shared<TrackedPoint>())
{	
}