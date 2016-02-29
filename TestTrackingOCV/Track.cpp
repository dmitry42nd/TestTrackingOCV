#include "stdafx.h"

#include "Track.h"

Track::Track() : 
  myBC(std::make_shared<TrackedPoint>()),
  bestCandidate(NULL)
{	
}