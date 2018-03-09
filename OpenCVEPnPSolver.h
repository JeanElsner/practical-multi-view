#ifndef OPEN_CV_EPNP_SOLVER_H
#define OPEN_CV_EPNP_SOLVER_H
#include "BasePnPSolver.h"
#include "Tracker.h"

class OpenCVEPnPSolver :
	public BasePnPSolver
{
public:
	Tracker* tracker;

	OpenCVEPnPSolver(Tracker* tracker) : tracker(tracker) {}

	virtual void solvePnP(Frame & src, Frame & next, cv::Mat & R, cv::Mat & t);

};

#endif