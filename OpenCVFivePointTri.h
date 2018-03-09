#ifndef OPEN_CV_FIVE_POINT_TRI_H
#define OPEN_CV_FIVE_POINT_TRI_H
#include "BaseTriangulator.h"
#include "Tracker.h"

class OpenCVFivePointTri :
	public BaseTriangulator
{
public:
	Tracker* tracker;

	OpenCVFivePointTri(Tracker* tracker) : tracker(tracker) {}

	virtual void triangulate(Frame & src, Frame & next, cv::Mat & R, cv::Mat & t);

};

#endif