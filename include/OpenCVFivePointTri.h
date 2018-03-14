#ifndef OPEN_CV_FIVE_POINT_TRI_H
#define OPEN_CV_FIVE_POINT_TRI_H
#include "BaseTriangulator.h"
#include "OdometryPipeline.h"

class OpenCVFivePointTri :
	public BaseTriangulator
{
public:
	OdometryPipeline* tracker;

	OpenCVFivePointTri(OdometryPipeline* tracker) : tracker(tracker) {}

	virtual void triangulate(Frame & src, Frame & next, cv::Mat & R, cv::Mat & t);

};

#endif