#ifndef BASE_TRIANGULATOR_H
#define BASE_TRIANGULATOR_H
#include "Frame.h"

// Abstract base class, used to triangulate 2D feature correspondences between frames
class BaseTriangulator
{
public:

	/**
		Triangulates new 3D points from the 2D feature correspondences
		found in the two frames and estimates the pose between them.
		Additionally, may remove 3D point outliers.

		@param src First image
		@param next Second image
		@param R Estimated rotation
		@param t Estimated translation
	*/
	virtual void triangulate(Frame& src, Frame& next, cv::Mat& R, cv::Mat& t) = 0;
};
#endif
