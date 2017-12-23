#ifndef FEATURE3D_H
#define FEATURE3D_H

#include "Feature.h"

class Feature3D
{
public:
	cv::Point3f p3f;

	int feature;

	/**
		Constructs an object based on the given
		3d space coordinates and corresponding 2d feature
	**/
	Feature3D(double x, double y, double z, int feature) : Feature3D(cv::Point3f(x, y, z), feature) {}

	/**
		Constructs an object based on the given 3d point
		and corresponding 2d feature
	**/
	Feature3D(cv::Point3f p3f, int feature) : p3f(p3f), feature(feature) {}

	/**
		Gets the 3d points this feature corresponds to

		@return The cv::Point3f object
	**/
	cv::Point3f getPoint() { return p3f; }

	void transform(const cv::Mat& R, const cv::Mat& t);

	void transformInv(const cv::Mat & R, const cv::Mat & t);

	void rotate(const cv::Mat& R);

	void translate(const cv::Mat& t);

	static void translatePoint(const cv::Mat& t, cv::Point3f& point3d);

	static void rotatePoint(const cv::Mat& R, double& x, double& y, double& z);

	static void rotatePoint(const cv::Mat& R, cv::Point3f&);

	static void rotatePoint(const double * R, double & x, double & y, double & z);

	static void projectPoint(const cv::Mat& R, const cv::Mat& t, const cv::Mat& camera, const cv::Point3f& point3d, cv::Point2f& point2d);

	static void projectPoint(const double* R, const double* t, const double* camera, const double* point3d, double* point2d);
};

#endif