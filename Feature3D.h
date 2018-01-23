#ifndef FEATURE3D_H
#define FEATURE3D_H

#include "Feature.h"

class Feature3D
{
public:
	cv::Point3f p3f;

	/**
		Constructs an object based on the given 3D coordinates
	**/
	Feature3D(double x, double y, double z) : Feature3D(cv::Point3f(x, y, z)) {}

	/**
		Constructs an object based on the given 3D point
	**/
	Feature3D(cv::Point3f p3f) : p3f(p3f) {}

	/**
		Gets the 3d points this feature corresponds to

		@return The cv::Point3f object
	**/
	cv::Point3f getPoint() { return p3f; }

	/**
		Applies a linear transformation

		@param R Rotation matrix
		@param t Translation vector
	**/
	void transform(const cv::Mat& R, const cv::Mat& t);

	/**
		Inverse linear transformation

		@param R Rotation matrix
		@param t Translation vector
	**/
	void transformInv(const cv::Mat & R, const cv::Mat & t);


	/**
		Translates this 3D points

		@param x Translation in x direction
		@param y Translation in y direction
		@param z Translation in z direction
	**/
	void translate(const double x, const double y, const double z);

	/**
		Rotates this 3D point

		@param R Rotation matrix
	**/
	void rotate(const cv::Mat& R);


	/**
		Translates this 3D point

		@param t Translation vector
	**/
	void translate(const cv::Mat& t);

	/**
		Updates the coordinates of this 3D point

		@param x New x coordinate
		@param y New y coordinate
		@param z New z coordinates
	**/
	void update(const double x, const double y, const double z);

	/**
		Translates the given point

		@param t Translation vector
		@param point3d 3D point
	**/
	static void translatePoint(const cv::Mat& t, cv::Point3f& point3d);

	/**
		Rotates the given 3D point

		@param R Rotation matrix
		@param x The points x coordinate
		@param y The points y coordinate
		@param z The points z coordinate
	**/
	static void rotatePoint(const cv::Mat& R, double& x, double& y, double& z);

	/**
		Rotates the given 3D point

		@param R Rotation matrix
		@param point3d The 3D point
	**/
	static void rotatePoint(const cv::Mat& R, cv::Point3f& point3d);

	/**
		Rotates the given 3D point

		@param R Rotation matrix
		@param x The points x coordinate
		@param y The points y coordinate
		@param z The points z coordinate
	**/
	static void rotatePoint(const double * R, double & x, double & y, double & z);


	/**
		Inversively rotates the given 3D point

		@param R Rotation matrix
		@param x The points x coordinate
		@param y The points y coordinate
		@param z The points z coordinate
	**/
	static void rotatePointInverse(const double * R, double & x, double & y, double & z);

	/**
		Projects the given 3D point onto the cameras image plane

		@param R Orientation of the camera pose
		@param t Position of the camera pose
		@param camera Camera matrix
		@param point3d The 3D point to project
		@param point2d The projected point on the image plane
	**/
	static void projectPoint(const cv::Mat& R, const cv::Mat& t, const cv::Mat& camera, const cv::Point3f& point3d, cv::Point2f& point2d);

	/**
		Projects the given 3D point onto the cameras image plane

		@param R Orientation of the camera pose
		@param t Positon of the camera pose
		@param camera Camera matrix
		@param point3d The 3D point to project
		@param point2d The projected point on the image plane
	**/
	static void projectPoint(const double* R, const double* t, const double* camera, const double* point3d, double* point2d);
};

#endif