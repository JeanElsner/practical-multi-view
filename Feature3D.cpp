#include "Feature3D.h"

void Feature3D::projectPoint(const cv::Mat& R, const cv::Mat& t, const cv::Mat& camera, const cv::Point3f& point3d, cv::Point2f& point2d)
{
	cv::Point3f p3f = point3d;
	
	translatePoint(-t, p3f);
	cv::transpose(R, R);
	rotatePoint(R, p3f);
	p3f.z *= -1;

	double magic_z = p3f.z ? 1. / p3f.z : 1;

	point2d.x = p3f.x * magic_z * camera.at<double>(0, 0) + camera.at<double>(0, 2);
	point2d.y = p3f.y * magic_z * camera.at<double>(1, 1) + camera.at<double>(1, 2);
}

void Feature3D::projectPoint(
	const double * R, const double * t, const double * camera, const double * point3d, double * point2d)
{
	double x_p = point3d[0], y_p = point3d[1], z_p = point3d[2];

	x_p -= t[0];
	y_p -= t[1];
	z_p -= t[2];
	rotatePointInverse(R, x_p, y_p, z_p);
	z_p *= -1;

	double magic_z = z_p ? 1. / z_p : 1;

	point2d[0] = x_p * magic_z * camera[0] + camera[2];
	point2d[1] = y_p * magic_z * camera[4] + camera[5];
}

void Feature3D::rotatePoint(const cv::Mat& R, cv::Point3f& point3d)
{
	double x = point3d.x, y = point3d.y, z = point3d.z;
	rotatePoint(R, x, y, z);
	point3d.x = x;
	point3d.y = y;
	point3d.z = z;
}

void Feature3D::rotatePoint(const cv::Mat& R, double& x, double& y, double& z)
{
	double x_R, y_R, z_R;
	const double* ptr = R.ptr<double>(0);
	x_R = ptr[0] * x + ptr[1] * y + ptr[2] * z;
	ptr = R.ptr<double>(1);
	y_R = ptr[0] * x + ptr[1] * y + ptr[2] * z;
	ptr = R.ptr<double>(2);
	z_R = ptr[0] * x + ptr[1] * y + ptr[2] * z;

	x = x_R;
	y = y_R;
	z = z_R;
}

void Feature3D::rotatePoint(const double* R, double& x, double& y, double& z)
{
	double x_R, y_R, z_R;

	x_R = R[0] * x + R[1] * y + R[2] * z;
	y_R = R[3] * x + R[4] * y + R[5] * z;
	z_R = R[6] * x + R[7] * y + R[8] * z;

	x = x_R;
	y = y_R;
	z = z_R;
}

void Feature3D::rotatePointInverse(const double* R, double& x, double& y, double& z)
{
	double x_R, y_R, z_R;

	x_R = R[0] * x + R[3] * y + R[6] * z;
	y_R = R[1] * x + R[4] * y + R[7] * z;
	z_R = R[2] * x + R[5] * y + R[8] * z;

	x = x_R;
	y = y_R;
	z = z_R;
}

void Feature3D::transform(const cv::Mat & R, const cv::Mat & t)
{
	rotate(R);
	translate(t);
}

void Feature3D::transformInv(const cv::Mat& R, const cv::Mat& t)
{
	cv::Mat inv;
	cv::transpose(R, inv);
	translate(-t);
	rotate(inv);
}

void Feature3D::translate(const double x, const double y, const double z)
{
	update(p3f.x + x, p3f.y + y, p3f.z + z);
}

void Feature3D::translate(const cv::Mat& t)
{
	p3f.x += t.at<double>(0);
	p3f.y += t.at<double>(1);
	p3f.z += t.at<double>(2);
}

void Feature3D::update(const double x, const double y, const double z)
{
	p3f.x = x;
	p3f.y = y;
	p3f.z = z;
}

void Feature3D::translatePoint(const cv::Mat& t, cv::Point3f& point3d)
{
	point3d.x += t.at<double>(0);
	point3d.y += t.at<double>(1);
	point3d.z += t.at<double>(2);
}

void Feature3D::rotate(const cv::Mat& R)
{
	double x_0, y_0, z_0;

	const double* ptr = R.ptr<double>(0);
	x_0 = ptr[0] * p3f.x + ptr[1] * p3f.y + ptr[2] * p3f.z;
	ptr = R.ptr<double>(1);
	y_0 = ptr[0] * p3f.x + ptr[1] * p3f.y + ptr[2] * p3f.z;
	ptr = R.ptr<double>(2);
	z_0 = ptr[0] * p3f.x + ptr[1] * p3f.y + ptr[2] * p3f.z;

	p3f.x = x_0;
	p3f.y = y_0;
	p3f.z = z_0;
}
