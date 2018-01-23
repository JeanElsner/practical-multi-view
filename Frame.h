#ifndef FRAME_H
#define FRAME_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <unordered_map>
#include <memory>
#include "Feature3D.h"

class Frame
{
private:
	bool computed_harris = false, computed_gradient = false;
	cv::Mat harris, grad_x, grad_y;

	void computeSpatialGradient();
	void computeHarrisMatrix();
	void init();

	Frame() { }

public:
	std::unordered_map<Feature, std::weak_ptr<Feature3D>, Feature::Hasher> map;

	// Original image and black and white version
	cv::Mat orig, bw;

	// Frame number in sequence
	int frame;

	/**
		Constructs a frame object based on the given image

		@param orig Image
	*/
	Frame(cv::Mat& orig);

	/**
		Constructs a frame object based on the given image file

		@param file_name File name pointing to an image file
	*/
	Frame(const std::string& file_name);

	/**
		Returns the frame's derivative with regard to x

		@return Spatial gradient matrix
	*/
	cv::Mat& getSpatialGradientX();

	/**
		Returns the frame's derivative with regard to y

		@return Spatial gradient matrix
	*/
	cv::Mat& getSpatialGradientY();

	/**
		Returns this image's harris matrix as a three-channel image
		with the second moments Ixx, Iyy and Ixy stored in channels.

		@param int x Pixel column
		@param int y Pixel row
		@return
	*/
	cv::Mat& getHarrisMatrix();

	/**
		Checks whether this frame is empty

		@return True if frame is empty
	*/
	bool isEmpty() { return orig.empty(); }

	/**
		Returns a region of interest of this frame,
		including all the computed moments etc.

		@param rect A rectangular describing the region of interest
		@return New frame confined to the ROI
	*/
	Frame regionOfInterest(cv::Rect& rect);

	/**
		Count the currently tracked 3D points

		@return Number of tracked points
	**/
	int count3DPoints();

	/**
		Get a 2D feature corresponding to the given 3D point

		@param f3d A 3D feature point
		@return 2D feature
	**/
	Feature get2DFeature(std::weak_ptr<Feature3D>& f3d);
};
#endif