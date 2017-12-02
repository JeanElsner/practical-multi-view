#ifndef FRAME_H
#define FRAME_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

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
	cv::Mat orig, bw;

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
};
#endif