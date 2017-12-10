#ifndef TRACKER_H
#define TRACKER_H
#include "Feature.h";
#include "Frame.h"
#include "BaseFeatureExtractor.h"
#include "BaseFeatureMatcher.h"
#include <vector>
#include <exception>

class Tracker
{
private:
	bool init = false;

public:
	int tracked_features = 0;
	int min_tracked_features = 3000;
	int tracked_features_tol = 2000;
	int init_frames = 5;
	int grid_size[2] = {255, 255};

	bool verbose = true;
	bool fancy_video = false;

	std::vector<cv::String> file_names;
	std::vector<int> timestamps;

	// Camera matrix
	cv::Mat camera = cv::Mat_<double>(3, 3);

	cv::Mat map = cv::Mat::zeros(512, 512, CV_8UC3);

	std::vector<Feature> features;
	std::vector<Frame> frames;

	BaseFeatureExtractor* extractor;
	BaseFeatureMatcher* matcher;

	std::vector<cv::Mat> t, R, gt_t, gt_R;

	cv::Mat curr_t, curr_R;

	std::vector<double> ticktock;

	/**
	Starts a timer, will be tiered if called multiple times
	*/
	void tick() { ticktock.push_back(cv::getTickCount()); }

	/**
	Returns the time since the last tick in seconds

	@return Time since last call to tick
	*/
	double tock();

	class GridSection
	{
	public:
		// Coordinates within the grid
		int x, y;

		Frame frame;

		std::vector<Feature> features;

		GridSection(Frame frame, int x, int y): frame(frame), x(x), y(y) { }
	};

	class TrackerException : public std::exception
	{
	public:
		TrackerException(const char* msg): std::exception(msg) { }
	};

	/**
		Constructor, configurating the object
		based on a given configuration file.

		@param cfg Path to the configuration file
	*/
	Tracker(std::string cfg);

	Tracker() {};

	/**
		Adds a frame to the tracker extracting new
		features as needed.

		@param frame Frame to add
	*/
	void addFrame(Frame& frame);

	/**
		Initialises the tracker. That is, the frame with the
		best selection of features is chosen as a starting point
		for the tracker.
	*/
	void initialise();

	/**
		Divides the frame into a grid of regions of interest
		according to this object's setting.

		@param fr The source frame to be divided
		@return A vector containing the grid
	*/
	std::vector<GridSection> getGridROI(Frame& fr);

	/**
		Compute the standard deviation of a list of values

		@param val List of values
		@return The standard deviation
	*/
	double standardDeviation(std::vector<double> val);


	/**
		Parses a KITTI calibration file for the camera matrix

		@param filename Calibration file name
		@param num_calib The number (identifier) of the camera matrix
	*/
	void parseCalibration(std::string filename, int num_calib);

	/**
		Parses a KITTI pose file containing the ground truth

		@param filename Pose file name
	*/
	void parsePoses(std::string filename);

	/**
		Splits a string at the given delimeter

		@param string The string to split
		@param delim The delimeter to split by
		@return A list of the tokens
	*/
	std::vector<std::string> split(const std::string& str, const std::string& delim = " ");

	/**
		Start the visual odometry process
	*/
	void start();

private:
	void countTrackedFeatures();
};
#endif
