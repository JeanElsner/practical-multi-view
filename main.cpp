#include <opencv2/highgui.hpp>
#include <iostream>
#include "Tracker.h"

int main(int argc, char** argv)
{
	if (argc < 2)
		return -1;
	try
	{
		Tracker tracker(argv[1]);
		tracker.startPipeline();

		cv::VideoWriter video;
		video = cv::VideoWriter(tracker.video_path, CV_FOURCC('M', 'J', 'P', 'G'), 10, tracker.frames[0].orig.size());

		for (auto& f : tracker.frames)
			video.write(f.orig);

		video.release();
	}
	catch (Tracker::TrackerException &e)
	{
		std::cerr << e.what() << std::endl << "Tracker configuration failed";
		return -1;
	}
	return 0;
}
