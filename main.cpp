#include <opencv2/highgui.hpp>
#include <iostream>
#include "OdometryPipeline.h"

int main(int argc, char** argv)
{
	if (argc < 2)
		return -1;
	try
	{
		OdometryPipeline tracker(argv[1]);
		tracker.startPipeline();

		if (tracker.fancy_video)
		{
			cv::VideoWriter video;
			video = cv::VideoWriter(tracker.video_path, CV_FOURCC('M', 'J', 'P', 'G'), 10, tracker.frames[0]->orig.size());

			for (auto& f : tracker.frames)
				video.write(f->orig);

			video.release();
		}
	}
	catch (OdometryPipeline::OdometryPipelineException &e)
	{
		std::cerr << e.what() << std::endl << "Odometry Pipeline configuration failed";
		return -1;
	}
	return 0;
}
