#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include "Feature.h"
#include "Frame.h"

using namespace cv;
using namespace std;

unsigned char GRID_SIZE[2] = { 255, 255 };

std::vector<double> ticktock;

bool operator== (const Feature& lhs, const Feature& rhs)
{
	if (lhs.row == rhs.row && lhs.column == rhs.column)
		return true;
	return false;
}

bool operator!= (const Feature& lhs, const Feature& rhs)
{
	return !(lhs == rhs);
}

/**
	Starts a timer, will be tiered if called multiple times
*/
void tick()
{
	ticktock.push_back(getTickCount());
}

/**
	Returns the time since the last tick in seconds

	@return Time since last call to tick
*/
double tock()
{
	if (ticktock.empty())
		return 0;
	double tock = ticktock.back();
	ticktock.pop_back();
	return (getTickCount() - tock) / getTickFrequency();
}

/**
	Checks whether a feature has any neighbors within
	a given (Manhattan) distance.

	@param f The feature to check
	@param feats A list of features to check against
	@param distance The distance in pixels
	@returns True if a neigihbor was found, false otherwise
*/
bool has_neighbor(const Feature f, const vector<Feature> feats, int distance = 3)
{
	for (auto const& ff : feats)
	{
		if (abs(ff.row - f.row) < distance && abs(ff.column - f.column) < distance)
			return true;
	}
	return false;
}

/**
	Divides the image into a grid, so as to distribute
	the calculated features across the entire image.

	@param I Input image
	@param size The size of the grid blocks in pixels
	@param fn The number of features per block
	@param extr Vector of functions used to extract features
*/
vector<Feature> grid_feature_extraction(Frame& I, const unsigned char size[2], const int fn,
	vector<vector<Feature>(*)(const Mat&, const Mat&, int row_offset, int col_offset)> extr)
{
	int gr = I.bw.rows / size[0];
	int gc = I.bw.cols / size[1];
	vector<Feature> feats;

	for (int r = 0; r < I.bw.rows; r += size[0])
	{
		for (int c = 0; c < I.bw.cols; c += size[1])
		{
			Rect rec(c, r, min((int)size[1], I.bw.cols - c), min((int)size[0], I.bw.rows - r));
			Mat roi = I.bw(rec);
			Mat roi_H = I.getHarrisMatrix()(rec);
			
			for (auto const& func : extr)
			{
				int count = 0;
				vector<Feature> new_feats = func(roi, roi_H, r, c);
				for (auto const& f : new_feats)
				{
					if (count >= fn)
						break;
					if (!has_neighbor(f, feats))
					{
						count++;
						feats.push_back(f);
					}
				}
			}
		}
	}
	return feats;
}

void FAST_corner_detector(Mat& I)
{
	// TODO
}

// TODO: doc
void gaussian_window_3x3(const Mat& src, Mat& dst)
{
	Mat kernel = (Mat_<double>(3, 3) << 1. / 16., 1. / 8., 1. / 16., 1. / 8., 1. / 4., 1. / 8., 1. / 16., 1. / 8., 1. / 16);
	filter2D(src, dst, -1, kernel);
	return;

	for (int r = 1; r < src.rows - 1; r++)
	{
		const double* prev = src.ptr<double>(r - 1);
		const double* curr = src.ptr<double>(r);
		const double* next = src.ptr<double>(r + 1);
		float* p_dst = dst.ptr<float>(r);

		for (int c = 1; c < src.cols - 1; c++)
		{
			p_dst[c] = 1. / 8. * (prev[c - 1] + prev[c + 1] + next[c - 1] + next[c + 1])
				+ 1. / 16. * (prev[c] + curr[c - 1] + curr[c + 1] + next[c]) + 1. / 4. * (curr[c]);
		}
	}
}

// TODO: doc
void compute_shi_tomasi_response(const Mat& H, Mat& dst, float quality = .4)
{
	for (int r = 0; r < H.rows; r++)
	{
		const float* p_H = H.ptr<float>(r);
		float* p_dst = dst.ptr<float>(r);

		for (int c = 0; c < (H.cols-1)*3; c +=3)
		{
			float Ixx = p_H[c + 0];
			float Iyy = p_H[c + 1];
			float Ixy = p_H[c + 2];

			float B = -Ixx - Iyy;
			float C = Ixx*Iyy - pow(Ixy, 2);

			float lambda_1 = (-B + sqrt(pow(B, 2) - 4 * C)) / 2;
			float lambda_2 = (-B - sqrt(pow(B, 2) - 4 * C)) / 2;

			//p_dst[c / 3] = lambda_1*lambda_2 - 0.04f*pow(lambda_1 +lambda_2, 2);
			p_dst[c / 3] = min(lambda_1, lambda_2);
		}
	}
	double r_min, r_max;
	minMaxLoc(dst, &r_min, &r_max);
	threshold(dst, dst, r_max*quality, 255, CV_THRESH_BINARY);
}

// TODO: doc
vector<Feature> shi_tomasi_detector(const Mat& src, const Mat& H, int row_offset=0, int col_offset=0)
{
	Mat R(src.size(), CV_32FC1);
	compute_shi_tomasi_response(H, R);
	vector<Feature> feats;
	
	for (int j = 0; j < src.rows; j++)
	{
		float* p_R = R.ptr<float>(j);

		for (int i = 0; i < src.cols; i++)
		{
			if (p_R[i] == 255)
			{
				Feature f;
				f.row = j + row_offset;
				f.column = i + col_offset;
				f.detector = Feature::extractor::shi_tomasi;
				feats.push_back(f);
			}
		}
	}
	return feats;
}

// TODO: doc
vector<Feature> opencv_good_features(const Mat& src, const Mat& H, int row_offset = 0, int col_offset = 0)
{
	vector<Point2f> corners;
	goodFeaturesToTrack(src, corners, 100, 0.01, 5, Mat(), 3, 3, false, 0.04);

	vector<Feature> feats;
	for (auto const& c : corners)
	{
		Feature f;
		f.row = c.y+row_offset;
		f.column = c.x + col_offset;
		f.detector = Feature::extractor::cv_good;
		feats.push_back(f);
	}
	return feats;
}

// TODO: doc
uchar bilinear_subpixel(const Mat& src, float sub_x, float sub_y)
{
	int x = (int)sub_x;
	int y = (int)sub_y;
	float a = sub_x - (float)x;
	float c = sub_y - (float)y;

	int x0 = borderInterpolate(x, src.cols, cv::BORDER_REFLECT_101);
	int y0 = borderInterpolate(y, src.rows, cv::BORDER_REFLECT_101);
	int x1 = borderInterpolate(x + 1, src.cols, cv::BORDER_REFLECT_101);
	int y1 = borderInterpolate(y + 1, src.rows, cv::BORDER_REFLECT_101);

	return (uchar)cvRound((src.at<char>(y0, x0) * (1.f - a) + src.at<char>(y0, x1) * a) * (1.f - c)
		+ (src.at<char>(y1, x0) * (1.f - a) + src.at<char>(y1, x1) * a) * c);
}

// TODO: doc
void track_features(Frame& f_src, const Frame& f_next, const vector<Feature>& feats, vector<Feature>& new_feats, int window = 15, int iter = 5)
{
	vector<Mat> grad;
	grad.push_back(f_src.getSpatialGradientX());
	grad.push_back(f_src.getSpatialGradientY());
	
	int _win = ceil((float)window / 2.f);

	for (auto const& f : feats)
	{
		double v[2] = { .0, .0 };
		Mat G = Mat::zeros(2, 2, CV_64FC1);

		for (schar x = -_win; x < _win+1; x++)
		{
			for (schar y = -_win; y < _win+1; y++)
			{
				if (f.column + x < 0 || f.column + x >= f_src.bw.cols
					|| f.row + y < 0 || f.row + y >= f_src.bw.rows)
					continue;
				G.at<double>(Point(0, 0)) += pow(grad[0].at<double>(Point(f.column + x, f.row + y)), 2);
				G.at<double>(Point(1, 1)) += pow(grad[1].at<double>(Point(f.column + x, f.row + y)), 2);
				G.at<double>(Point(1, 0)) += (grad[0].at<double>(Point(f.column + x, f.row + y))
					* grad[1].at<double>(Point(f.column + x, f.row + y)));
				G.at<double>(Point(0, 1)) += (grad[0].at<double>(Point(f.column + x, f.row + y))
					* grad[1].at<double>(Point(f.column + x, f.row + y)));
				/*G.at<double>(Point(0, 0)) += H.at<Vec3f>(Point(f.column + x, f.row + y))[0];
				G.at<double>(Point(1, 1)) += H.at<Vec3f>(Point(f.column + x, f.row + y))[1];
				G.at<double>(Point(1, 0)) += H.at<Vec3f>(Point(f.column + x, f.row + y))[2];
				G.at<double>(Point(0, 1)) += H.at<Vec3f>(Point(f.column + x, f.row + y))[2];*/
			}
		}
		Mat b = Mat::zeros(2, 1, CV_64FC1);
		Mat err;
		for (size_t j = 0; j < iter; j++)
		{
			for (schar x = -_win; x < _win+1; x++)
			{
				for (schar y = -_win; y < _win+1; y++)
				{
					if (f.column + x + v[0] < 0 || f.column + x + v[0] + 1 > f_src.bw.cols
						|| f.row + y + v[1] < 0 || f.row + y + v[1] + 1 > f_src.bw.rows)
						continue;
					if (f.column + x < 1 || f.column + x + 1 >= f_next.bw.cols
						|| f.row + y < 1 || f.row + y + 1 >= f_next.bw.rows)
						continue;

					/*double Ix = (double)src.at<uchar>(Point(f.column + x + 1, f.row + y))
						- (double)src.at<uchar>(Point(f.column + x - 1, f.row + y));
					double Iy = (double)src.at<uchar>(Point(f.column + x, f.row + y  + 1))
						- (double)src.at<uchar>(Point(f.column + x, f.row + y - 1));
					Ix *= 1. / 2.;
					Iy *= 1. / 2.;*/

					b.at<double>(Point(0, 0)) += (((double)f_src.bw.at<uchar>(Point(f.column + x, f.row + y)) -
						//(double)next.at<uchar>(Point(f.column + x + v[0], f.row + y + v[1]))) * 
						(double)bilinear_subpixel(f_next.bw, (float)f.column + (float)x + v[0], (float)f.row + (float)y + v[1])) *
						grad[0].at<double>(Point(f.column + x, f.row + y)));
						//(double)bilinear_subpixel(next, (float)f.column + (float)x + v[0], (float)f.row + (float)y + v[1])) * Ix;
						//sqrt(H.at<Vec3f>(Point(f.column + x, f.row + y))[0]);
						//H.at<Vec3f>(Point(f.column + x, f.row + y))[0];
					b.at<double>(Point(0, 1)) += (((double)f_src.bw.at<uchar>(Point(f.column + x, f.row + y)) -
						//(double)next.at<uchar>(Point(f.column + x + v[0], f.row + y + v[1]))) *
						(double)bilinear_subpixel(f_next.bw, (float)f.column + (float)x + v[0], (float)f.row + (float)y + v[1])) *
						grad[1].at<double>(Point(f.column + x, f.row + y)));
						//(double)bilinear_subpixel(next, (float)f.column + (float)x + v[0], (float)f.row + (float)y + v[1])) * Iy;
						//sqrt(H.at<Vec3f>(Point(f.column + x, f.row + y))[1]);sqrt(H.at<Vec3f>(Point(f.column + x, f.row + y))[1]);
						//H.at<Vec3f>(Point(f.column + x, f.row + y))[1];
				}
			}
			Mat G_inv = Mat::zeros(Size(2, 2), G.type());
			double det = G.at<double>(Point(0, 0))*G.at<double>(Point(1, 1))
				- pow(G.at<double>(Point(1, 0)), 2);
			
			if (det != 0)
			{
				G_inv.at<double>(Point(0, 0)) = 1. / det*G.at<double>(Point(1, 1));
				G_inv.at<double>(Point(1, 0)) = -1. / det*G.at<double>(Point(1, 0));
				G_inv.at<double>(Point(0, 1)) = -1. / det*G.at<double>(Point(0, 1));
				G_inv.at<double>(Point(1, 1)) = 1. / det*G.at<double>(Point(0, 0));
			}
			err = G_inv*b;
			
			v[0] += err.at<double>(Point(0, 0));
			v[1] += err.at<double>(Point(0, 1));
		}
		Feature corr;
		if ((pow(err.at<double>(Point(0, 0)), 2) + pow(err.at<double>(Point(0, 1)), 2)) > 1)
			corr.tracked = false;
		else
			corr.tracked = true;
		corr.column = (int)round((double)f.column + v[0]);
		corr.row = (int)round((double)f.row + v[1]);
		new_feats.push_back(corr);

	}
}

// TODO: doc
void scale_feats(vector<Feature>& feats, float scale)
{
	for (auto & f : feats)
	{
		f.column = (int)((float)f.column*scale);
		f.row = (int)((float)f.row*scale);
	}
}

// TODO: doc
vector<Feature> knn_features(Feature& f, const vector<Feature>& feats, int n = 7)
{
	vector<Feature> nearest_vec;
	Feature nearest;
	
	for (int k = 0; k < n; k++)
	{
		float dist = 0;

		for (auto const& ff : feats)
		{
			if (f != ff)
			{
				bool b = true;

				for (auto const& fff : nearest_vec)
				{
					if (ff == fff)
						b = false;
				}
				if (b)
				{
					float _dist = f.distance(ff);

					if (_dist < dist || dist == 0)
					{
						dist = _dist;
						nearest = ff;
					}
				}
			}
		}
		nearest_vec.push_back(nearest);
	}
	return nearest_vec;
}

// TODO: doc
float compare_features(const Mat& src, int src_x, int src_y, const Mat& cmp, int cmp_x, int cmp_y, int window)
{
	int _win = ceil((float)window / 2.f);
	float err = 0;

	for (schar x = -_win; x < _win + 1; x++)
	{

		for (schar y = -_win; y < _win + 1; y++)
		{
			if (src_x + x < 0 || src_y + y < 0 || cmp_x + x < 0 || cmp_y + y < 0
				|| src_x + x >= src.cols || src_y + y >= src.rows || cmp_x + x >= cmp.cols || cmp_y + y >= cmp.rows)
				continue;
			err += pow((float)src.at<uchar>(Point(src_x + x, src_y + y)) - (float)cmp.at<uchar>(Point(cmp_x + x, cmp_y + y)), 2);
		}
	}
	err = sqrt(err)/(pow(window, 2));
	return err;
}

// TODO: doc
void knn_tracker(const Frame& f_src, Frame& f_cmp, vector<Feature> feats, 
	vector<Feature>& new_feats, vector<bool>& tracked, int window = 31, float threshold = 2)
{
	vector<vector<Feature>(*)(const Mat&, const Mat&, int row_offset, int col_offset)> funcs;
	//funcs.push_back(shi_tomasi_detector);
	funcs.push_back(opencv_good_features);

	vector<Feature> cmp_feats = grid_feature_extraction(f_cmp, GRID_SIZE, 35, funcs);

	for (auto & f : feats)
	{
		vector<Feature> nn = knn_features(f, cmp_feats);
		Feature best_fit;
		float err = 0;

		for (auto const& ff : nn)
		{
			float _err = compare_features(f_src.bw, f.column, f.row, f_cmp.bw, ff.column, ff.row, window);

			if (_err < err || err == 0)
			{
				err = _err;
				best_fit = ff;
			}
		}
		if (err < threshold)
			tracked.push_back(true);
		else
			tracked.push_back(false);
		new_feats.push_back(best_fit);
	}
}

int main(int argc, char** argv)
{
	if (argc < 2)
		return -1;
	cv::String path(argv[1]);
	vector<cv::String> fn;
	cv::glob(path, fn, true);

	vector<Frame> I;
	vector<Mat> G;
	vector<Feature> feats;

	VideoWriter video;

	tick();
	for (size_t k = 0; k < fn.size(); ++k)
	{
		//if (k > 2) break;

		Frame frame(fn[k]);
		
		if (frame.isEmpty())
			continue;

		I.push_back(frame);
		
		vector<vector<Feature>(*)(const Mat&, const Mat&, int row_offset, int col_offset)> funcs;
		funcs.push_back(shi_tomasi_detector);
		funcs.push_back(opencv_good_features);

		/*
		Mat H(im.size(), CV_32FC3);
		compute_harris_matrix(im, H);
		G.push_back(H);
		feats = grid_feature_extraction(im, H, GRID_SIZE, 50, funcs);

		for (auto & f : feats)
		{
			circle(imc, f.point(), 5, Scalar(0, 255, 255));
		}
		imshow("test", imc);
		waitKey(20);*/
		

		// TODO: put these three section into trackers
		
		if (k == 0)
		{
			Mat H = frame.getHarrisMatrix();
			G.push_back(H);
			feats = grid_feature_extraction(frame, GRID_SIZE, 35, funcs);

			video = VideoWriter("../../my-feats/tracker.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, frame.bw.size());
		}
		
		//////////////////////////////
		// My knn tracker
		//////////////////////////////
		/*
		if (k > 0)
		{
			vector<feature> new_feats;
			vector<bool> tracked;

			knn_tracker(I[k - 1], I[k], feats, new_feats, tracked);

			for (int i = 0; i < new_feats.size(); i++)
			{
				if (!tracked[i])
					continue;

				circle(imc, new_feats[i].point(), 5, Scalar(0, 255, 255));
				line(imc, feats[i].point(), new_feats[i].point(), Scalar(0, 255, 255));
			}
			Mat H(im.size(), CV_32FC3);
			compute_harris_matrix(im, H);
			G.push_back(H);
			feats = grid_feature_extraction(im, H, GRID_SIZE, 35, funcs);

			imshow("test", imc);
			waitKey(20);
			video.write(imc);
		}*/
		

		////////////////////////////////
		// Opencv lukas-kanade tracker
		////////////////////////////////
		
		vector<Point2f> points, next_points;

		for (auto const& f : feats)
		{
			points.push_back(Point2f(f.column, f.row));
		}
		
		if (k > 0)
		{
			vector<uchar> status;
			vector<float> err;
			calcOpticalFlowPyrLK(I[k - 1].bw, I[k].bw, points, next_points, status, err, Size(31, 31), 8);
			
			for (int i = 0; i < next_points.size(); i++)
			{
				if (status[i])
				{
					circle(frame.orig, next_points[i], 5, Scalar(0, 255, 255));
					line(frame.orig, points[i], next_points[i], Scalar(0, 255, 255));
					points.push_back(next_points[i]);
				}
			}
			feats.clear();

			for (auto const& p : next_points)
			{
				feats.push_back(Feature(p.x, p.y));
			}
		}
		imshow("test", frame.orig);
		video.write(frame.orig);
		waitKey(20);
		
		
		//////////////////////////////////
		// my klt tracker
		//////////////////////////////////
		/*Mat H(im.size(), CV_32FC3);
		compute_harris_matrix(im, H);
		G.push_back(H);
		if (k > 0)
		{
			vector<feature> new_feats, old_feats;
			old_feats = feats;

			for (int j = 2; j >= 0; j--)
			{
				Mat src, next, H;
				next = Mat::zeros(I[k].size() / 2, I[k].type());

				resize(I[k], next, I[k].size() / (int)pow(2, j));
				resize(I[k-1], src, I[k-1].size() / (int)pow(2, j));
				resize(G[k-1], H, G[k-1].size() / (int)pow(2, j));
				
				scale_feats(feats, 1.f / pow(2, j));
				track_features(src, next, feats, new_feats, 31);
				
				scale_feats(feats, pow(2, j));
				scale_feats(new_feats, pow(2, j));

				if (j > 0)
				{
					feats = new_feats;
					new_feats.clear();
				}
			}

			for (int i = 0; i < new_feats.size(); i++)
			{
				if (!new_feats[i].tracked)
					continue;
				circle(imc, Point(new_feats[i].column, new_feats[i].row), 5, Scalar(0, 255, 255));
				arrowedLine(imc, new_feats[i].point(), old_feats[i].point(), Scalar(0, 255, 255));
			}
			feats.clear();

			for (auto const& f : new_feats)
			{
				feats.push_back(f);
			}
			imshow("test", imc);
			waitKey(20);
		}*/

		/*for (auto const& f : feats)
		{
			switch (f.detector)
			{
			case DETECTOR_SHITOMASI:
				circle(imc, Point(f.column, f.row), 5, Scalar(0, 255, 255));
				break;
			case DETECTOR_CVGOOD:
				circle(imc, Point(f.column, f.row), 5, Scalar(0, 0, 255));
				break;
			}
		}*/
		stringstream f;
		f << "../../my-feats/" << k << ".png";
		string ff; f >> ff;
		cv::imwrite(ff, frame.orig);
	}
	video.release();
	cout << tock();
	waitKey();
	return 0;
}