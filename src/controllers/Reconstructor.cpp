/*
 * Reconstructor.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */
#include "Reconstructor.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <cassert>
#include <iostream>
#include <opencv2/imgproc.hpp> // drawing shapes
#include <numeric>

#include "../utilities/General.h"

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

	vector<Point2f> voxels2dCoordinates;
	vector<Point2f> voxelProjected0;
	vector<Point3f> labelledVoxelProjected0;
	vector<Point3f> labelledVoxelProjected1;
	vector<Point2f> voxelProjected1;
	vector<vector<int>> visibleVoxelLabel;
	vector<Point2f> centers;
	vector<int> labels;
	vector<Mat> offlineHistograms0;
	vector<Mat> offlineHistograms1;
	vector<Mat> histograms0;
	vector<Mat> histograms1;
	int frame = 0;
	bool online = true;

/**
 * Constructor
 * Voxel reconstruction class
 */
Reconstructor::Reconstructor(
		const vector<Camera*> &cs) :
				m_cameras(cs),
				m_height(2560),
				m_step(32)
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_plane_size.area() > 0)
			assert(m_plane_size.width == m_cameras[c]->getSize().width && m_plane_size.height == m_cameras[c]->getSize().height);
		else
			m_plane_size = m_cameras[c]->getSize();
	}

	const size_t edge = 2 * m_height;
	m_voxels_amount = (edge / m_step) * (edge / m_step) * (m_height / m_step);

	initialize();
}

/**
 * Deconstructor
 * Free the memory of the pointer vectors
 */
Reconstructor::~Reconstructor()
{
	for (size_t c = 0; c < m_corners.size(); ++c)
		delete m_corners.at(c);
	for (size_t v = 0; v < m_voxels.size(); ++v)
		delete m_voxels.at(v);
}

/**
 * Create some Look Up Tables
 * 	- LUT for the scene's box corners
 * 	- LUT with a map of the entire voxelspace: point-on-cam to voxels
 * 	- LUT with a map of the entire voxelspace: voxel to cam points-on-cam
 */
void Reconstructor::initialize()
{
	// Cube dimensions from [(-m_height, m_height), (-m_height, m_height), (0, m_height)]
	const int xL = -m_height;
	const int xR = m_height;
	const int yL = -m_height;
	const int yR = m_height;
	const int zL = 0;
	const int zR = m_height;
	const int plane_y = (yR - yL) / m_step;
	const int plane_x = (xR - xL) / m_step;
	const int plane = plane_y * plane_x;

	// Save the 8 volume corners
	// bottom
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zL));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zL));

	// top
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zR));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zR));

	// Acquire some memory for efficiency
	cout << "Initializing " << m_voxels_amount << " voxels ";
	m_voxels.resize(m_voxels_amount);

	int z;
	int pdone = 0;
#pragma omp parallel for schedule(static) private(z) shared(pdone)
	for (z = zL; z < zR; z += m_step)
	{
		const int zp = (z - zL) / m_step;
		int done = cvRound((zp * plane / (double) m_voxels_amount) * 100.0);

#pragma omp critical
		if (done > pdone)
		{
			pdone = done;
			cout << done << "%..." << flush;
		}

		int y, x;
		for (y = yL; y < yR; y += m_step)
		{
			const int yp = (y - yL) / m_step;

			for (x = xL; x < xR; x += m_step)
			{
				const int xp = (x - xL) / m_step;

				// Create all voxels
				Voxel* voxel = new Voxel;
				voxel->x = x;
				voxel->y = y;
				voxel->z = z;
				voxel->camera_projection = vector<Point>(m_cameras.size());
				voxel->valid_camera_projection = vector<int>(m_cameras.size(), 0);

				const int p = zp * plane + yp * plane_x + xp;  // The voxel's index

				for (size_t c = 0; c < m_cameras.size(); ++c)
				{
					Point point = m_cameras[c]->projectOnView(Point3f((float) x, (float) y, (float) z));

					// Save the pixel coordinates 'point' of the voxel projection on camera 'c'
					voxel->camera_projection[(int) c] = point;

					// If it's within the camera's FoV, flag the projection
					if (point.x >= 0 && point.x < m_plane_size.width && point.y >= 0 && point.y < m_plane_size.height)
						voxel->valid_camera_projection[(int) c] = 1;
				}

				//Writing voxel 'p' is not critical as it's unique (thread safe)
				m_voxels[p] = voxel;
			}
		}
	}

	cout << "done!" << endl;

	if (online)
	{
		for (int i = 0; i < 4; i++)
		{
			FileStorage fs("histogram" + string(to_string(i)) + ".yml", FileStorage::READ);
			Mat hist;
			fs["hist"] >> hist;
			fs.release();

			offlineHistograms0.push_back(hist);
		}
		for (int i = 10; i < 14; i++)
		{
			FileStorage fs("histogram" + string(to_string(i)) + ".yml", FileStorage::READ);
			Mat hist;
			fs["hist"] >> hist;
			fs.release();

			offlineHistograms1.push_back(hist);
		}

		
	}
}

Mat colorModel(vector<Point2f> voxelProjected, Mat frame_rgb, Mat frame_hsv, vector<int> indeces, int histID = 100)
{
	Mat mask = frame_rgb.clone();
	mask.setTo(cv::Scalar(0, 0, 0));

	for (int i : indeces)
	{
		mask.at<Vec3b>(voxelProjected.at(i)) = frame_rgb.at<Vec3b>(voxelProjected.at(i));
	}

	cvtColor(mask, mask, COLOR_BGR2GRAY);
	threshold(mask, mask, 5, 255, THRESH_BINARY);

	Mat element = getStructuringElement(MORPH_RECT, Size(9, 9));
	// dilation followed by erosion to close small gaps
	morphologyEx(mask, mask, MORPH_CLOSE, element, Point(-1, -1));
	Mat outFrame;
	
	frame_rgb.copyTo(outFrame, mask);
	imshow("Display window", outFrame);
	
	if (frame > 70 && frame < 100) {
		waitKey(0);
	}
	//waitKey(0);

	int h_bins = 70;
	int s_bins = 30;


	int histSize[] = { h_bins, s_bins };

	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 250 };


	const float* ranges[] = { h_ranges, s_ranges };
	int channels[] = { 0, 1 };
	Mat hist;
	calcHist(&frame_hsv, 1, channels, mask, hist, 2, histSize, ranges, true, false);
	normalize(hist, hist, 1, 0, NORM_MINMAX);

	if (histID < 99) {
		FileStorage fs("histogram" + string(to_string(histID)) + ".yml", FileStorage::WRITE);
		fs << "hist" << hist;
		fs.release();
	}

	return hist;
}




/**
 * Count the amount of camera's each voxel in the space appears on,
 * if that amount equals the amount of cameras, add that voxel to the
 * visible_voxels vector
 */
void Reconstructor::update()
{
	frame += 1;
	voxels2dCoordinates.clear();
	voxelProjected0.clear();
	voxelProjected1.clear();
	m_visible_voxels.clear();
	visibleVoxelLabel.clear();
	histograms0.clear();
	histograms1.clear();
	std::vector<Voxel*> visible_voxels;

	int v;
#pragma omp parallel for schedule(static) private(v) shared(visible_voxels)
	for (v = 0; v < (int)m_voxels_amount; ++v)
	{
		int camera_counter = 0;
		Voxel* voxel = m_voxels[v];

		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			if (voxel->valid_camera_projection[c])
			{
				const Point point = voxel->camera_projection[c];

				//If there's a white pixel on the foreground image at the projection point, add the camera
				if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 255) ++camera_counter;
			}
		}

		// If the voxel is present on all cameras
		if (camera_counter == m_cameras.size())
		{
#pragma omp critical //push_back is critical
			visible_voxels.push_back(voxel);
			voxels2dCoordinates.push_back(Point2f(voxel->x, voxel->y));

			voxelProjected0.push_back(m_cameras[0]->projectOnView(Point3f(voxel->x, voxel->y, voxel->z)));
			voxelProjected1.push_back(m_cameras[1]->projectOnView(Point3f(voxel->x, voxel->y, voxel->z)));
			
			

		}
	}


	kmeans(voxels2dCoordinates, 4, labels, TermCriteria(CV_TERMCRIT_EPS, 100, 0.01), 8, KMEANS_RANDOM_CENTERS, centers);
	//trying to deal with occlusions
	Point3f pos0 = m_cameras[0]->getCameraLocation();
	Point3f pos1 = m_cameras[1]->getCameraLocation();
	vector<double> centerCamera0Distance;
	vector<double> centerCamera1Distance;
	for (int i = 0; i < 4; i++)
	{
		centerCamera0Distance.push_back(norm(pos0 - Point3f(centers[i].x, centers[i].y, 0)));
	}
	for (int i = 0; i < 4; i++)
	{
		centerCamera1Distance.push_back(norm(pos1 - Point3f(centers[i].x, centers[i].y, 0)));
	}
	

	Mat frame_rgb0 = m_cameras[0]->getFrame();
	Mat frame_rgb1 = m_cameras[1]->getFrame();
	Mat frame_hsv0;
	Mat frame_hsv1;
	cvtColor(frame_rgb0, frame_hsv0, COLOR_BGR2HSV);
	cvtColor(frame_rgb1, frame_hsv1, COLOR_BGR2HSV);
	//initialize vector of vectors
	for (int i = 0; i < 4; i++)
	{
		vector<int> label;
		visibleVoxelLabel.push_back(label);
	}

	for (int i = 0; i < visible_voxels.size(); i++)
	{	
		visible_voxels[i]->label = labels[i];
		labelledVoxelProjected0.push_back(Point3f(voxelProjected0[i].x, voxelProjected0[i].y, labels[i]));
		labelledVoxelProjected1.push_back(Point3f(voxelProjected1[i].x, voxelProjected1[i].y, labels[i]));
		visibleVoxelLabel[labels[i]].push_back(int(i));
	}
	//still trying to deal with occlusions
	vector<int> y(centerCamera0Distance.size());
	iota(y.begin(), y.end(), 0);
	auto comparator = [&centerCamera0Distance](int a, int b) { return centerCamera0Distance[a] < centerCamera0Distance[b]; };
	sort(y.begin(), y.end(), comparator);

	//for (int i : y)
	//{
	//	for (int j : visibleVoxelLabel[labels[i]])
	//	{

	//	}
	//	//starting from label i
	//	//write 2d coord in another vector of vectors, first index being person, only if those coordinates are not already in it
	//}

	
	if (!online) {
		if (frame == 3) {
			for (int i = 0; i < 4; i++)
			{
				size_t half_size = visibleVoxelLabel[i].size() / 2;
				vector<int> visibleVoxelLabel0(visibleVoxelLabel[i].begin() + half_size, visibleVoxelLabel[i].end());
				half_size = visibleVoxelLabel[i].size() / 2;
				vector<int> visibleVoxelLabel1(visibleVoxelLabel[i].begin() + half_size, visibleVoxelLabel[i].end());

				histograms0.push_back(colorModel(voxelProjected0, frame_rgb0, frame_hsv0, visibleVoxelLabel0, i));
				histograms1.push_back(colorModel(voxelProjected1, frame_rgb1, frame_hsv1, visibleVoxelLabel1, i + 10));

			}
		}
	}
	else {
		for (int i = 0; i < 4; i++)
		{
			size_t half_size = visibleVoxelLabel[i].size() / 2;
			vector<int> visibleVoxelLabel0(visibleVoxelLabel[i].begin() + half_size, visibleVoxelLabel[i].end());
			half_size = visibleVoxelLabel[i].size() / 2;
			vector<int> visibleVoxelLabel1(visibleVoxelLabel[i].begin() + half_size, visibleVoxelLabel[i].end());
			histograms0.push_back(colorModel(voxelProjected0, frame_rgb0, frame_hsv0, visibleVoxelLabel0));
			histograms1.push_back(colorModel(voxelProjected1, frame_rgb1, frame_hsv1, visibleVoxelLabel1));
		}

		//waitKey(0);

		double distance0;
		double distance1;
		int correspondingLabel[4];
		vector<double> distance;
		vector<vector<double>> distances;
		vector<int> unavailableLabels;

		distances.clear();
		unavailableLabels.clear();

		for (int i = 0; i < 4; i++)
		{
			distance.clear();
			for (int j = 0; j < 4; j++)
			{
				distance0 = compareHist(histograms0[j], offlineHistograms0[i], HISTCMP_CHISQR);
				distance1 = compareHist(histograms1[j], offlineHistograms1[i], HISTCMP_CHISQR);
				distance.push_back(distance0 + distance1);
			}
			distances.push_back(distance);
		}

		bool found = false;

		for (int i = 0; i < 4; i++)
		{
			found = false;
			while (!found) {

				int minElementIndex = std::min_element(distances[i].begin(), distances[i].end()) - distances[i].begin();

				if (std::find(unavailableLabels.begin(), unavailableLabels.end(), minElementIndex) == unavailableLabels.end())
				{
					unavailableLabels.push_back(minElementIndex);
					correspondingLabel[minElementIndex] = i;
					found = true;
				}
				else {
					distances[i][minElementIndex] = 99999999999;
				}
			}
		}



		for (int i = 0; i < visible_voxels.size(); i++)
		{
			if (visible_voxels[i]->label == 0)
			{
				visible_voxels[i]->label = correspondingLabel[0];

			}
			else if (visible_voxels[i]->label == 1)
			{
				visible_voxels[i]->label = correspondingLabel[1];

			}
			else if (visible_voxels[i]->label == 2)
			{
				visible_voxels[i]->label = correspondingLabel[2];

			}
			else if (visible_voxels[i]->label == 3)
			{
				visible_voxels[i]->label = correspondingLabel[3];

			}

		}
	}

	m_visible_voxels.insert(m_visible_voxels.end(), visible_voxels.begin(), visible_voxels.end());
}


} /* namespace nl_uu_science_gmt */
