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
	vector<Point2f> voxelProjected1;
	vector<Point2f> voxelProjected2;
	vector<Point2f> voxelProjected3;
	vector<vector<Point2f>> projected_voxels_by_cluster0;
	vector<vector<Point2f>> projected_voxels_by_cluster1;
	vector<vector<Point2f>> projected_voxels_by_cluster2;
	vector<vector<Point2f>> projected_voxels_by_cluster3;
	vector<vector<int>> visible_voxels_by_label;
	vector<Point2f> centers;
	vector<int> labels;
	vector<Mat> offlineHistograms0;
	vector<Mat> offlineHistograms1;
	vector<Mat> offlineHistograms2;
	vector<Mat> offlineHistograms3;
	vector<Mat> histograms0;
	vector<Mat> histograms1;
	vector<Mat> histograms2;
	vector<Mat> histograms3;
	vector<vector<Point2f>> positionOverTime;
	int frame = 0;
	bool online = true;

	/**
	 * Constructor
	 * Voxel reconstruction class
	 */
	Reconstructor::Reconstructor(
		const vector<Camera*>& cs) :
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
		m_corners.push_back(new Point3f((float)xL, (float)yL, (float)zL));
		m_corners.push_back(new Point3f((float)xL, (float)yR, (float)zL));
		m_corners.push_back(new Point3f((float)xR, (float)yR, (float)zL));
		m_corners.push_back(new Point3f((float)xR, (float)yL, (float)zL));

		// top
		m_corners.push_back(new Point3f((float)xL, (float)yL, (float)zR));
		m_corners.push_back(new Point3f((float)xL, (float)yR, (float)zR));
		m_corners.push_back(new Point3f((float)xR, (float)yR, (float)zR));
		m_corners.push_back(new Point3f((float)xR, (float)yL, (float)zR));

		// Acquire some memory for efficiency
		cout << "Initializing " << m_voxels_amount << " voxels ";
		m_voxels.resize(m_voxels_amount);

		int z;
		int pdone = 0;
#pragma omp parallel for schedule(static) private(z) shared(pdone)
		for (z = zL; z < zR; z += m_step)
		{
			const int zp = (z - zL) / m_step;
			int done = cvRound((zp * plane / (double)m_voxels_amount) * 100.0);

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
						Point point = m_cameras[c]->projectOnView(Point3f((float)x, (float)y, (float)z));

						// Save the pixel coordinates 'point' of the voxel projection on camera 'c'
						voxel->camera_projection[(int)c] = point;

						// If it's within the camera's FoV, flag the projection
						if (point.x >= 0 && point.x < m_plane_size.width && point.y >= 0 && point.y < m_plane_size.height)
							voxel->valid_camera_projection[(int)c] = 1;
					}

					//Writing voxel 'p' is not critical as it's unique (thread safe)
					m_voxels[p] = voxel;
				}
			}
		}
		vector<Point2f> position2f;
		vector<Point3f> position3f;
		for (int i = 0; i < 4; i++) {
			positionOverTime.push_back(position2f);
		}
		cout << "done!" << endl;

		if (online)
		{
			ReadHistograms();
		}
	}

	void Reconstructor::ReadHistograms()
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

		for (int i = 20; i < 24; i++)
		{
			FileStorage fs("histogram" + string(to_string(i)) + ".yml", FileStorage::READ);
			Mat hist;
			fs["hist"] >> hist;
			fs.release();

			offlineHistograms2.push_back(hist);
		}

		for (int i = 30; i < 34; i++)
		{
			FileStorage fs("histogram" + string(to_string(i)) + ".yml", FileStorage::READ);
			Mat hist;
			fs["hist"] >> hist;
			fs.release();

			offlineHistograms3.push_back(hist);
		}
	}

	Mat colorModel(vector<Point2f> voxelProjected, Mat frame_rgb, Mat frame_hsv, int histID = 100)
	{
		Mat mask = frame_rgb.clone();
		mask.setTo(cv::Scalar(0, 0, 0));

		for (Point2f p : voxelProjected)
		{
			mask.at<Vec3b>(p) = frame_rgb.at<Vec3b>(p);
		}

		cvtColor(mask, mask, COLOR_BGR2GRAY);
		threshold(mask, mask, 5, 255, THRESH_BINARY);

		Mat element = getStructuringElement(MORPH_RECT, Size(4, 4));
		// dilation followed by erosion to close small gaps
		morphologyEx(mask, mask, MORPH_CLOSE, element, Point(-1, -1));
		Mat outFrame;

		frame_rgb.copyTo(outFrame, mask);
		imshow("mask", mask);
		imshow("Display window", outFrame);
		imshow("hsv", frame_hsv);

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
		ClearVectors();
		frame += 1;
		vector<Voxel*> visible_voxels = determineVisibleVoxels();
		// clustering
		kmeans(voxels2dCoordinates, 4, labels, TermCriteria(CV_TERMCRIT_EPS, 100, 0.01), 8, KMEANS_RANDOM_CENTERS, centers);

		vector<double> camera0_distances_to_centers = getDistancesFromCameraToCenters(0);
		vector<double> camera1_distances_to_centers = getDistancesFromCameraToCenters(1);
		vector<double> camera2_distances_to_centers = getDistancesFromCameraToCenters(2);
		vector<double> camera3_distances_to_centers = getDistancesFromCameraToCenters(3);

		Mat frame_rgb0, frame_hsv0, frame_rgb1, frame_hsv1, frame_rgb2, frame_hsv2, frame_rgb3, frame_hsv3;
		tie(frame_rgb0, frame_hsv0) = getFrameColorValues(0);
		tie(frame_rgb1, frame_hsv1) = getFrameColorValues(1);
		tie(frame_rgb2, frame_hsv2) = getFrameColorValues(2);
		tie(frame_rgb3, frame_hsv3) = getFrameColorValues(3);

		//initialize vector of vectors
		for (int i = 0; i < 4; i++)
		{
			vector<int> label;
			visible_voxels_by_label.push_back(label);
		}

		for (int i = 0; i < visible_voxels.size(); i++)
		{
			visible_voxels[i]->label = labels[i];
			visible_voxels_by_label[labels[i]].push_back(i);
		}
		// calculate distance between clusters and cameras to determine the order, from closest to furthest away
		vector<int> clusters_ordered_by_distance0 = get_ordered_clusters(camera0_distances_to_centers);
		vector<int> clusters_ordered_by_distance1 = get_ordered_clusters(camera1_distances_to_centers);
		vector<int> clusters_ordered_by_distance2 = get_ordered_clusters(camera2_distances_to_centers);
		vector<int> clusters_ordered_by_distance3 = get_ordered_clusters(camera3_distances_to_centers);

		vector<Point2f> clusterCoordinates;
		clusterCoordinates.clear();
		Mat cameraShot0 = frame_rgb0.clone();
		cameraShot0.setTo(cv::Scalar(0, 0, 0));
		Mat cameraShot1 = frame_rgb1.clone();
		cameraShot1.setTo(cv::Scalar(0, 0, 0));
		Mat cameraShot2 = frame_rgb2.clone();
		cameraShot2.setTo(cv::Scalar(0, 0, 0));
		Mat cameraShot3 = frame_rgb3.clone();
		cameraShot3.setTo(cv::Scalar(0, 0, 0));
		Vec3b black = cameraShot0.at<Vec3b>(voxelProjected0.at(1));

		for (int i = 0; i < 4; i++)
		{
			projected_voxels_by_cluster0.push_back(clusterCoordinates);
			projected_voxels_by_cluster1.push_back(clusterCoordinates);
			projected_voxels_by_cluster2.push_back(clusterCoordinates);
			projected_voxels_by_cluster3.push_back(clusterCoordinates);
		}
		// Occlusion mechanism: start filling up the vector with the voxels of the closest person, only add voxel if the corresponding pixel has not been used yet
		for (int i = 0; i < 4; i++)
		{
			Mat element = getStructuringElement(MORPH_RECT, Size(2, 2));
			// dilation followed by erosion to close small gaps
			morphologyEx(cameraShot0, cameraShot0, MORPH_CLOSE, element, Point(-1, -1));
			morphologyEx(cameraShot1, cameraShot1, MORPH_CLOSE, element, Point(-1, -1));
			morphologyEx(cameraShot2, cameraShot2, MORPH_CLOSE, element, Point(-1, -1));
			morphologyEx(cameraShot3, cameraShot3, MORPH_CLOSE, element, Point(-1, -1));
		
			
			for (int j : visible_voxels_by_label[clusters_ordered_by_distance0[i]])
			{
				if (cameraShot0.at<Vec3b>(voxelProjected0.at(j)) == black)
				{
					cameraShot0.at<Vec3b>(voxelProjected0.at(j)) = Vec3b(200, 200, 200);
					projected_voxels_by_cluster0[clusters_ordered_by_distance0[i]].push_back(voxelProjected0[j]);
				}
			}
			for (int j : visible_voxels_by_label[clusters_ordered_by_distance1[i]])
			{
				if (cameraShot1.at<Vec3b>(voxelProjected1.at(j)) == black)
				{
					cameraShot1.at<Vec3b>(voxelProjected1.at(j)) = Vec3b(200, 200, 200);
					projected_voxels_by_cluster1[clusters_ordered_by_distance1[i]].push_back(voxelProjected1[j]);
				}
			}
			for (int j : visible_voxels_by_label[clusters_ordered_by_distance2[i]])
			{
				if (cameraShot2.at<Vec3b>(voxelProjected2.at(j)) == black)
				{
					cameraShot2.at<Vec3b>(voxelProjected2.at(j)) = Vec3b(200, 200, 200);
					projected_voxels_by_cluster2[clusters_ordered_by_distance2[i]].push_back(voxelProjected2[j]);
				}
			}
			for (int j : visible_voxels_by_label[clusters_ordered_by_distance3[i]])
			{
				if (cameraShot3.at<Vec3b>(voxelProjected3.at(j)) == black)
				{
					cameraShot3.at<Vec3b>(voxelProjected3.at(j)) = Vec3b(200, 200, 200);
					projected_voxels_by_cluster3[clusters_ordered_by_distance3[i]].push_back(voxelProjected3[j]);
				}
			}
		}

		
		// save histograms to files if in offline mode
		if (!online) {
			if (frame == 44) {
				for (int i = 0; i < 4; i++)
				{
					size_t half_size = projected_voxels_by_cluster0[i].size() / 2;
					vector<Point2f> halfClusterVoxelProjected0(projected_voxels_by_cluster0[i].begin() + half_size, projected_voxels_by_cluster0[i].end());

					histograms0.push_back(colorModel(halfClusterVoxelProjected0, frame_rgb0, frame_hsv0, i));
				}
			}
			if (frame == 63) {
				for (int i = 0; i < 4; i++)
				{
					size_t half_size = projected_voxels_by_cluster1[i].size() / 2;
					vector<Point2f> halfClusterVoxelProjected1(projected_voxels_by_cluster1[i].begin() + half_size, projected_voxels_by_cluster1[i].end());

					histograms1.push_back(colorModel(halfClusterVoxelProjected1, frame_rgb1, frame_hsv1, i + 10));
				}
			}
			if (frame == 180) {
				for (int i = 0; i < 4; i++)
				{
					size_t half_size = projected_voxels_by_cluster2[i].size() / 2;
					vector<Point2f> halfClusterVoxelProjected2(projected_voxels_by_cluster2[i].begin() + half_size, projected_voxels_by_cluster2[i].end());
					half_size = projected_voxels_by_cluster3[i].size() / 2;
					vector<Point2f> halfClusterVoxelProjected3(projected_voxels_by_cluster3[i].begin() + half_size, projected_voxels_by_cluster3[i].end());

					histograms2.push_back(colorModel(halfClusterVoxelProjected2, frame_rgb2, frame_hsv2, i + 20));
				}
			}

			if (frame == 508) {
				for (int i = 0; i < 4; i++)
				{
					size_t half_size = projected_voxels_by_cluster3[i].size() / 2;
					vector<Point2f> halfClusterVoxelProjected3(projected_voxels_by_cluster3[i].begin() + half_size, projected_voxels_by_cluster3[i].end());

					histograms3.push_back(colorModel(halfClusterVoxelProjected3, frame_rgb3, frame_hsv3, i + 30));
				}
			}
		}
		else {
			for (int i = 0; i < 4; i++)
			{
				size_t half_size = projected_voxels_by_cluster0[i].size() / 2;
				vector<Point2f> halfClusterVoxelProjected0(projected_voxels_by_cluster0[i].begin() + half_size, projected_voxels_by_cluster0[i].end());
				half_size = projected_voxels_by_cluster1[i].size() / 2;
				vector<Point2f> halfClusterVoxelProjected1(projected_voxels_by_cluster1[i].begin() + half_size, projected_voxels_by_cluster1[i].end());
				half_size = projected_voxels_by_cluster2[i].size() / 2;
				vector<Point2f> halfClusterVoxelProjected2(projected_voxels_by_cluster2[i].begin() + half_size, projected_voxels_by_cluster2[i].end());
				half_size = projected_voxels_by_cluster3[i].size() / 2;
				vector<Point2f> halfClusterVoxelProjected3(projected_voxels_by_cluster3[i].begin() + half_size, projected_voxels_by_cluster3[i].end());

				histograms0.push_back(colorModel(halfClusterVoxelProjected0, frame_rgb0, frame_hsv0));
				histograms1.push_back(colorModel(halfClusterVoxelProjected1, frame_rgb1, frame_hsv1));
				histograms2.push_back(colorModel(halfClusterVoxelProjected2, frame_rgb2, frame_hsv2));
				histograms3.push_back(colorModel(halfClusterVoxelProjected3, frame_rgb3, frame_hsv3));
			}

			// calculate histogram differences to match the offline models with the online ones
			double distance0;
			double distance1;
			double distance2;
			double distance3;
			vector<double> distances_per_camera;
			int correspondingLabel[4];
			int correspondingLabelReversed[4];
			vector<double> distance;
			vector<vector<double>> distances;
			vector<int> unavailableLabels;

			distances.clear();
			unavailableLabels.clear();
			double sum_distances = 0;

			for (int i = 0; i < 4; i++)
			{
				distance.clear();
				for (int j = 0; j < 4; j++)
				{
					distances_per_camera.clear();
					sum_distances = 0;
					
					distance0 = compareHist(histograms0[j], offlineHistograms0[i], HISTCMP_CHISQR);
					distance1 = compareHist(histograms1[j], offlineHistograms1[i], HISTCMP_CHISQR);
					distance2 = compareHist(histograms2[j], offlineHistograms2[i], HISTCMP_CHISQR);
					distance3 = compareHist(histograms3[j], offlineHistograms3[i], HISTCMP_CHISQR);
					distances_per_camera.push_back(distance0);
					distances_per_camera.push_back(distance1);
					distances_per_camera.push_back(distance2);
					distances_per_camera.push_back(distance3);
					sum_distances = distances_per_camera[get_ordered_clusters(distances_per_camera)[0]] + distances_per_camera[get_ordered_clusters(distances_per_camera)[1]];
					
					distance.push_back(sum_distances);
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
						correspondingLabelReversed[i] = minElementIndex;
						found = true;
					}
					else {
						distances[i][minElementIndex] = 99999999999999999;
					}
				}
			}
			// position of clusters in time, on the floor
			m_trails0.push_back(new Point3f(centers[correspondingLabelReversed[0]].x, centers[correspondingLabelReversed[0]].y, 3));
			m_trails1.push_back(new Point3f(centers[correspondingLabelReversed[1]].x, centers[correspondingLabelReversed[1]].y, 3));
			m_trails2.push_back(new Point3f(centers[correspondingLabelReversed[2]].x, centers[correspondingLabelReversed[2]].y, 3));
			m_trails3.push_back(new Point3f(centers[correspondingLabelReversed[3]].x, centers[correspondingLabelReversed[3]].y, 3));

			for (int i = 0; i < 4; i++) {
				positionOverTime[correspondingLabel[i]].push_back(centers[i]);
			}

			FileStorage fs("positionOverTime.yml", FileStorage::WRITE);
			fs << "pos" + string(to_string(frame)) << positionOverTime;

			// assign correct label
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

	void Reconstructor::AddToProjection(cv::Mat& camera_shot, cv::Point2f voxel_position, std::vector<cv::Point2f> projected_voxels, cv::Vec3b black)
	{
		if (camera_shot.at<Vec3b>(voxel_position) == black)
		{
			camera_shot.at<Vec3b>(voxel_position) = Vec3b(200, 200, 200);
			projected_voxels.push_back(voxel_position);
		}
	}

	vector<nl_uu_science_gmt::Reconstructor::Voxel*> Reconstructor::determineVisibleVoxels()
	{
		vector<Voxel*> visible_voxels;
		int voxel_index;
#pragma omp parallel for schedule(static) private(v) shared(visible_voxels)
		for (voxel_index = 0; voxel_index < (int)m_voxels_amount; ++voxel_index)
		{
			int camera_counter = 0;
			Voxel* voxel = m_voxels[voxel_index];

			for (size_t camera_index = 0; camera_index < m_cameras.size(); ++camera_index)
			{
				if (voxel->valid_camera_projection[camera_index])
				{
					const Point point = voxel->camera_projection[camera_index];

					//If there's a white pixel on the foreground image at the projection point, add the camera
					if (m_cameras[camera_index]->getForegroundImage().at<uchar>(point) == 255) ++camera_counter;
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
				voxelProjected2.push_back(m_cameras[2]->projectOnView(Point3f(voxel->x, voxel->y, voxel->z)));
				voxelProjected3.push_back(m_cameras[3]->projectOnView(Point3f(voxel->x, voxel->y, voxel->z)));
			}
		}

		return visible_voxels;
	}

	/// <summary>
	/// Returns a vector of distances from the specified camera to the centers of the clusters.
	/// </summary>
	/// <param name="camera_index">The zero-based index of the camera.</param>
	vector<double> Reconstructor::getDistancesFromCameraToCenters(int camera_index)
	{
		Point3f camera_location = m_cameras[camera_index]->getCameraLocation();
		vector<double> camera_distances_to_centers;
		for (int cluster_index = 0; cluster_index < 4; cluster_index++)
		{
			camera_distances_to_centers.push_back(norm(camera_location - Point3f(centers[cluster_index].x, centers[cluster_index].y, 0)));
		}

		return camera_distances_to_centers;
	}

	/// <summary>
	/// Retrieves a frame from the speficied camera and returns the RGB and HSV representations.
	/// </summary>
	/// <param name="camera_index">The zero-based index of the camera.</param>
	tuple<Mat, Mat> Reconstructor::getFrameColorValues(int camera_index)
	{
		Mat frame_rgb = m_cameras[camera_index]->getFrame();
		Mat frame_hsv;
		cvtColor(frame_rgb, frame_hsv, COLOR_BGR2HSV);

		return make_tuple(frame_rgb, frame_hsv);
	}

	/// <summary>
	/// Returns a vector of indices corresponding to the clusters, ordered by their distance to the camera, from closest to furthest.
	/// </summary>
	/// <param name="camera_distances_to_centers">A vector containing the distance from the camera to each of the clusters.</param>
	vector<int> Reconstructor::get_ordered_clusters(vector<double> camera_distances_to_centers)
	{
		vector<int> clusters_ordered_by_distance(camera_distances_to_centers.size());
		iota(clusters_ordered_by_distance.begin(), clusters_ordered_by_distance.end(), 0);
		auto comparator = [&camera_distances_to_centers](int a, int b) { return camera_distances_to_centers[a] < camera_distances_to_centers[b]; };
		sort(clusters_ordered_by_distance.begin(), clusters_ordered_by_distance.end(), comparator);

		return clusters_ordered_by_distance;
	}

	void Reconstructor::ClearVectors()
	{
		voxels2dCoordinates.clear();
		voxelProjected0.clear();
		voxelProjected1.clear();
		voxelProjected2.clear();
		voxelProjected3.clear();
		projected_voxels_by_cluster0.clear();
		projected_voxels_by_cluster1.clear();
		projected_voxels_by_cluster2.clear();
		projected_voxels_by_cluster3.clear();
		m_visible_voxels.clear();
		visible_voxels_by_label.clear();
		histograms0.clear();
		histograms1.clear();
		histograms2.clear();
		histograms3.clear();
	}


} /* namespace nl_uu_science_gmt */