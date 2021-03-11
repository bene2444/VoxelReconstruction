#include "CameraCalibration.h"
#include <iostream>
#include <opencv2\calib3d.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2/imgproc.hpp>

#pragma warning(disable : 4996)

using namespace std;
using namespace cv;

/// <summary>
/// Parses the command line arguments given to the program.
/// </summary>
CommandLineParser parseCommandLineArguments(int argc, char* argv[])
{
	const String keys
		= "{help h usage ? |           | print this message            }"
		"{@settings      |default.xml| input setting file            }"
		"{d              |           | actual distance between top-left and top-right corners of "
		"the calibration grid }"
		"{winSize        | 11        | Half of search window for cornerSubPix }";
	CommandLineParser parser(argc, argv, keys);
	parser.about("This is a camera calibration sample.\n"
		"Usage: camera_calibration [configuration_file -- default ./default.xml]\n"
		"Near the sample file you'll find the configuration file, which has detailed help of "
		"how to edit it. It may be any OpenCV supported file format XML/YAML.");
	if (!parser.check()) {
		parser.printErrors();
		throw Exception();
	}

	if (parser.has("help")) {
		parser.printMessage();
		throw Exception();
	}

	return parser;
}

/// <summary>
/// Retrieves the settings given in the config.xml file.
/// </summary>
//Settings getSettingsFromConfig(CommandLineParser parser) 
Settings getSettingsFromConfig(string inputSettingsFile)
{
	Settings settings;
	FileStorage fs(inputSettingsFile, FileStorage::READ); // Read the settings
	if (!fs.isOpened())
	{
		cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
		throw Exception();
	}
	fs["Settings"] >> settings;
	fs.release();                                         // close Settings file

	if (!settings.goodInput)
	{
		cout << "Invalid input detected. Application stopping. " << endl;
		throw Exception();
	}

	return settings;
}

/// <summary>
/// Calibrates the camera using the specified settings, either with supplied images or using the webcam.
/// </summary>
//int calibration(int argc, char* argv[])
int calibration(string settingsPath)
{
	Settings settings = getSettingsFromConfig(settingsPath);

	int winSize = 11;

	float grid_width = settings.squareSize * (settings.boardSize.width - 1);
	bool release_object = false;

	vector<vector<Point2f> > imagePoints;
	Mat cameraMatrix, distCoeffs;
	Size imageSize;
	int mode = settings.inputType == Settings::IMAGE_LIST ? CAPTURING : DETECTION;
	clock_t prevTimestamp = 0;
	const Scalar RED(0, 0, 255), GREEN(0, 255, 0);
	const char ESC_KEY = 27;

	for (;;)
	{
		if (mode == CALIBRATED)
			return 1;

		Mat view;
		bool blinkOutput = false;

		
		view = settings.nextImage();

		//-----  If no more image, or got enough, then stop calibration and show result -------------
		if (mode == CAPTURING && imagePoints.size() >= (size_t)settings.nrFrames)
		{
			if (runCalibrationAndSave(settings, imageSize, cameraMatrix, distCoeffs, imagePoints, grid_width,
				release_object))
				mode = CALIBRATED;
			else
				mode = DETECTION;
		}
		if (view.empty())          // If there are no more images stop the loop
		{
			// if calibration threshold was not reached yet, calibrate now
			if (mode != CALIBRATED && !imagePoints.empty())
				runCalibrationAndSave(settings, imageSize, cameraMatrix, distCoeffs, imagePoints, grid_width,
					release_object);
			break;
		}

		imageSize = view.size();  // Format input image.
		if (settings.flipVertical)    flip(view, view, 0);

		vector<Point2f> pointBuf;

		bool found;

		int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK;

		found = findChessboardCorners(view, settings.boardSize, pointBuf, chessBoardFlags);
		if (found)
		{
			// improve the found corners' coordinate accuracy for chessboard
			Mat viewGray;
			cvtColor(view, viewGray, COLOR_BGR2GRAY);
			cornerSubPix(viewGray, pointBuf, Size(winSize, winSize),
				Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));

			if (mode == CAPTURING &&  (!settings.inputCapture.isOpened() || clock() - prevTimestamp > settings.delay * 1e-3 * CLOCKS_PER_SEC))
			{
				imagePoints.push_back(pointBuf);
				prevTimestamp = clock();
				blinkOutput = settings.inputCapture.isOpened();
			}

			// Draw the corners.
			drawChessboardCorners(view, settings.boardSize, Mat(pointBuf), found);
		}
		//! [output_text]
		string msg = (mode == CAPTURING) ? "100/100" :
			mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
		int baseLine = 0;
		Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
		Point textOrigin(view.cols - 2 * textSize.width - 10, view.rows - 2 * baseLine - 10);

		if (mode == CAPTURING)
		{
			if (settings.showUndistorsed)
				msg = format("%d/%d Undist", (int)imagePoints.size(), settings.nrFrames);
			else
				msg = format("%d/%d", (int)imagePoints.size(), settings.nrFrames);
		}

		putText(view, msg, textOrigin, 1, 1, mode == CALIBRATED ? GREEN : RED);

		if (blinkOutput)
			bitwise_not(view, view);
		//! [output_text]
		//------------------------- Video capture  output  undistorted ------------------------------
		//! [output_undistorted]
		if (mode == CALIBRATED && settings.showUndistorsed)
		{
			Mat temp = view.clone();
			undistort(temp, view, cameraMatrix, distCoeffs);
		}
		//! [output_undistorted]
		//------------------------------ Show image and check for input commands -------------------
		//! [await_input]
		imshow("Magic cube", view);
		char key = (char)waitKey(settings.inputCapture.isOpened() ? 50 : settings.delay);

		if (key == ESC_KEY)
			break;

		if (key == 'u' && mode == CALIBRATED)
			settings.showUndistorsed = !settings.showUndistorsed;

		mode = CAPTURING;
		//imagePoints.clear();
		//! [await_input]
	}


	return 0;
}


static void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners,
	Settings::Pattern patternType /*= Settings::CHESSBOARD*/)
{
	corners.clear();

	for (int i = 0; i < boardSize.height; ++i)
		for (int j = 0; j < boardSize.width; ++j)
			corners.push_back(Point3f(j * squareSize, i * squareSize, 0));
}

/// <summary>
/// Runs the calibration of the camera
/// </summary>
static bool runCalibration(Settings& settings, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
	vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs,
	vector<float>& reprojErrs, double& totalAvgErr, vector<Point3f>& newObjPoints,
	float grid_width, bool release_object)
{
	//! [fixed_aspect]
	cameraMatrix = Mat::eye(3, 3, CV_64F);
	if (settings.flag & CALIB_FIX_ASPECT_RATIO)
		cameraMatrix.at<double>(0, 0) = settings.aspectRatio;
	//! [fixed_aspect]
	distCoeffs = Mat::zeros(8, 1, CV_64F);

	vector<vector<Point3f> > objectPoints(1);
	//objectPoints[0].clear();
	calcBoardCornerPositions(settings.boardSize, settings.squareSize, objectPoints[0], settings.calibrationPattern);
	objectPoints[0][settings.boardSize.width - 1].x = objectPoints[0][0].x + grid_width;
	newObjPoints = objectPoints[0];

	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	//Find intrinsic and extrinsic camera parameters
	double rms;

	int iFixedPoint = -1;
	if (release_object)
		iFixedPoint = settings.boardSize.width - 1;
	rms = calibrateCameraRO(objectPoints, imagePoints, imageSize, iFixedPoint,
		cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints,
		settings.flag | CALIB_USE_LU);

	if (release_object) {
		cout << "New board corners: " << endl;
		cout << newObjPoints[0] << endl;
		cout << newObjPoints[settings.boardSize.width - 1] << endl;
		cout << newObjPoints[settings.boardSize.width * (settings.boardSize.height - 1)] << endl;
		cout << newObjPoints.back() << endl;
	}

	cout << "Re-projection error reported by calibrateCamera: " << rms << endl;

	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

	objectPoints.clear();
	objectPoints.resize(imagePoints.size(), newObjPoints);
	totalAvgErr = rms;

	return ok;
}

/// <summary>
/// Print camera parameters to the output file
/// </summary>
static void saveCameraParams(Settings& settings, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const vector<float>& reprojErrs, const vector<vector<Point2f> >& imagePoints,
	double totalAvgErr, const vector<Point3f>& newObjPoints)
{
	FileStorage fs(settings.outputFileName, FileStorage::WRITE);

	time_t tm;
	time(&tm);
	struct tm* t2 = localtime(&tm);
	char buf[1024];
	strftime(buf, sizeof(buf), "%c", t2);

	fs << "CameraMatrix" << cameraMatrix;
	fs << "DistortionCoeffs" << distCoeffs;
}

/// <summary>
/// Runs the entire calibration loop, ignoring an image that does not contribute positively, and outputs the camera parameters to out_camera_data.xml
/// </summary>
bool runCalibrationAndSave(Settings& s, Size imageSize, Mat& cameraMatrix, Mat& distCoeffs,
	vector<vector<Point2f> > imagePoints, float grid_width, bool release_object)
{
	int worstImageIndex = -1;
	vector<vector<Point2f>> bestImages = imagePoints;
	vector<Mat> rvecs, tvecs;
	vector<float> reprojErrs;
	double bestAvgError = 100;
	vector<Point3f> newObjPoints;

	runCalibration(s, imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs, reprojErrs,
		bestAvgError, newObjPoints, grid_width, release_object);

	bool ok = runCalibration(s, imageSize, cameraMatrix, distCoeffs, bestImages, rvecs, tvecs, reprojErrs,
		bestAvgError, newObjPoints, grid_width, release_object);

	cout << (ok ? "Calibration succeeded" : "Calibration failed")
		<< ". best re projection error = " << bestAvgError << ", excluding image " << worstImageIndex <<  endl;

	if (ok)
		saveCameraParams(s, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, reprojErrs, imagePoints,
			bestAvgError, newObjPoints);
	return ok;
}
