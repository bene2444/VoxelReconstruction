#include <cstdlib>
#include <string>
#include <iostream>


#include "utilities/General.h"
#include "VoxelReconstruction.h"


using namespace nl_uu_science_gmt;
using namespace std;
using namespace cv;

int main(
		int argc, char** argv)
{
	VoxelReconstruction::showKeys();
	VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	vr.run(argc, argv);
	

	return EXIT_SUCCESS;
}
