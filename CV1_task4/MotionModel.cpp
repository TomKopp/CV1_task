
#include "MotionModel.h"
MotionModel::MotionModel(double stdXY_, double stdSize_)
: stdXY(stdXY_)
, stdSize(stdSize_)
{
	// IMPLEMENT - DONE
	// create nomral distributions using std::normal_distribution<double> class for the motion
	// in x, y, and the change of the window size
	// store the distributions
	distributionXY = std::normal_distribution<double>(0, stdXY);
	distributionSize = std::normal_distribution<double>(0, stdSize);
}


Particle MotionModel::move(const Particle & p, std::mt19937& engine) {
	//Particle result(0, 0, 50);
	// IMPLEMENT - DONE
	// move the particle randomly according to the motion model defined by the stored distributions
	// return the particle

	return Particle(
		p.x + distributionXY(engine)
		, p.y + distributionXY(engine)
		, p.size + distributionSize(engine)
	);
}
