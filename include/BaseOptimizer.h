#ifndef BASE_OPTIMIZER_H
#define BASE_OPTIMIZER_H
#include "Frame.h"

// Abstract optimizer base class
class BaseOptimizer
{
public:

	/**
		Applies the optimizer to the given frame

		@param src Image frame
	*/
	virtual void apply(Frame& src) = 0;
};
#endif
