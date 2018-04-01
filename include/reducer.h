#include <iostream>

#ifndef REDUCER_H_
#define REDUCER_H_

class Reducer{
public:
	Reducer(){};
	~Reducer(){};
	static float reduce_sum_wrapper(int n, float *d_idata);
	static float reduce_max_wrapper(int n, float *d_idata);
	static float reduce_min_wrapper(int n, float *d_idata);

private:
};

#endif
