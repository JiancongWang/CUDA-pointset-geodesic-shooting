// This class contains method to convert float, float2, float3, float4, float9
// to vector<double> mainly for use in VTK optimizer.
// All methods in this class should be static. No single class variable should be stored
#ifndef FLOAT2DVEC_H_
#define FLOAT2DVEC_H_

#include <vnl/vnl_vector.h>

class Float2DVec{
public:
	static void tall2wide(const vnl_vector<double> &x, float2 * p, int k);
	static void tall2wide(const vnl_vector<double> &x, float3 * p, int k);
	static void wide2tall(float2 * p, vnl_vector<double> &x, int k);
	static void wide2tall(float3 * p, vnl_vector<double> &x, int k);

	static void tall2wide_dd(const float * x, float2 * p, int k);
	static void tall2wide_dd(const float * x, float3 * p, int k);
	static void wide2tall_dd(float2 * p, float * x, int k);
	static void wide2tall_dd(float3 * p, float * x, int k);

	static void multiply_by_constant(float * x, float n, int k);

private:
};

#endif
