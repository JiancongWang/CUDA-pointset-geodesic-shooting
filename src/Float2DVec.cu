#include "../include/Float2DVec.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_functions.h"
#include "helper_cuda.h"
#include <vnl/vnl_vector.h>

__global__ void multiplyByKernel1D(float * input, float n, int col ){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= col)
		return;
	input[c] *= n;
}


__global__ void double2floatKernel1D(double * temp, float3 * p , int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= k)
		return;

	p[c].x = (float)temp[c];
	p[c].y = (float)temp[c+k];
	p[c].z = (float)temp[c+2*k];
}

__global__ void double2floatKernel1D(double * temp, float2 * p , int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= k)
		return;

	p[c].x = (float)temp[c];
	p[c].y = (float)temp[c+k];
}

__global__ void float2doubleKernel1D(float2 * p, double * temp, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= k)
		return;

	temp[c] = (double) p[c].x;
	temp[c+k] = (double) p[c].y;
}

__global__ void float2doubleKernel1D(float3 * p, double * temp, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= k)
		return;

	temp[c] = (double) p[c].x;
	temp[c+k] = (double) p[c].y;
	temp[c+2*k] = (double) p[c].z;
}

__global__ void tall2wideDDKernel1D(const float * x, float2 * p, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= k)
		return;

	p[c].x = x[c];
	p[c].y = x[c+k];
}

__global__ void tall2wideDDKernel1D(const float * x, float3 * p, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= k)
		return;

	p[c].x = x[c];
	p[c].y = x[c+k];
	p[c].z = x[c+2*k];
}

__global__ void wide2tallDDKernel1D(float2 * p, float * x,  int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= k)
		return;

	x[c] = p[c].x;
	x[c+k] = p[c].y;
}

__global__ void wide2tallDDKernel1D(float3 * p, float * x,  int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= k)
		return;

	x[c] = p[c].x;
	x[c+k] = p[c].y;
	x[c+2*k] = p[c].z;
}



void Float2DVec::tall2wide(const vnl_vector<double> &x, float2 * d_p, int k){
	double * h_temp = new double[2*k];
	double * d_temp;
	checkCudaErrors(cudaMalloc((void **)&d_temp, 2*k*sizeof(double)));

	x.copy_out(h_temp);
	checkCudaErrors(cudaMemcpy(d_temp, h_temp, 2*k * sizeof(double), cudaMemcpyHostToDevice));

	dim3 threads = dim3(256, 1, 1);
	dim3 blocks = dim3( (k+255)/256, 1, 1);
	double2floatKernel1D <<< blocks, threads >>> (d_temp, d_p, k);

	checkCudaErrors(cudaFree(d_temp));
	delete [] h_temp;
}

void Float2DVec::tall2wide(const vnl_vector<double> &x, float3 * d_p, int k){
	double * h_temp = new double[3*k];
	double * d_temp;

	//	float3 * h_p = new float3[k];
	checkCudaErrors(cudaMalloc((void **)&d_temp, 3*k*sizeof(double)));

	x.copy_out(h_temp);
	checkCudaErrors(cudaMemcpy(d_temp, h_temp, 3*k*sizeof(double), cudaMemcpyHostToDevice));

	dim3 threads = dim3(256, 1, 1);
	dim3 blocks = dim3( (k+255)/256, 1, 1);
	double2floatKernel1D <<< blocks, threads >>> (d_temp, d_p, k);

	//	checkCudaErrors(cudaMemcpy(h_p, d_p, k*sizeof(float3), cudaMemcpyDeviceToHost));
	//	for (int i=0; i<k; i++){
	//		printf("x: (%f, %f, %f)\n", x[i], x[i+k], x[i+2*k]);
	//		printf("h_p0: (%f, %f, %f)\n", h_temp[i], h_temp[i+k], h_temp[i+2*k]);
	//		printf("d_p0: (%f, %f, %f)\n", h_p[i].x, h_p[i].y, h_p[i].z);
	//	}

	checkCudaErrors(cudaFree(d_temp));
	delete [] h_temp;
	//	delete [] h_p;
}

void Float2DVec::wide2tall(float2 * d_p, vnl_vector<double> &x, int k){
	double * h_temp = new double[2*k];
	double * d_temp;
	checkCudaErrors(cudaMalloc((void **)&d_temp, 2*k*sizeof(double)));

	dim3 threads = dim3(256, 1, 1);
	dim3 blocks = dim3( (k+255)/256, 1, 1);
	float2doubleKernel1D <<< blocks, threads >>> (d_p, d_temp, k);

	checkCudaErrors(cudaMemcpy(h_temp, d_temp, 2*k * sizeof(double), cudaMemcpyDeviceToHost));

	x.copy_in(h_temp);

	checkCudaErrors(cudaFree(d_temp));
	delete [] h_temp;
}

void Float2DVec::wide2tall(float3 * d_p, vnl_vector<double> &x, int k){
	double * h_temp = new double[3*k];
	//	float3 * h_p = new float3[k];
	double * d_temp;
	checkCudaErrors(cudaMalloc((void **)&d_temp, 3*k*sizeof(double)));

	dim3 threads = dim3(256, 1, 1);
	dim3 blocks = dim3( (k+255)/256, 1, 1);
	float2doubleKernel1D <<< blocks, threads >>> (d_p, d_temp, k);

	checkCudaErrors(cudaMemcpy(h_temp, d_temp, 3*k * sizeof(double), cudaMemcpyDeviceToHost));
	//	checkCudaErrors(cudaMemcpy(h_p, d_p, k * sizeof(float3), cudaMemcpyDeviceToHost));

	x.copy_in(h_temp);

	//	for (int i=0; i<k; i++){
	//		printf("x: (%f, %f, %f)\n", x[i], x[i+k], x[i+2*k]);
	//		printf("h_p0: (%f, %f, %f)\n", h_temp[i], h_temp[i+k], h_temp[i+2*k]);
	//		printf("d_p0: (%f, %f, %f)\n", h_p[i].x, h_p[i].y, h_p[i].z);
	//	}

	checkCudaErrors(cudaFree(d_temp));
	//	delete [] h_p;
	delete [] h_temp;
}


void Float2DVec::tall2wide_dd(const float * x, float2 * p, int k){
	dim3 threads = dim3(256, 1, 1);
	dim3 blocks = dim3( (k+255)/256, 1, 1);
	tall2wideDDKernel1D <<< blocks, threads >>> (x, p, k);
}

void Float2DVec::tall2wide_dd(const float * x, float3 * p, int k){
	dim3 threads = dim3(256, 1, 1);
	dim3 blocks = dim3( (k+255)/256, 1, 1);
	tall2wideDDKernel1D <<< blocks, threads >>> (x, p, k);
}

void Float2DVec::wide2tall_dd(float2 * p, float * x, int k){
	dim3 threads = dim3(256, 1, 1);
	dim3 blocks = dim3( (k+255)/256, 1, 1);
	wide2tallDDKernel1D <<< blocks, threads >>> (p, x, k);
}

void Float2DVec::wide2tall_dd(float3 * p, float * x, int k){
	dim3 threads = dim3(256, 1, 1);
	dim3 blocks = dim3( (k+255)/256, 1, 1);
	wide2tallDDKernel1D <<< blocks, threads >>> (p, x, k);
}


void Float2DVec::multiply_by_constant(float * x, float n, int k){
	dim3 threads = dim3(256, 1, 1);
	dim3 blocks = dim3( (3*k+255)/256, 1, 1);
	multiplyByKernel1D <<< blocks, threads >>> (x, n, k);
}









