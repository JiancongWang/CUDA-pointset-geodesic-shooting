#include <vnl/algo/vnl_svd.h>
#include <vnl/vnl_cost_function.h>
#include <vnl/algo/vnl_lbfgs.h>
#include <vnl/vnl_cost_function.h>

#include <vnl/algo/vnl_lbfgsb.h>
#include <vnl/algo/vnl_brent_minimizer.h>

#include <vtkPointData.h>
#include <vtkDoubleArray.h>
#include <vtkQuadricClustering.h>
#include <vtkSmartPointer.h>

#include <stdio.h>

#include "../include/PointSetShootingProblem.h"
#include "../include/ReadWriteVTK.h"
#include "../include/Float2DVec.h"
#include "../include/CommandLineHelper.h"

#include "../include/hamiltonian.h"
#include "../include/check.h"

#include "helper_functions.h"
#include "helper_cuda.h"

#include <unistd.h>
#include "../include/timer.h"

#include <math.h>


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Class function of the cost function classes used by the VNL library */
/* The cost functions that get put into ITK VNL optimizer */
PointSetShootingCostFunction_CUDA::PointSetShootingCostFunction_CUDA(
		const ShootingParameters &param, const ShootingData &data)
: vnl_cost_function(){
	this->param = param; // Set parameters
	this->data = data; // Set data
	this->dim = data.k * data.VDIM; // Set number of variable to optimize
	this->gradient_time = 0.0;

	// Initialize alpha and beta
	if (data.VDIM==2){
		checkCudaErrors(cudaMemset(data.alpha2d, 0, data.k*sizeof(float2)));
		checkCudaErrors(cudaMemset(data.beta2d, 0, data.k*sizeof(float2)));
	}else if (data.VDIM==3){
		checkCudaErrors(cudaMemset(data.alpha3d, 0, data.k*sizeof(float3)));
		checkCudaErrors(cudaMemset(data.beta3d, 0, data.k*sizeof(float3)));
	}else{
		printf("Dimension is not 2 or 3 at line number %d in in function %s, file %s\n", __LINE__, __func__, __FILE__);
		exit(-1);
	}

	// Initialize p0 - which is useless but in the original code
	if (data.VDIM==2){
		PointSetHamiltonianSystem_CUDA::initP_CUDA2D(data.q02d, data.qT2d, data.p02d, data.N, data.k, true);
	}
	else if (data.VDIM==3){
		PointSetHamiltonianSystem_CUDA::initP_CUDA3D(data.q03d, data.qT3d, data.p03d, data.N, data.k, true);
	}
	else{
		printf("Dimension is not 2 or 3 at line number %d in in function %s, file %s\n", __LINE__, __func__, __FILE__);
		exit(-1);
	}
	checkCudaErrors(cudaDeviceSynchronize());

}

void PointSetShootingCostFunction_CUDA::compute(vnl_vector<double> const& x, double *f, vnl_vector<double>* g){
	checkCudaErrors(cudaDeviceSynchronize());
	if (this->data.VDIM ==2 )
		compute2D(x, f, g);
	else if (this->data.VDIM ==3 )
		compute3D(x, f, g);
	else{
		printf("Dimension is not 2 or 3 at line number %d in in function %s, file %s\n", __LINE__, __func__, __FILE__);
		exit(-1);
	}
	checkCudaErrors(cudaDeviceSynchronize());
}

void PointSetShootingCostFunction_CUDA::compute2D(vnl_vector<double> const& x, double *f, vnl_vector<double>* g){
	// Initialize the p0-vector
	Float2DVec::tall2wide(x, data.p02d, data.k);

	// Perform flow
	float H = PointSetHamiltonianSystem_CUDA::FlowHamiltonian_CUDA2D_Restricted(data.q02d, data.p02d, data.q12d, data.p12d, data.hq2d, data.hp2d, data.Qt2d, data.Pt2d, param.sigma, data.k, data.N, 1024, true, true);

	// Compute the landmark errors
	float fnorm_sq = PointSetHamiltonianSystem_CUDA::landmarkError_CUDA2D(data.q12d, data.qT2d, data.alpha2d, data.k, true);

	// Compute the landmark part of the objective
	if(f)
		*f = (double)(H + param.lambda * 0.5 * fnorm_sq);

	if(g){
		PointSetHamiltonianSystem_CUDA::FlowGradientBackward_CUDA2D_Restricted(data.Qt2d, data.Pt2d, data.alpha2d, data.beta2d, data.grad2d, (float)param.sigma, data.k, data.N, 1024, true);

		// Compute hp at (qT, p0)
		PointSetHamiltonianSystem_CUDA::ComputeHamiltonianJet_CUDA2D_Restricted(data.Qt2d[data.N - 1], data.p02d,
				data.hq2d, data.hp2d, param.sigma, data.k, 1024, true);

		PointSetHamiltonianSystem_CUDA::combineGradient_CUDA2D(data.grad2d, data.hp2d, data.k, param.lambda, true);

		// Pack the gradient into the output vector
		vnl_vector<double> gg(2*data.k);
		Float2DVec::wide2tall(data.grad2d, gg, data.k);
		*g = gg;
	}
}

void PointSetShootingCostFunction_CUDA::compute3D(vnl_vector<double> const& x, double *f, vnl_vector<double>* g){
	// Forget memory control here. Let's do it outside here
	// Initialize the p0-vector
	Float2DVec::tall2wide(x, data.p03d, data.k); // Debug: tall2wide seems normal

	float H = PointSetHamiltonianSystem_CUDA::FlowHamiltonian_CUDA3D_Restricted(data.q03d, data.p03d, data.q13d, data.p13d, data.hq3d, data.hp3d, data.Qt3d, data.Pt3d, param.sigma, data.k, data.N, 128, true, true);

	// Compute the landmark errors
	float fnorm_sq = PointSetHamiltonianSystem_CUDA::landmarkError_CUDA3D(data.q13d, data.qT3d, data.alpha3d, data.k, true);

	// Compute the landmark part of the objective
	if(f)
		*f = (double)(H + param.lambda * 0.5 * fnorm_sq);

	if(g){
		PointSetHamiltonianSystem_CUDA::FlowGradientBackward_CUDA3D_Restricted(data.Qt3d, data.Pt3d, data.alpha3d, data.beta3d, data.grad3d, (float)param.sigma, data.k, data.N, 128, true);

		//Compute hp at (qT, p0)
		PointSetHamiltonianSystem_CUDA::ComputeHamiltonianJet_CUDA3D_Restricted(data.Qt3d[data.N - 1], data.p03d,
				data.hq3d, data.hp3d, param.sigma, data.k, 1024, true);

		PointSetHamiltonianSystem_CUDA::combineGradient_CUDA3D(data.grad3d, data.hp3d, data.k, param.lambda, true);

		// Pack the gradient into the output vector
		vnl_vector<double> gg(3*data.k);
		Float2DVec::wide2tall(data.grad3d, gg, data.k);
		*g = gg;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
PointSetShootingTransversalityCostFunction_CUDA::PointSetShootingTransversalityCostFunction_CUDA(
		const ShootingParameters &param, const ShootingData &data)
: vnl_cost_function(){
	this->param = param; // Set parameters
	this->data = data; // Set data
	this->dim = data.k * data.VDIM; // Set number of variable to optimize

	// Initialize alpha and beta
	if (data.VDIM==2){
		checkCudaErrors(cudaMemset(data.alpha2d, 0, data.k*sizeof(float2)));
		checkCudaErrors(cudaMemset(data.beta2d, 0, data.k*sizeof(float2)));
	}else if (data.VDIM==3){
		checkCudaErrors(cudaMemset(data.alpha3d, 0, data.k*sizeof(float3)));
		checkCudaErrors(cudaMemset(data.beta3d, 0, data.k*sizeof(float3)));
	}else{
		printf("Dimension is not 2 or 3 at line number %d in in function %s, file %s\n", __LINE__, __func__, __FILE__);
		exit(-1);
	}

	// Initialize p0 - which is useless but in the original code
	if (data.VDIM==2){
		PointSetHamiltonianSystem_CUDA::initP_CUDA2D(data.q02d, data.qT2d, data.p02d, data.N, data.k, true);
	}
	else if (data.VDIM==3){
		PointSetHamiltonianSystem_CUDA::initP_CUDA3D(data.q03d, data.qT3d, data.p03d, data.N, data.k, true);
	}
	else{
		printf("Dimension is not 2 or 3 at line number %d in in function %s, file %s\n", __LINE__, __func__, __FILE__);
		exit(-1);
	}
	checkCudaErrors(cudaDeviceSynchronize());
}

void PointSetShootingTransversalityCostFunction_CUDA::compute(vnl_vector<double> const& x, double *f, vnl_vector<double>* g){
	checkCudaErrors(cudaDeviceSynchronize());
	if (this->data.VDIM ==2 )
		compute2D(x, f, g);
	else if (this->data.VDIM ==3 )
		compute3D(x, f, g);
	else{
		printf("Dimension is not 2 or 3 at line number %d in in function %s, file %s\n", __LINE__, __func__, __FILE__);
		exit(-1);
	}
	checkCudaErrors(cudaDeviceSynchronize());
}

void PointSetShootingTransversalityCostFunction_CUDA::compute3D(vnl_vector<double> const& x, double *f, vnl_vector<double>* g){
	// Initialize the p0-vector
	Float2DVec::tall2wide(x, data.p03d, data.k);

	// Perform flow
	float H = PointSetHamiltonianSystem_CUDA::FlowHamiltonian_CUDA3D(data.q03d, data.p03d, data.q13d, data.p13d, data.hq3d, data.hp3d, data.Qt3d, data.Pt3d, param.sigma, data.k, data.N, true, true);

	// Compute G and alpha/beta
	float Gnorm_sq, dsq;
	PointSetHamiltonianSystem_CUDA::GAlphaBeta_CUDA3D(data.q13d, data.qT3d, data.p13d,
			data.alpha3d, data.beta3d, Gnorm_sq, dsq, param.lambda, data.k, true);
	if(f)
		*f = (double)(0.5 * Gnorm_sq);

	if(g){
		// Multiply gradient of f. wrt q1 (alpha) by Jacobian of q1 wrt p0
		PointSetHamiltonianSystem_CUDA::FlowGradientBackward_CUDA3D(data.Qt3d, data.Pt3d, data.alpha3d, data.beta3d, data.grad3d, (float)param.sigma, data.k, data.N, true);

		// Pack the gradient into the output vector
		vnl_vector<double> gg(3*data.k);
		Float2DVec::wide2tall(data.grad3d, gg, data.k);
		*g = gg;
	}

	// Print the current state
	printf("H=%8.6f   Edist=%8.6f   E=%8.6f   |G|=%8.6f\n",
			H, 0.5 * param.lambda * dsq, H + 0.5 * param.lambda * dsq, sqrt(Gnorm_sq));
}

void PointSetShootingTransversalityCostFunction_CUDA::compute2D(vnl_vector<double> const& x, double *f, vnl_vector<double>* g){
	// Initialize the p0-vector
	Float2DVec::tall2wide(x, data.p02d, data.k);

	// Perform flow
	float H = PointSetHamiltonianSystem_CUDA::FlowHamiltonian_CUDA2D(data.q02d, data.p02d, data.q12d, data.p12d, data.hq2d, data.hp2d, data.Qt2d, data.Pt2d, param.sigma, data.k, data.N, true, true);

	// Compute G and alpha/beta
	float Gnorm_sq, dsq;
	PointSetHamiltonianSystem_CUDA::GAlphaBeta_CUDA2D(data.q12d, data.qT2d, data.p12d,
			data.alpha2d, data.beta2d, Gnorm_sq, dsq, param.lambda, data.k, true);
	if(f)
		*f = (double)(0.5 * Gnorm_sq);

	if(g){
		// Multiply gradient of f. wrt q1 (alpha) by Jacobian of q1 wrt p0
		PointSetHamiltonianSystem_CUDA::FlowGradientBackward_CUDA2D(data.Qt2d, data.Pt2d, data.alpha2d, data.beta2d, data.grad2d, (float)param.sigma, data.k, data.N, true);

		// Pack the gradient into the output vector
		vnl_vector<double> gg(2*data.k);
		Float2DVec::wide2tall(data.grad2d, gg, data.k);
		*g = gg;
	}

	// Print the current state
	printf("H=%8.6f   Edist=%8.6f   E=%8.6f   |G|=%8.6f\n",
			H, 0.5 * param.lambda * dsq, H + 0.5 * param.lambda * dsq, sqrt(Gnorm_sq));

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Class functions of the shooting problem class */
// TODO: implement this if it turns out useful
void PointSetShootingProblem_CUDA::minimize_Allassonniere(const ShootingParameters &param, const ShootingData &data){
}

void PointSetShootingProblem_CUDA::minimize_QuasiAllassonniere(
		const ShootingParameters &param,
		const ShootingData &data){
	// Create the minimization problem
	PointSetShootingTransversalityCostFunction_CUDA cost_fn(param, data);
	vnl_vector<double> x(data.VDIM * data.k);

	// Create initial/final solution
	if (data.VDIM==2){
		PointSetHamiltonianSystem_CUDA::initP_CUDA2D(data.q02d, data.qT2d, data.p02d, data.N, data.k, true);
		Float2DVec::wide2tall(data.p02d, x, data.k);
	}
	else if (data.VDIM==3){
		PointSetHamiltonianSystem_CUDA::initP_CUDA3D(data.q03d, data.qT3d, data.p03d, data.N, data.k, true);
		Float2DVec::wide2tall(data.p03d, x, data.k);
	}
	else{
		printf("Dimension is not 2 or 3 at line number %d in in function %s, file %s\n", __LINE__, __func__, __FILE__);
		exit(-1);
	}

	// Solve the minimization problem
	vnl_lbfgsb optimizer(cost_fn);
	optimizer.set_f_tolerance(1e-9);
	optimizer.set_x_tolerance(1e-4);
	optimizer.set_g_tolerance(1e-6);
	optimizer.set_trace(true);
	optimizer.set_max_function_evals(param.iter_newton);
	optimizer.minimize(x);

	// Take the optimal solution
	if (data.VDIM==2){
		Float2DVec::tall2wide(x, data.p02d, data.k);
	}
	else if (data.VDIM==3){
		Float2DVec::tall2wide(x, data.p03d, data.k);
	}
	else{
		printf("Dimension is not 2 or 3 at line number %d in in function %s, file %s\n", __LINE__, __func__, __FILE__);
		exit(-1);
	}
}

void PointSetShootingProblem_CUDA::minimize_gradient(const ShootingParameters &param,
		const ShootingData &data){
	// Create the minimization problem
	PointSetShootingCostFunction_CUDA cost_fn(param, data);
	vnl_vector<double> x(data.VDIM * data.k);

	// Create initial/final solution
	if (data.VDIM==2){
		PointSetHamiltonianSystem_CUDA::initP_CUDA2D(data.q02d, data.qT2d, data.p02d, data.N, data.k, true);
		Float2DVec::wide2tall(data.p02d, x, data.k);
	}
	else if (data.VDIM==3){
		PointSetHamiltonianSystem_CUDA::initP_CUDA3D(data.q03d, data.qT3d, data.p03d, data.N, data.k, true);
		Float2DVec::wide2tall(data.p03d, x, data.k); // Debug: wide2tall works!
		// Debug: x survive till this point
	}
	else{
		printf("Dimension is not 2 or 3 at line number %d in in function %s, file %s\n", __LINE__, __func__, __FILE__);
		exit(-1);
	}

	// Solve the minimization problem
	vnl_lbfgsb optimizer(cost_fn);
	optimizer.set_f_tolerance(1e-9);
	optimizer.set_x_tolerance(1e-4);
	optimizer.set_g_tolerance(1e-6);
	optimizer.set_trace(true);
	optimizer.set_max_function_evals(param.iter_grad);

	// vnl_conjugate_gradient optimizer(cost_fn);
	// optimizer.set_trace(true);
	optimizer.minimize(x);

	// Take the optimal solution
	if (data.VDIM==2){
		Float2DVec::tall2wide(x, data.p02d, data.k);
	}
	else if (data.VDIM==3){
		Float2DVec::tall2wide(x, data.p03d, data.k);
	}
	else{
		printf("Dimension is not 2 or 3 at line number %d in in function %s, file %s\n", __LINE__, __func__, __FILE__);
		exit(-1);
	}

}


void PointSetShootingProblem_CUDA::ReadVTKPointSetWithSampling( const ShootingParameters &param, ShootingData &data, vtkPolyData * pTemplate, vtkPolyData *pTarget ){
	// Get the number of vertices and dimensionality
	Check::check(pTemplate->GetNumberOfPoints() == pTarget->GetNumberOfPoints(), "Meshes don't match");

	data.np = pTemplate->GetNumberOfPoints();
	data.k = data.np;

	printf("Original data is %d points \n", data.np);

	printf("Checking right after read\n");
	for(int i = 0; i < 1; i++){
		pTemplate->GetPoint(i);
	}

	printf("Done checking right after read\n");

	// Mapping of matrix index to vertex - start with identity mapping
	data.index = std::vector<int>(data.np);
	for(int i = 0; i < data.np; i++)
		data.index[i] = i;

	// Downsample meshes if needed
	if(param.downsample < 1.0){
		// Downsample by random shuffle - not very efficient, just lazy
		data.k = (int)(param.downsample * data.np);
		std::random_shuffle(data.index.begin(), data.index.end());
	}
	else if(param.qcdiv.size()){
		// Store the index of each input point in VTK mesh
		vtkSmartPointer<vtkIntArray> arr_idx = vtkIntArray::New();
		arr_idx->SetNumberOfComponents(1);
		arr_idx->SetNumberOfTuples(data.np);
		arr_idx->SetName("Index");
		for(int i = 0; i < data.np; i++)
			arr_idx->SetTuple1(i, i);
		pTemplate->GetPointData()->AddArray(arr_idx);

		// Perform quadric clustering
		vtkSmartPointer<vtkQuadricClustering> qc = vtkQuadricClustering::New();
		qc->SetInputData(pTemplate);
		qc->SetUseInputPoints(1);

		// Set the divisions array - it must be 3
		int div[3];
		for(int a = 0; a < 3; a++)
			div[a] = (a < param.qcdiv.size()) ? param.qcdiv[a] : 1;
		qc->SetNumberOfDivisions(div);

		// Run the filter
		qc->Update();

		// Generate the index
		vtkPolyData *qc_result = qc->GetOutput();
		vtkDataArray *qc_index = qc_result->GetPointData()->GetArray("Index");

		data.k = qc_result->GetNumberOfPoints();
		for(int i = 0; i < data.k; i++)
			data.index[i] = qc_index->GetTuple1(i);
	}

	// Report number of actual vertices being used
	printf("Performing geodesic shooting with %d landmarks\n", data.k);

}


void PointSetShootingProblem_CUDA::AllocateCUDAMemAndCopyData(const ShootingParameters &param, ShootingData &data, vtkPolyData * pTemplate, vtkPolyData *pTarget){
	// Landmarks and initial momentum
	// TODO: change the input format here

	if (data.VDIM==2){
		float2 * h_q0 = new float2[data.k];
		float2 * h_qT = new float2[data.k];
		float2 * h_p0 = new float2[data.k];

		for(int i = 0; i < data.k; i++){
			h_q0[i].x = pTemplate->GetPoint(data.index[i])[0];
			h_q0[i].y = pTemplate->GetPoint(data.index[i])[1];

			h_qT[i].x = pTarget->GetPoint(data.index[i])[0];
			h_qT[i].y = pTarget->GetPoint(data.index[i])[1];

			h_p0[i].x = ( h_qT[i].x - h_q0[i].x ) / (float)param.N;
			h_p0[i].y = ( h_qT[i].y - h_q0[i].y ) / (float)param.N;
		}

		checkCudaErrors(cudaMalloc((void **)&data.q02d, data.k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&data.qT2d, data.k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&data.p02d, data.k*sizeof(float2)));

		checkCudaErrors(cudaMemcpy(data.q02d, h_q0, data.k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(data.qT2d, h_qT, data.k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(data.p02d, h_p0, data.k * sizeof(float2), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMalloc((void **)&data.alpha2d, data.k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&data.beta2d, data.k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&data.dalpha2d, data.k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&data.dbeta2d, data.k*sizeof(float2)));

		checkCudaErrors(cudaMalloc((void **)&data.H_iter, sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&data.grad2d, data.k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&data.grad_linear_2d, 2*data.k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&data.x2d, 2*data.k*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&data.hq2d, data.k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&data.hp2d, data.k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&data.q12d, data.k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&data.p12d, data.k*sizeof(float2)));

		data.Qt2d = std::vector<float2*>(data.N);
		data.Pt2d = std::vector<float2*>(data.N);

		for (int i=0; i<data.N; i++){
			float2 * d_q;
			float2 * d_p;
			checkCudaErrors(cudaMalloc((void **)&d_q, data.k*sizeof(float2)));
			checkCudaErrors(cudaMalloc((void **)&d_p, data.k*sizeof(float2)));
			data.Qt2d[i] = d_q;
			data.Pt2d[i] = d_p;
		}

		delete [] h_q0;
		delete [] h_qT;
		delete [] h_p0;

	}else if (data.VDIM == 3){

		printf("Allocating 3d memory\n");

		float3 * h_q0 = new float3[data.k];
		float3 * h_qT = new float3[data.k];
		float3 * h_p0 = new float3[data.k];

		for(int i = 0; i < data.k; i++){
			h_q0[i].x = pTemplate->GetPoint(data.index[i])[0];
			h_q0[i].y = pTemplate->GetPoint(data.index[i])[1];
			h_q0[i].z = pTemplate->GetPoint(data.index[i])[2];

			h_qT[i].x = pTarget->GetPoint(data.index[i])[0];
			h_qT[i].y = pTarget->GetPoint(data.index[i])[1];
			h_qT[i].z = pTarget->GetPoint(data.index[i])[2];

			h_p0[i].x = ( h_qT[i].x - h_q0[i].x ) / (float)param.N;
			h_p0[i].y = ( h_qT[i].y - h_q0[i].y ) / (float)param.N;
			h_p0[i].z = ( h_qT[i].z - h_q0[i].z ) / (float)param.N;
		}
		printf("Done copying data in host\n");

		checkCudaErrors(cudaMalloc((void **)&data.q03d, data.k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&data.qT3d, data.k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&data.p03d, data.k*sizeof(float3)));

		checkCudaErrors(cudaMemcpy(data.q03d, h_q0, data.k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(data.qT3d, h_qT, data.k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(data.p03d, h_p0, data.k * sizeof(float3), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMalloc((void **)&data.alpha3d, data.k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&data.beta3d, data.k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&data.dalpha3d, data.k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&data.dbeta3d, data.k*sizeof(float3)));

		checkCudaErrors(cudaMalloc((void **)&data.H_iter, sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&data.grad3d, data.k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&data.grad_linear_3d, 3*data.k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&data.x3d, 3*data.k*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&data.hq3d, data.k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&data.hp3d, data.k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&data.q13d, data.k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&data.p13d, data.k*sizeof(float3)));

		data.Qt3d = std::vector<float3*>(data.N);
		data.Pt3d = std::vector<float3*>(data.N);

		for (int i=0; i<data.N; i++){
			float3 * d_q;
			float3 * d_p;
			checkCudaErrors(cudaMalloc((void **)&d_q, data.k*sizeof(float3)));
			checkCudaErrors(cudaMalloc((void **)&d_p, data.k*sizeof(float3)));
			data.Qt3d[i] = d_q;
			data.Pt3d[i] = d_p;
		}

		delete [] h_q0;
		delete [] h_qT;
		delete [] h_p0;

	}else {
		printf("Dimension is not 2 or 3 at line number %d in in function %s, file %s\n", __LINE__, __func__, __FILE__);
		exit(-1);
	}

	printf("Done Allocating 3d memory\n");

}

void PointSetShootingProblem_CUDA::WriteVTKPOintSet( const ShootingParameters &param, ShootingData &data,
		vtkPolyData * pTemplate, vtkPolyData *pTarget){

	vtkDoubleArray *arr_p = vtkDoubleArray::New();
	arr_p->SetNumberOfComponents(data.VDIM);
	arr_p->SetNumberOfTuples(data.np);
	arr_p->SetName("InitialMomentum");
	for(unsigned int a = 0; a < data.VDIM; a++)
		arr_p->FillComponent(a, 0);

	if (data.VDIM==2){
		float2 * h_p0 = new float2[data.k];
		checkCudaErrors(cudaMemcpy(h_p0, data.p02d, data.k * sizeof(float2), cudaMemcpyDeviceToHost));
		for(unsigned int i = 0; i < data.k; i++){
			arr_p->SetComponent(data.index[i], 0, h_p0[i].x);
		}
		for(unsigned int i = 0; i < data.k; i++){
			arr_p->SetComponent(data.index[i], 1, h_p0[i].y);
		}
		delete [] h_p0;

	}else if (data.VDIM==3) {
		float3 * h_p0 = new float3[data.k];
		checkCudaErrors(cudaMemcpy(h_p0, data.p03d, data.k * sizeof(float3), cudaMemcpyDeviceToHost));
		for(unsigned int i = 0; i < data.k; i++){
			arr_p->SetComponent(data.index[i], 0, h_p0[i].x);
		}
		for(unsigned int i = 0; i < data.k; i++){
			arr_p->SetComponent(data.index[i], 1, h_p0[i].y);
		}
		for(unsigned int i = 0; i < data.k; i++){
			arr_p->SetComponent(data.index[i], 2, h_p0[i].z);
		}
		delete [] h_p0;

	} else {
		printf("Dimension is not 2 or 3 at line number %d in in function %s, file %s\n", __LINE__, __func__, __FILE__);
		exit(-1);
	}

	pTemplate->GetPointData()->AddArray(arr_p);

	/* Output the final result */
	WriteVTKData(pTemplate, param.fnOutput);

	// If saving paths requested
	if(param.fnOutputPaths.size()){
		// Create and flow a system
		if (data.VDIM==2){
			PointSetHamiltonianSystem_CUDA::FlowHamiltonian_CUDA2D(data.q02d, data.p02d, data.q12d, data.p12d,
					data.hq2d, data.hp2d,
					data.Qt2d, data.Pt2d, param.sigma, data.k, data.N, true, false);

			// Apply the flow to the points in the rest of the mesh
			vtkDoubleArray *arr_v = vtkDoubleArray::New();
			arr_v->SetNumberOfComponents(data.VDIM);
			arr_v->SetNumberOfTuples(data.np);
			arr_v->SetName("Velocity");
			pTemplate->GetPointData()->AddArray(arr_v);

			// Apply Euler method to the mesh points
			double dt = 1.0/ (float)(data.N-1);
			for(unsigned int t = 1; t < param.N; t++){
				for(unsigned int i = 0; i < data.np; i++){
					float2 qi, vi;
					float qqi[2];
					qi.x = pTemplate->GetPoint(i)[0];
					qi.y = pTemplate->GetPoint(i)[1];

					// Interpolate the velocity at each mesh point
					PointSetHamiltonianSystem_CUDA::InterpolateVelocity_CUDA2D(t-1, qi, vi, data.Qt2d, data.Pt2d, param.sigma, data.k, true);

					// Update the position using Euler's method
					qqi[0] = qi.x + dt * vi.x;
					qqi[1] = qi.y + dt * vi.y;

					pTemplate->GetPoints()->SetPoint(i, qqi);
				}
				// Output the intermediate mesh
				char buffer[1024];
				sprintf(buffer, param.fnOutputPaths.c_str(), t);
				WriteVTKData(pTemplate, buffer);

			}

		}else if (data.VDIM==3){
			PointSetHamiltonianSystem_CUDA::FlowHamiltonian_CUDA3D(data.q03d, data.p03d, data.q13d, data.p13d,
					data.hq3d, data.hp3d,
					data.Qt3d, data.Pt3d, param.sigma, data.k, data.N, true, false);

			// Apply the flow to the points in the rest of the mesh
			vtkDoubleArray *arr_v = vtkDoubleArray::New();
			arr_v->SetNumberOfComponents(data.VDIM);
			arr_v->SetNumberOfTuples(data.np);
			arr_v->SetName("Velocity");
			pTemplate->GetPointData()->AddArray(arr_v);

			// Apply Euler method to the mesh points
			double dt = 1.0/ (float)(data.N-1);
			for(unsigned int t = 1; t < param.N; t++){
				for(unsigned int i = 0; i < data.np; i++){
					float3 qi, vi;
					float qqi[3];
					qi.x = pTemplate->GetPoint(i)[0];
					qi.y = pTemplate->GetPoint(i)[1];
					qi.z = pTemplate->GetPoint(i)[2];

					// Interpolate the velocity at each mesh point
					PointSetHamiltonianSystem_CUDA::InterpolateVelocity_CUDA3D(t-1, qi, vi, data.Qt3d, data.Pt3d, param.sigma, data.k, true);

					// Update the position using Euler's method
					qqi[0] = qi.x + dt * vi.x;
					qqi[1] = qi.y + dt * vi.y;
					qqi[2] = qi.z + dt * vi.z;

					pTemplate->GetPoints()->SetPoint(i, qqi);
				}

				// Output the intermediate mesh
				char buffer[1024];
				sprintf(buffer, param.fnOutputPaths.c_str(), t);
				WriteVTKData(pTemplate, buffer);
			}
		} else{
			printf("Dimension is not 2 or 3 at line number %d in in function %s, file %s\n", __LINE__, __func__, __FILE__);
			exit(-1);

		}

	}
}


void PointSetShootingProblem_CUDA::FreeCUDAMem(const ShootingParameters &param, ShootingData &data){
	if (data.VDIM==2){
		checkCudaErrors(cudaFree(data.q02d));
		checkCudaErrors(cudaFree(data.qT2d));
		checkCudaErrors(cudaFree(data.p02d));

		checkCudaErrors(cudaFree(data.alpha2d));
		checkCudaErrors(cudaFree(data.beta2d));
		checkCudaErrors(cudaFree(data.dalpha2d));
		checkCudaErrors(cudaFree(data.dbeta2d));

		checkCudaErrors(cudaFree(data.H_iter));
		checkCudaErrors(cudaFree(data.grad2d));
		checkCudaErrors(cudaFree(data.grad_linear_2d));
		checkCudaErrors(cudaFree(data.x2d));

		checkCudaErrors(cudaFree(data.hq2d));
		checkCudaErrors(cudaFree(data.hp2d));
		checkCudaErrors(cudaFree(data.q12d));
		checkCudaErrors(cudaFree(data.p12d));

		for (int i=0; i<data.N; i++){
			checkCudaErrors(cudaFree(data.Qt2d[i]));
			checkCudaErrors(cudaFree(data.Pt2d[i]));
		}

	}else if (data.VDIM == 3){
		checkCudaErrors(cudaFree(data.q03d));
		checkCudaErrors(cudaFree(data.qT3d));
		checkCudaErrors(cudaFree(data.p03d));

		checkCudaErrors(cudaFree(data.alpha3d));
		checkCudaErrors(cudaFree(data.beta3d));
		checkCudaErrors(cudaFree(data.dalpha3d));
		checkCudaErrors(cudaFree(data.dbeta3d));

		checkCudaErrors(cudaFree(data.H_iter));
		checkCudaErrors(cudaFree(data.grad3d));
		checkCudaErrors(cudaFree(data.grad_linear_3d));
		checkCudaErrors(cudaFree(data.x3d));

		checkCudaErrors(cudaFree(data.hq3d));
		checkCudaErrors(cudaFree(data.hp3d));
		checkCudaErrors(cudaFree(data.q13d));
		checkCudaErrors(cudaFree(data.p13d));

		for (int i=0; i<data.N; i++){
			checkCudaErrors(cudaFree(data.Qt3d[i]));
			checkCudaErrors(cudaFree(data.Pt3d[i]));
		}

	}else {
		printf("Dimension is not 2 or 3 at line number %d in in function %s, file %s\n", __LINE__, __func__, __FILE__);
		exit(-1);
	}
}


void PointSetShootingProblem_CUDA::shooting(const ShootingParameters &param){
	/* Read the datasets */
	vtkPolyData *pTemplate;
	vtkPolyData *pTarget;
	/* Read the datasets and put them into a vtk object pointer- these two pointers seems to be alive only in the function call */
	pTemplate = ReadVTKData(param.fnTemplate);
	pTarget = ReadVTKData(param.fnTarget);

	ShootingData data;

	data.N = param.N;
	data.VDIM = param.dim;

	// Read in the data and do some random sampling on it
	PointSetShootingProblem_CUDA::ReadVTKPointSetWithSampling( param, data, pTemplate, pTarget );
	PointSetShootingProblem_CUDA::AllocateCUDAMemAndCopyData( param, data, pTemplate, pTarget );

	/* Actual gradient descend */
	// Run some iterations of gradient descent

	// Do timing

	long long tic = GetTimeMs64();
	if(param.iter_grad > 0){
		PointSetShootingProblem_CUDA::minimize_gradient(param, data);
		//		PointSetShootingProblem_CUDA::minimize_gradient_CudaLBFGS(param, data);
	}
	long long toc = GetTimeMs64();

	printf("The optimization takes %d ms.\n", toc-tic);

	// Note that the newtonian / allenXXX update is disabled here.
	PointSetShootingProblem_CUDA::WriteVTKPOintSet( param, data, pTemplate, pTarget);

	PointSetShootingProblem_CUDA::FreeCUDAMem( param, data );

}

void PointSetShootingProblem_CUDA::AllocateJustForShooting(const ShootingParameters &param, ShootingData &data,
		vtkPolyData * mesh){
	// Landmarks and initial momentum
	// TODO: change the input format here

	data.np = mesh->GetNumberOfPoints();
	data.k = data.np;

//	printf("k is %d\n", (int)data.k );

	if(!mesh)
	{
		cerr << "Failed to read mesh from " << param.momemtumPaths << endl;
		return;
	}

	// Read the momentum field
	vtkDataArray *arr_p0 = mesh->GetPointData()->GetArray("InitialMomentum");
	if(!arr_p0 || arr_p0->GetNumberOfComponents() != param.dim)
	{
		cerr << "Failed to read initial momentum from " << param.momemtumPaths << endl;
		return;
	}

	// Count the number of non-null entries
	vector<unsigned int> index;
	for(unsigned int i = 0; i < arr_p0->GetNumberOfTuples(); i++)
	{
		bool has_value = 1;
		for(unsigned int a = 0; a < param.dim; a++)
		{
			if(std::isnan(arr_p0->GetComponent(i,a)))
			{
				has_value = 0;
				break;
			}
		}
		if(has_value)
			index.push_back(i);
	}

	if (data.VDIM==2){
		// Populate the q0 and p0 arrays
		float2 * h_q0 = new float2[data.k];
		float2 * h_qT = new float2[data.k];
		float2 * h_p0 = new float2[data.k];

		for(int i = 0; i < data.k; i++){
			h_q0[i].x = mesh->GetPoint(index[i])[0];
			h_q0[i].y = mesh->GetPoint(index[i])[1];

			h_qT[i].x = 0;
			h_qT[i].y = 0;

			h_p0[i].x = arr_p0->GetComponent(i,0);
			h_p0[i].y = arr_p0->GetComponent(i,1);
		}

		checkCudaErrors(cudaMalloc((void **)&data.q02d, data.k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&data.qT2d, data.k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&data.p02d, data.k*sizeof(float2)));

		checkCudaErrors(cudaMemcpy(data.q02d, h_q0, data.k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(data.qT2d, h_qT, data.k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(data.p02d, h_p0, data.k * sizeof(float2), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMalloc((void **)&data.alpha2d, data.k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&data.beta2d, data.k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&data.dalpha2d, data.k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&data.dbeta2d, data.k*sizeof(float2)));

		checkCudaErrors(cudaMalloc((void **)&data.H_iter, sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&data.grad2d, data.k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&data.grad_linear_2d, 2*data.k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&data.x2d, 2*data.k*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&data.hq2d, data.k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&data.hp2d, data.k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&data.q12d, data.k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&data.p12d, data.k*sizeof(float2)));

		data.Qt2d = std::vector<float2*>(data.N);
		data.Pt2d = std::vector<float2*>(data.N);

		for (int i=0; i<data.N; i++){
			float2 * d_q;
			float2 * d_p;
			checkCudaErrors(cudaMalloc((void **)&d_q, data.k*sizeof(float2)));
			checkCudaErrors(cudaMalloc((void **)&d_p, data.k*sizeof(float2)));
			data.Qt2d[i] = d_q;
			data.Pt2d[i] = d_p;
		}

		delete [] h_q0;
		delete [] h_qT;
		delete [] h_p0;

	}else if (data.VDIM == 3){

		printf("Allocating 3d memory\n");
		// Populate the q0 and p0 arrays

		float3 * h_q0 = new float3[data.k];
		float3 * h_qT = new float3[data.k];
		float3 * h_p0 = new float3[data.k];

		for(int i = 0; i < data.k; i++){
			h_q0[i].x = mesh->GetPoint(index[i])[0];
			h_q0[i].y = mesh->GetPoint(index[i])[1];
			h_q0[i].z = mesh->GetPoint(index[i])[2];

			h_qT[i].x = 0;
			h_qT[i].y = 0;
			h_qT[i].z = 0;

			h_p0[i].x = arr_p0->GetComponent(i,0);
			h_p0[i].y = arr_p0->GetComponent(i,1);
			h_p0[i].z = arr_p0->GetComponent(i,2);
		}
		printf("Done copying data in host\n");

		checkCudaErrors(cudaMalloc((void **)&data.q03d, data.k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&data.qT3d, data.k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&data.p03d, data.k*sizeof(float3)));

		checkCudaErrors(cudaMemcpy(data.q03d, h_q0, data.k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(data.qT3d, h_qT, data.k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(data.p03d, h_p0, data.k * sizeof(float3), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMalloc((void **)&data.alpha3d, data.k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&data.beta3d, data.k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&data.dalpha3d, data.k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&data.dbeta3d, data.k*sizeof(float3)));

		checkCudaErrors(cudaMalloc((void **)&data.H_iter, sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&data.grad3d, data.k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&data.grad_linear_3d, 3*data.k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&data.x3d, 3*data.k*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&data.hq3d, data.k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&data.hp3d, data.k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&data.q13d, data.k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&data.p13d, data.k*sizeof(float3)));

		data.Qt3d = std::vector<float3*>(data.N);
		data.Pt3d = std::vector<float3*>(data.N);

		for (int i=0; i<data.N; i++){
			float3 * d_q;
			float3 * d_p;
			checkCudaErrors(cudaMalloc((void **)&d_q, data.k*sizeof(float3)));
			checkCudaErrors(cudaMalloc((void **)&d_p, data.k*sizeof(float3)));
			data.Qt3d[i] = d_q;
			data.Pt3d[i] = d_p;
		}

		delete [] h_q0;
		delete [] h_qT;
		delete [] h_p0;

	}else {
		printf("Dimension is not 2 or 3 at line number %d in in function %s, file %s\n", __LINE__, __func__, __FILE__);
		exit(-1);
	}

	printf("Done Allocating 3d memory\n");

}

void PointSetShootingProblem_CUDA::WriteQ0(const ShootingParameters &param, ShootingData &data, vtkPolyData * mesh){
	vtkDoubleArray *arr_p = vtkDoubleArray::New();
	arr_p->SetNumberOfComponents(data.VDIM);
	arr_p->SetNumberOfTuples(data.np);
	arr_p->SetName("FinalPoints");
	for(unsigned int a = 0; a < data.VDIM; a++)
		arr_p->FillComponent(a, 0);

	if (data.VDIM==2){
		float2 * h_q1 = new float2[data.k];
		checkCudaErrors(cudaMemcpy(h_q1, data.q12d, data.k * sizeof(float2), cudaMemcpyDeviceToHost));
		for(unsigned int i = 0; i < data.k; i++){
			arr_p->SetComponent(i, 0, h_q1[i].x);
		}
		for(unsigned int i = 0; i < data.k; i++){
			arr_p->SetComponent(i, 1, h_q1[i].y);
		}
		delete [] h_q1;

	}else if (data.VDIM==3) {
		float3 * h_q1 = new float3[data.k];
		checkCudaErrors(cudaMemcpy(h_q1, data.q13d, data.k * sizeof(float3), cudaMemcpyDeviceToHost));

		printf("Copying\n");

		for(unsigned int i = 0; i < data.k; i++){
			arr_p->SetComponent(i, 0, h_q1[i].x);
		}
		for(unsigned int i = 0; i < data.k; i++){
			arr_p->SetComponent(i, 1, h_q1[i].y);
		}
		for(unsigned int i = 0; i < data.k; i++){
			arr_p->SetComponent(i, 2, h_q1[i].z);
		}

		printf("Done Copying\n");

		delete [] h_q1;

	} else {
		printf("Dimension is not 2 or 3 at line number %d in in function %s, file %s\n", __LINE__, __func__, __FILE__);
		exit(-1);
	}

	mesh->GetPointData()->AddArray(arr_p);

	/* Output the final result */
	WriteVTKData(mesh, param.q1outPaths);

}


void PointSetShootingProblem_CUDA::CalculateForward(const ShootingParameters &param){
	ShootingData data;
	data.N = param.N;
	data.VDIM = param.dim;

	vtkPolyData *mesh = ReadVTKData(param.momemtumPaths);

	// Allocate device mem and copy template pointset and momentum to device mem
	PointSetShootingProblem_CUDA::AllocateJustForShooting(param, data, mesh);

	// Flow forward
	printf("Flowing\n");
	float H = PointSetHamiltonianSystem_CUDA::FlowHamiltonian_CUDA3D_Restricted(data.q03d, data.p03d, data.q13d, data.p13d, data.hq3d, data.hp3d, data.Qt3d, data.Pt3d, param.sigma, data.k, data.N, 128, true, true);

	// Save
	printf("Saving\n");
	PointSetShootingProblem_CUDA::WriteQ0(param, data, mesh);

	// Free CUDA space
	printf("Releasing\n");
	PointSetShootingProblem_CUDA::FreeCUDAMem(param, data);
}
