#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "helper_functions.h"
#include "helper_cuda.h"

#include <stdio.h>

#include "../include/hamiltonian.h"
#include "hamiltonian_kernel.cu"
#include "../include/reducer.h"


void checkCublasErrors(cublasStatus_t ret){
	if (ret != CUBLAS_STATUS_SUCCESS){
		printf("cublasCreate returned error code %d, line(%d)\n", ret, __LINE__);
		exit(-1);
	}
}

/* Actual CUDA functions */
///////////////////////////////////////// 3D functions//////////////////////////////////////////////
float PointSetHamiltonianSystem_CUDA::ComputeHamiltonianJet_CUDA3D(float3 * h_q, float3 * h_p,
		float3 * h_hq, float3 * h_hp, float9 * h_hqq, float9 * h_hqp, float9 * h_hpp,
		float sigma, int k, bool flag_hessian, bool dataInDevice ){
	// Parameters
	float f = -0.5 / (sigma * sigma);
	long k2 = k*k;

	dim3 threads;
	dim3 blocks;

	// Initialize cublas
	cublasHandle_t handle;
	checkCublasErrors( cublasCreate(&handle) );
	//	cublasOperation_t trans = CUBLAS_OP_N; // AT  if transa == CUBLAS_OP_T
	cublasOperation_t trans = CUBLAS_OP_T; // AT  if transa == CUBLAS_OP_T
	float alf=1.0;
	float beta=0;

	// Some memory control stuff
	float3 * d_q;
	float3 * d_p;

	if (dataInDevice){
		d_q = h_q;
		d_p = h_p;
	} else {
		checkCudaErrors(cudaMalloc((void **)&d_q, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_p, k*sizeof(float3)));

		checkCudaErrors(cudaMemcpy(d_q, h_q, k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_p, h_p, k * sizeof(float3), cudaMemcpyHostToDevice));
	}


	// Start timer
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	// Initialize hamiltonian
	float H = 0.0;

	// allocate the memory
	float * d_pi_pj;
	float * d_pi_pj_g;
	float3 * d_dq;
	float * d_g;

	checkCudaErrors(cudaMalloc((void **)&d_pi_pj, k2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_pi_pj_g, k2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_dq, k2*sizeof(float3)));
	checkCudaErrors(cudaMalloc((void **)&d_g, k2*sizeof(float)));

	// Calculate the pi_pj, dq, g and (pi_pj * g)
	threads = dim3(16, 16, 1);
	blocks = dim3( (k+15)/16, (k+15)/16, 1);
	dqpipjKernel <<< blocks, threads >>> (d_q, d_dq, d_g, f, d_p, d_pi_pj, k);
	multiplyKernel2D <<< blocks, threads >>> (d_pi_pj, d_g, d_pi_pj_g, k, k);

	float * h_pi_pj_g = new float[k2];

	// Calculate H
	H = 0.5 * Reducer::reduce_sum_wrapper(k2, d_pi_pj_g);
	checkCudaErrors(cudaFree(d_pi_pj_g));

	// Calculate the 1st derivative
	//printf("Calculating 1st derivative...\n");
	float * d_pi_pj_g1_dq_x;
	float * d_pi_pj_g1_dq_y;
	float * d_pi_pj_g1_dq_z;

	float * d_p_g_x;
	float * d_p_g_y;
	float * d_p_g_z;

	checkCudaErrors(cudaMalloc((void **)&d_pi_pj_g1_dq_x, k2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_pi_pj_g1_dq_y, k2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_pi_pj_g1_dq_z, k2*sizeof(float)));

	checkCudaErrors(cudaMalloc((void **)&d_p_g_x, k2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_p_g_y, k2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_p_g_z, k2*sizeof(float)));

	// Precompute the terms that need to be added up
	threads = dim3(16, 16, 1);
	blocks = dim3( (k+15)/16, (k+15)/16, 1);
	hqhpPreComputeKernel  <<< blocks, threads >>>  ( d_pi_pj, d_dq, d_g, d_p, f,
			d_pi_pj_g1_dq_x, d_pi_pj_g1_dq_y, d_pi_pj_g1_dq_z,
			d_p_g_x, d_p_g_y, d_p_g_z, k);

	float * d_one;
	checkCudaErrors(cudaMalloc((void **)&d_one, k*sizeof(float)));
	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	onesKernel1D  <<< blocks, threads >>>  (d_one, k);

	// Allocate the memory
	float * d_hq_x;
	float * d_hq_y;
	float * d_hq_z;

	float * d_hp_x;
	float * d_hp_y;
	float * d_hp_z;

	checkCudaErrors(cudaMalloc((void **)&d_hq_x, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_hq_y, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_hq_z, k*sizeof(float)));

	checkCudaErrors(cudaMalloc((void **)&d_hp_x, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_hp_y, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_hp_z, k*sizeof(float)));

	// Use CUBLAS to multiply the terms by one vector to add up
	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_pi_pj_g1_dq_x, k, d_one, 1, &beta, d_hq_x, 1) );
	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_pi_pj_g1_dq_y, k, d_one, 1, &beta, d_hq_y, 1) );
	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_pi_pj_g1_dq_z, k, d_one, 1, &beta, d_hq_z, 1) );

	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_p_g_x, k, d_one, 1, &beta, d_hp_x, 1) );
	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_p_g_y, k, d_one, 1, &beta, d_hp_y, 1) );
	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_p_g_z, k, d_one, 1, &beta, d_hp_z, 1) );

	// clean up
	checkCudaErrors(cudaFree(d_pi_pj_g1_dq_x));
	checkCudaErrors(cudaFree(d_pi_pj_g1_dq_y));
	checkCudaErrors(cudaFree(d_pi_pj_g1_dq_z));

	checkCudaErrors(cudaFree(d_p_g_x));
	checkCudaErrors(cudaFree(d_p_g_y));
	checkCudaErrors(cudaFree(d_p_g_z));

	// TODO: copy the result back to host
	float3 * d_hq;
	float3 * d_hp;
	if (dataInDevice){
		d_hq = h_hq;
		d_hp = h_hp;
	} else {
		checkCudaErrors(cudaMalloc((void **)&d_hq, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_hp, k*sizeof(float3)));
	}

	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	Float2Float3Kernel1D  <<< blocks, threads >>>  ( d_hq_x, d_hq_y, d_hq_z, d_hq, k);
	Float2Float3Kernel1D  <<< blocks, threads >>>  ( d_hp_x, d_hp_y, d_hp_z, d_hp, k);

	checkCudaErrors(cudaFree(d_hq_x));
	checkCudaErrors(cudaFree(d_hq_y));
	checkCudaErrors(cudaFree(d_hq_z));

	checkCudaErrors(cudaFree(d_hp_x));
	checkCudaErrors(cudaFree(d_hp_y));
	checkCudaErrors(cudaFree(d_hp_z));


	// Some memory control stuff
	if (dataInDevice){
		// Do nothing. Duty to manage memory relies on outside code
	} else {
		checkCudaErrors(cudaMemcpy(h_hq, d_hq, k * sizeof(float3), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_hp, d_hp, k * sizeof(float3), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_hq));
		checkCudaErrors(cudaFree(d_hp));
	}


	//printf("Done 1st derivative.\n");

	// Calculate the 2nd derivatives
	if (flag_hessian){
		//printf("Calculating 2nd derivative...\n");
		//printf("Calculating hqq...\n");
		///////////////////////////////////////////////////////////////////////////////////////////////////////
		/* hqq */
		float * d_hqq_xx; float * d_hqq_xy; float * d_hqq_xz;
		float * d_hqq_yx; float * d_hqq_yy; float * d_hqq_yz;
		float * d_hqq_zx; float * d_hqq_zy; float * d_hqq_zz;

		// Allocate memory
		//printf("hqq: Allocating mem...\n");
		checkCudaErrors(cudaMalloc((void **)&d_hqq_xx, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqq_xy, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqq_xz, k2*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&d_hqq_yx, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqq_yy, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqq_yz, k2*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&d_hqq_zx, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqq_zy, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqq_zz, k2*sizeof(float)));

		// Precompute the terms
		//printf("hqq: Precomputing...\n");
		threads = dim3(16, 16, 1);
		blocks = dim3( (k+15)/16, (k+15)/16, 1);
		hqqPreComputeKernel  <<< blocks, threads >>>  (d_pi_pj, d_g, d_dq, f,
				d_hqq_xx, d_hqq_xy, d_hqq_xz,
				d_hqq_yx, d_hqq_yy, d_hqq_yz,
				d_hqq_zx, d_hqq_zy, d_hqq_zz, k);

		// The diagonal terms need sum - again use cublas
		float * d_hqq_diag_xx; float * d_hqq_diag_xy; float * d_hqq_diag_xz;
		float * d_hqq_diag_yx; float * d_hqq_diag_yy; float * d_hqq_diag_yz;
		float * d_hqq_diag_zx; float * d_hqq_diag_zy; float * d_hqq_diag_zz;

		checkCudaErrors(cudaMalloc((void **)&d_hqq_diag_xx, k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqq_diag_xy, k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqq_diag_xz, k*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&d_hqq_diag_yx, k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqq_diag_yy, k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqq_diag_yz, k*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&d_hqq_diag_zx, k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqq_diag_zy, k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqq_diag_zz, k*sizeof(float)));

		// cublas sum
		//printf("hqq: cublas sum...\n");
		float * d_mone;
		checkCudaErrors(cudaMalloc((void **)&d_mone, k*sizeof(float)));
		threads = dim3(256, 1, 1);
		blocks = dim3( (k+255)/256, 1, 1);
		fillKernel1D <<< blocks, threads >>> (d_mone, k, -1 );
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqq_xx, k, d_mone, 1, &beta, d_hqq_diag_xx, 1) );
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqq_xy, k, d_mone, 1, &beta, d_hqq_diag_xy, 1) );
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqq_xz, k, d_mone, 1, &beta, d_hqq_diag_xz, 1) );

		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqq_yx, k, d_mone, 1, &beta, d_hqq_diag_yx, 1) );
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqq_yy, k, d_mone, 1, &beta, d_hqq_diag_yy, 1) );
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqq_yz, k, d_mone, 1, &beta, d_hqq_diag_yz, 1) );

		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqq_zx, k, d_mone, 1, &beta, d_hqq_diag_zx, 1) );
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqq_zy, k, d_mone, 1, &beta, d_hqq_diag_zy, 1) );
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqq_zz, k, d_mone, 1, &beta, d_hqq_diag_zz, 1) );
		checkCudaErrors(cudaFree(d_mone));

		// Copy the diagonal terms into the matrix
		//printf("hqq: copy diagonal term...\n");
		threads = dim3(256, 1, 1);
		blocks = dim3( (k+255)/256, 1, 1);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqq_xx, d_hqq_diag_xx, k);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqq_xy, d_hqq_diag_xy, k);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqq_xz, d_hqq_diag_xz, k);

		copyToDiagonal  <<< blocks, threads >>>  (d_hqq_yx, d_hqq_diag_yx, k);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqq_yy, d_hqq_diag_yy, k);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqq_yz, d_hqq_diag_yz, k);

		copyToDiagonal  <<< blocks, threads >>>  (d_hqq_zx, d_hqq_diag_zx, k);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqq_zy, d_hqq_diag_zy, k);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqq_zz, d_hqq_diag_zz, k);

		checkCudaErrors(cudaFree(d_hqq_diag_xx));
		checkCudaErrors(cudaFree(d_hqq_diag_xy));
		checkCudaErrors(cudaFree(d_hqq_diag_xz));

		checkCudaErrors(cudaFree(d_hqq_diag_yx));
		checkCudaErrors(cudaFree(d_hqq_diag_yy));
		checkCudaErrors(cudaFree(d_hqq_diag_yz));

		checkCudaErrors(cudaFree(d_hqq_diag_zx));
		checkCudaErrors(cudaFree(d_hqq_diag_zy));
		checkCudaErrors(cudaFree(d_hqq_diag_zz));


		// copy the result back to host
		//printf("hqq: copy back the result...\n");
		float9 * d_hqq;
		if (dataInDevice){
			d_hqq = h_hqq;
		}else{
			checkCudaErrors(cudaMalloc((void **)&d_hqq, k2*sizeof(float9)));
		}
		//printf("hqq: Done allocate memory...\n");

		threads = dim3(16, 16, 1);
		blocks = dim3( (k+15)/16, (k+15)/16, 1);
		Float2Float9Kernel2D   <<< blocks, threads >>> (
				d_hqq_xx, d_hqq_xy, d_hqq_xz,
				d_hqq_yx, d_hqq_yy, d_hqq_yz,
				d_hqq_zx, d_hqq_zy, d_hqq_zz, d_hqq, k);

		//printf("hqq: Done copy 9 float to float9...\n");

		checkCudaErrors(cudaFree(d_hqq_xx));
		checkCudaErrors(cudaFree(d_hqq_xy));
		checkCudaErrors(cudaFree(d_hqq_xz));

		checkCudaErrors(cudaFree(d_hqq_yx));
		checkCudaErrors(cudaFree(d_hqq_yy));
		checkCudaErrors(cudaFree(d_hqq_yz));

		checkCudaErrors(cudaFree(d_hqq_zx));
		checkCudaErrors(cudaFree(d_hqq_zy));
		checkCudaErrors(cudaFree(d_hqq_zz));

		if (dataInDevice){
			// Do nothing. Duty to manage memory relies on outside code
		}else{
			checkCudaErrors(cudaMemcpy(h_hqq, d_hqq, k2 * sizeof(float9), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaFree(d_hqq));
		}
		//printf("hqq: Done copy back to host...\n");

		//printf("Done hqq.\n");
		//printf("Calculating hqp...\n");
		////////////////////////////////////////////////////////////////////////////////////////////////////
		/* hqp */
		float * d_hqp_xx; float * d_hqp_xy; float * d_hqp_xz;
		float * d_hqp_yx; float * d_hqp_yy; float * d_hqp_yz;
		float * d_hqp_zx; float * d_hqp_zy; float * d_hqp_zz;

		float * d_hqp_ii_xx; float * d_hqp_ii_xy; float * d_hqp_ii_xz;
		float * d_hqp_ii_yx; float * d_hqp_ii_yy; float * d_hqp_ii_yz;
		float * d_hqp_ii_zx; float * d_hqp_ii_zy; float * d_hqp_ii_zz;

		// Allocate memory
		checkCudaErrors(cudaMalloc((void **)&d_hqp_xx, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_xy, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_xz, k2*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&d_hqp_yx, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_yy, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_yz, k2*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&d_hqp_zx, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_zy, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_zz, k2*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&d_hqp_ii_xx, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_ii_xy, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_ii_xz, k2*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&d_hqp_ii_yx, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_ii_yy, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_ii_yz, k2*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&d_hqp_ii_zx, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_ii_zy, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_ii_zz, k2*sizeof(float)));

		threads = dim3(16, 16, 1);
		blocks = dim3( (k+15)/16, (k+15)/16, 1);
		hqpPreComputeKernel <<< blocks, threads >>> (d_p, d_g, f, d_dq,
				d_hqp_xx, d_hqp_xy, d_hqp_xz,
				d_hqp_yx, d_hqp_yy, d_hqp_yz,
				d_hqp_zx, d_hqp_zy, d_hqp_zz,
				d_hqp_ii_xx, d_hqp_ii_xy, d_hqp_ii_xz,
				d_hqp_ii_yx, d_hqp_ii_yy, d_hqp_ii_yz,
				d_hqp_ii_zx, d_hqp_ii_zy, d_hqp_ii_zz, k);

		// The diagonal terms need sum - again use cublas
		float * d_hqp_diag_xx;  float * d_hqp_diag_xy;  float * d_hqp_diag_xz;
		float * d_hqp_diag_yx;  float * d_hqp_diag_yy;  float * d_hqp_diag_yz;
		float * d_hqp_diag_zx;  float * d_hqp_diag_zy;  float * d_hqp_diag_zz;

		checkCudaErrors(cudaMalloc((void **)&d_hqp_diag_xx, k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_diag_xy, k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_diag_xz, k*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&d_hqp_diag_yx, k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_diag_yy, k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_diag_yz, k*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&d_hqp_diag_zx, k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_diag_zy, k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_diag_zz, k*sizeof(float)));

		// cublas sum
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqp_ii_xx, k, d_one, 1, &beta, d_hqp_diag_xx, 1) );
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqp_ii_xy, k, d_one, 1, &beta, d_hqp_diag_xy, 1) );
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqp_ii_xz, k, d_one, 1, &beta, d_hqp_diag_xz, 1) );

		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqp_ii_yx, k, d_one, 1, &beta, d_hqp_diag_yx, 1) );
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqp_ii_yy, k, d_one, 1, &beta, d_hqp_diag_yy, 1) );
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqp_ii_yz, k, d_one, 1, &beta, d_hqp_diag_yz, 1) );

		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqp_ii_zx, k, d_one, 1, &beta, d_hqp_diag_zx, 1) );
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqp_ii_zy, k, d_one, 1, &beta, d_hqp_diag_zy, 1) );
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqp_ii_zz, k, d_one, 1, &beta, d_hqp_diag_zz, 1) );

		// Release
		checkCudaErrors(cudaFree(d_hqp_ii_xx));
		checkCudaErrors(cudaFree(d_hqp_ii_xy));
		checkCudaErrors(cudaFree(d_hqp_ii_xz));

		checkCudaErrors(cudaFree(d_hqp_ii_yx));
		checkCudaErrors(cudaFree(d_hqp_ii_yy));
		checkCudaErrors(cudaFree(d_hqp_ii_yz));

		checkCudaErrors(cudaFree(d_hqp_ii_zx));
		checkCudaErrors(cudaFree(d_hqp_ii_zy));
		checkCudaErrors(cudaFree(d_hqp_ii_zz));

		// copy the diagonal terms into the matrix
		threads = dim3(256, 1, 1);
		blocks = dim3( (k+255)/256, 1, 1);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqp_xx, d_hqp_diag_xx, k);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqp_xy, d_hqp_diag_xy, k);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqp_xz, d_hqp_diag_xz, k);

		copyToDiagonal  <<< blocks, threads >>>  (d_hqp_yx, d_hqp_diag_yx, k);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqp_yy, d_hqp_diag_yy, k);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqp_yz, d_hqp_diag_yz, k);

		copyToDiagonal  <<< blocks, threads >>>  (d_hqp_zx, d_hqp_diag_zx, k);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqp_zy, d_hqp_diag_zy, k);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqp_zz, d_hqp_diag_zz, k);

		checkCudaErrors(cudaFree(d_hqp_diag_xx));
		checkCudaErrors(cudaFree(d_hqp_diag_xy));
		checkCudaErrors(cudaFree(d_hqp_diag_xz));

		checkCudaErrors(cudaFree(d_hqp_diag_yx));
		checkCudaErrors(cudaFree(d_hqp_diag_yy));
		checkCudaErrors(cudaFree(d_hqp_diag_yz));

		checkCudaErrors(cudaFree(d_hqp_diag_zx));
		checkCudaErrors(cudaFree(d_hqp_diag_zy));
		checkCudaErrors(cudaFree(d_hqp_diag_zz));

		// copy the result back to host
		float9 * d_hqp;
		if (dataInDevice){
			d_hqp = h_hqp;
		}else{
			checkCudaErrors(cudaMalloc((void **)&d_hqp, k2*sizeof(float9)));
		}

		threads = dim3(16, 16, 1);
		blocks = dim3( (k+15)/16, (k+15)/16, 1);
		Float2Float9Kernel2D   <<< blocks, threads >>> (
				d_hqp_xx, d_hqp_xy, d_hqp_xz,
				d_hqp_yx, d_hqp_yy, d_hqp_yz,
				d_hqp_zx, d_hqp_zy, d_hqp_zz, d_hqp, k);

		checkCudaErrors(cudaFree(d_hqp_xx));
		checkCudaErrors(cudaFree(d_hqp_xy));
		checkCudaErrors(cudaFree(d_hqp_xz));

		checkCudaErrors(cudaFree(d_hqp_yx));
		checkCudaErrors(cudaFree(d_hqp_yy));
		checkCudaErrors(cudaFree(d_hqp_yz));

		checkCudaErrors(cudaFree(d_hqp_zx));
		checkCudaErrors(cudaFree(d_hqp_zy));
		checkCudaErrors(cudaFree(d_hqp_zz));

		if (dataInDevice){
			// Do nothing. Duty to manage memory relies on outside code
		}else{
			checkCudaErrors(cudaMemcpy(h_hqp, d_hqp, k2 * sizeof(float9), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaFree(d_hqp));
		}

		//printf("Done hqp.\n");
		//printf("Calculating hpp...\n");

		////////////////////////////////////////////////////////////////////////////////////////////
		/* hpp */
		float * d_hpp_xx; float * d_hpp_yy; float * d_hpp_zz;
		checkCudaErrors(cudaMalloc((void **)&d_hpp_xx, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hpp_yy, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hpp_zz, k2*sizeof(float)));
		threads = dim3(16, 16, 1);
		blocks = dim3( (k+15)/16, (k+15)/16, 1);
		hppPreComputeKernel  <<< blocks, threads >>>  (d_g, d_hpp_xx, d_hpp_yy, d_hpp_zz, k);

		// copy the result back to host
		float * d_zero;
		checkCudaErrors(cudaMalloc((void **)&d_zero, k2*sizeof(float)));
		checkCudaErrors(cudaMemset(d_zero, 0, k2*sizeof(float)));

		float9 * d_hpp;
		if (dataInDevice){
			d_hpp = h_hpp;
		}else{
			checkCudaErrors(cudaMalloc((void **)&d_hpp, k2*sizeof(float9)));
		}

		Float2Float9Kernel2D   <<< blocks, threads >>> (
				d_hpp_xx, d_zero, d_zero,
				d_zero, d_hpp_yy, d_zero,
				d_zero, d_zero, d_hpp_zz, d_hpp, k);

		checkCudaErrors(cudaFree(d_zero));

		if (dataInDevice){
			// Do nothing. Duty to manage memory relies on outside code
		}else{
			checkCudaErrors(cudaMemcpy(h_hpp, d_hpp, k2 * sizeof(float9), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaFree(d_hpp));
		}

		//printf("Done hpp.\n");
		//printf("Done 2nd derivative.\n");

	}

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);

	double hamiltonian_time = sdkGetTimerValue(&hTimer);
	//printf("Hamiltonian takes %f ms.\n", hamiltonian_time);

	// Clean up
	if (dataInDevice){
		// Do nothing. Duty to manage memory relies on outside code
	} else {
		checkCudaErrors( cudaFree(d_p) );
		checkCudaErrors( cudaFree(d_q) );
	}
	checkCudaErrors(cudaFree(d_pi_pj));
	checkCudaErrors(cudaFree(d_dq));
	checkCudaErrors(cudaFree(d_g));
	checkCudaErrors(cudaFree(d_one));

	checkCublasErrors( cublasDestroy(handle) );

	checkCudaErrors(cudaDeviceSynchronize());

	return H;

}

void PointSetHamiltonianSystem_CUDA::ApplyHamiltonianHessianToAlphaBeta_CUDA3D(float3 * h_q, float3 * h_p,
		float3 * h_alpha, float3 * h_beta,
		float3 * h_dalpha, float3 * h_dbeta,
		float sigma, int k, bool dataInDevice ){

	// Some variable
	float f = -0.5 / (sigma * sigma);
	long k2 = k*k;

	dim3 threads;
	dim3 blocks;

	// variables
	float3 * d_q;
	float3 * d_p;
	float3 * d_alpha;
	float3 * d_beta;

	// Initialize cublas
	cublasHandle_t handle;
	checkCublasErrors( cublasCreate(&handle) );
	//	cublasOperation_t trans = CUBLAS_OP_N; // AT  if transa == CUBLAS_OP_T
	cublasOperation_t trans = CUBLAS_OP_T; // AT  if transa == CUBLAS_OP_T
	float alf=1.0;
	float beta=0;

	// Some memory control stuff
	if (dataInDevice){
		d_q = h_q;
		d_p = h_p;
		d_alpha = h_alpha;
		d_beta = h_beta;
	} else {
		checkCudaErrors(cudaMalloc((void **)&d_q, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_p, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_alpha, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_beta, k*sizeof(float3)));

		checkCudaErrors(cudaMemcpy(d_q, h_q, k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_p, h_p, k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_alpha, h_alpha, k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_beta, h_beta, k * sizeof(float3), cudaMemcpyHostToDevice));
	}

	// Start timer
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	// allocate the memory for these intermediate variables
	float * d_pi_pj;
	float3 * d_dq;
	float * d_g;
	float * d_one;

	checkCudaErrors(cudaMalloc((void **)&d_one, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_pi_pj, k2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_dq, k2*sizeof(float3)));
	checkCudaErrors(cudaMalloc((void **)&d_g, k2*sizeof(float)));

	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	onesKernel1D  <<< blocks, threads >>>  (d_one, k);


	// Calculate the pi_pj, dq, g and (pi_pj * g)
	threads = dim3(16, 16, 1);
	blocks = dim3( (k+15)/16, (k+15)/16, 1);
	dqpipjKernel <<< blocks, threads >>> (d_q, d_dq, d_g, f, d_p, d_pi_pj, k);

	// Calculate the dj-di
	threads = dim3(16, 16, 1);
	blocks = dim3( (k+15)/16, (k+15)/16, 1);
	float3 * d_dbji;
	checkCudaErrors(cudaMalloc((void **)&d_dbji, k2*sizeof(float3)));
	dbjiKernel <<< blocks, threads >>> ( d_beta, d_dbji, k );

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/* dalpha */
	// Precompute for the da and aa terms
	float * d_da_pre_x;
	float * d_da_pre_y;
	float * d_da_pre_z;

	checkCudaErrors(cudaMalloc((void **)&d_da_pre_x, k2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_da_pre_y, k2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_da_pre_z, k2*sizeof(float)));

	threads = dim3(16, 16, 1);
	blocks = dim3( (k+15)/16, (k+15)/16, 1);
	dalphaPrecomputeKernel <<< blocks, threads >>> (d_pi_pj, d_dq, d_g, d_dbji, f, k,
			d_da_pre_x, d_da_pre_y, d_da_pre_z,
			d_p, d_alpha);

	// Use cublas to sum
	float * d_da_x;
	float * d_da_y;
	float * d_da_z;

	checkCudaErrors(cudaMalloc((void **)&d_da_x, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_da_y, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_da_z, k*sizeof(float)));

	// cublas m * v
	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_da_pre_x, k, d_one, 1, &beta, d_da_x, 1) );
	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_da_pre_y, k, d_one, 1, &beta, d_da_y, 1) );
	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_da_pre_z, k, d_one, 1, &beta, d_da_z, 1) );

	checkCudaErrors( cudaFree(d_da_pre_x) );
	checkCudaErrors( cudaFree(d_da_pre_y) );
	checkCudaErrors( cudaFree(d_da_pre_z) );

	// 3 float to float3
	float3 * d_dalpha;
	if (dataInDevice){
		d_dalpha = h_dalpha;
	} else {
		checkCudaErrors(cudaMalloc((void **)&d_dalpha, k*sizeof(float3)));
	}

	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	Float2Float3Kernel1D <<< blocks, threads >>> ( d_da_x, d_da_y, d_da_z, d_dalpha, k);

	// copy the result back to host mem
	if (dataInDevice){
		// Do nothing. Duty to manage memory relies on outside code
	} else {
		checkCudaErrors(cudaMemcpy(h_dalpha, d_dalpha, k * sizeof(float3), cudaMemcpyDeviceToHost));
		checkCudaErrors( cudaFree(d_dalpha) );
	}

	checkCudaErrors( cudaFree(d_da_x) );
	checkCudaErrors( cudaFree(d_da_y) );
	checkCudaErrors( cudaFree(d_da_z) );

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/* dbeta */

	// precompute
	float * d_db_pre_x;
	float * d_db_pre_y;
	float * d_db_pre_z;

	checkCudaErrors(cudaMalloc((void **)&d_db_pre_x, k2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_db_pre_y, k2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_db_pre_z, k2*sizeof(float)));

	threads = dim3(16, 16, 1);
	blocks = dim3( (k+15)/16, (k+15)/16, 1);
	dbetaPrecomputeKernel <<< blocks, threads >>> ( d_p, d_dq, d_g, d_dbji, f, k,
			d_db_pre_x, d_db_pre_y, d_db_pre_z, d_alpha);

	// Use cublas to sum
	float * d_db_x;
	float * d_db_y;
	float * d_db_z;

	checkCudaErrors(cudaMalloc((void **)&d_db_x, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_db_y, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_db_z, k*sizeof(float)));

	// cublas m * v
	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_db_pre_x, k, d_one, 1, &beta, d_db_x, 1) );
	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_db_pre_y, k, d_one, 1, &beta, d_db_y, 1) );
	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_db_pre_z, k, d_one, 1, &beta, d_db_z, 1) );

	checkCudaErrors( cudaFree(d_db_pre_x) );
	checkCudaErrors( cudaFree(d_db_pre_y) );
	checkCudaErrors( cudaFree(d_db_pre_z) );

	// 3 float to float3
	float3 * d_dbeta;
	if (dataInDevice){
		d_dbeta = h_dbeta;
	} else {
		checkCudaErrors(cudaMalloc((void **)&d_dbeta, k*sizeof(float3)));
	}

	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	Float2Float3Kernel1D <<< blocks, threads >>> ( d_db_x, d_db_y, d_db_z, d_dbeta, k);

	// add the alpha term
	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	addKernel1D <<< blocks, threads >>> (d_dbeta, d_alpha, d_dbeta, k );

	// copy the result back to host mem
	if (dataInDevice){
		// Do nothing. Duty to manage memory relies on outside code
	} else {
		checkCudaErrors(cudaMemcpy(h_dbeta, d_dbeta, k * sizeof(float3), cudaMemcpyDeviceToHost));
		checkCudaErrors( cudaFree(d_dbeta) );
	}

	checkCudaErrors( cudaFree(d_db_x) );
	checkCudaErrors( cudaFree(d_db_y) );
	checkCudaErrors( cudaFree(d_db_z) );

	// stop timer
	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);

	double alpha_beta_time = sdkGetTimerValue(&hTimer);
	//printf("Alpha_beta takes %f ms.\n", alpha_beta_time);

	// clean up
	if (dataInDevice){
		// Do nothing. Duty to manage memory relies on outside code
	} else {
		checkCudaErrors( cudaFree(d_p) );
		checkCudaErrors( cudaFree(d_q) );
		checkCudaErrors( cudaFree(d_alpha) );
		checkCudaErrors( cudaFree(d_beta) );
	}

	checkCudaErrors(cudaFree(d_pi_pj));
	checkCudaErrors(cudaFree(d_dq));
	checkCudaErrors(cudaFree(d_g));
	checkCudaErrors(cudaFree(d_dbji));
	checkCudaErrors(cudaFree(d_one));

	checkCublasErrors( cublasDestroy(handle) );

	checkCudaErrors(cudaDeviceSynchronize());


}

float PointSetHamiltonianSystem_CUDA::landmarkError_CUDA3D(float3 * h_q1, float3 * h_qT, float3 * h_alpha, int k, bool dataInDevice){
	// Variables
	float3 * d_q1;
	float3 * d_qT;
	float3 * d_alpha;
	float * d_alpha_mag;
	dim3 threads;
	dim3 blocks;

	// Some memory control stuff
	checkCudaErrors(cudaMalloc((void **)&d_alpha_mag, k*sizeof(float)));
	if (dataInDevice){
		d_q1 = h_q1;
		d_qT = h_qT;
		d_alpha = h_alpha;
	}else{
		checkCudaErrors(cudaMalloc((void **)&d_q1, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_qT, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_alpha, k*sizeof(float3)));

		checkCudaErrors(cudaMemcpy(d_q1, h_q1, k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_qT, h_qT, k * sizeof(float3), cudaMemcpyHostToDevice));
	}

	// Calculate the difference and the magnitude
	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	minusAndMagKernel1D <<< blocks, threads >>> (d_q1, d_qT, d_alpha, d_alpha_mag, k);
	float fnorm_sq = Reducer::reduce_sum_wrapper(k, d_alpha_mag);

	// Clean up
	checkCudaErrors(cudaFree(d_alpha_mag));
	if (dataInDevice){
		// Do nothing. Memory control relies on outside code
	}else{
		checkCudaErrors(cudaMemcpy(h_alpha, d_alpha, k * sizeof(float3), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_q1));
		checkCudaErrors(cudaFree(d_qT));
		checkCudaErrors(cudaFree(d_alpha));
	}

	return fnorm_sq;
}

void PointSetHamiltonianSystem_CUDA::combineGradient_CUDA3D(float3 * h_grad, float3 * h_hp, int k, float lambda, bool dataInDevice){
	// Variables
	float3 * d_grad;
	float3 * d_hp;
	dim3 threads;
	dim3 blocks;

	// Some memory control stuff
	if (dataInDevice){
		d_grad = h_grad;
		d_hp = h_hp;
	}else{
		checkCudaErrors(cudaMalloc((void **)&d_grad, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_hp, k*sizeof(float3)));

		checkCudaErrors(cudaMemcpy(d_grad, h_grad, k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_hp, h_grad, k * sizeof(float3), cudaMemcpyHostToDevice));
	}

	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	updateKernel1D <<< blocks, threads >>> (d_hp, d_grad, lambda, d_grad, k);

	// Clean up
	if (dataInDevice){
		// Do nothing. Memory control relies on outside code
	}else{
		checkCudaErrors(cudaMemcpy(h_grad, d_grad, k * sizeof(float3), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_grad));
		checkCudaErrors(cudaFree(d_hp));
	}

}

void PointSetHamiltonianSystem_CUDA::initP_CUDA3D(float3 * h_q0, float3 * h_qT, float3 * h_p0, int N, int k, bool dataInDevice){
	// Variables
	float3 * d_q0;
	float3 * d_qT;
	float3 * d_p0;
	dim3 threads;
	dim3 blocks;

	// Some memory control stuff
	if (dataInDevice){
		d_q0 = h_q0;
		d_qT = h_qT;
		d_p0 = h_p0;
	}else{
		checkCudaErrors(cudaMalloc((void **)&d_q0, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_qT, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_p0, k*sizeof(float3)));

		checkCudaErrors(cudaMemcpy(d_q0, h_q0, k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_qT, h_qT, k * sizeof(float3), cudaMemcpyHostToDevice));
	}

	// Calculate the difference and the magnitude
	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	minusAndDivideKernel1D <<< blocks, threads >>> (d_qT, d_q0, d_p0, (float) N, k );

	// Clean up
	if (dataInDevice){
		// Do nothing. Memory control relies on outside code
	}else{
		checkCudaErrors(cudaMemcpy(h_p0, d_p0, k * sizeof(float3), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_q0));
		checkCudaErrors(cudaFree(d_qT));
		checkCudaErrors(cudaFree(d_p0));
	}
}

void PointSetHamiltonianSystem_CUDA::GAlphaBeta_CUDA3D(float3 * h_q1, float3 * h_qT, float3 * h_p1,
		float3 * h_alpha, float3 * h_beta, float &Gnorm_sq, float &dsq, float lambda, int k, bool dataInDevice ){
	// Variables
	float3 * d_q1;
	float3 * d_qT;
	float3 * d_p1;
	float3 * d_alpha;
	float3 * d_beta;

	float * d_gnsq;
	float * d_dsq;

	dim3 threads;
	dim3 blocks;

	// Some memory control stuff
	checkCudaErrors(cudaMalloc((void **)&d_gnsq, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_dsq, k*sizeof(float)));
	if (dataInDevice){
		d_q1 = h_q1;
		d_qT = h_qT;
		d_p1 = h_p1;
		d_alpha = h_alpha;
		d_beta = h_beta;
	}else{
		checkCudaErrors(cudaMalloc((void **)&d_q1, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_qT, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_p1, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_alpha, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_beta, k*sizeof(float3)));

		checkCudaErrors(cudaMemcpy(d_q1, h_q1, k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_qT, h_qT, k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_p1, h_p1, k * sizeof(float3), cudaMemcpyHostToDevice));
	}

	// Compute G and alpha/beta
	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	GAlphaBetaKernel <<< blocks, threads >>> (d_q1, d_qT, d_p1, d_alpha, d_beta, d_gnsq, d_dsq, lambda, k);

	Gnorm_sq = Reducer::reduce_sum_wrapper(k, d_gnsq);
	dsq = Reducer::reduce_sum_wrapper(k, d_dsq);

	// Clean up
	checkCudaErrors(cudaFree(d_gnsq));
	checkCudaErrors(cudaFree(d_dsq));

	if (dataInDevice){
		// Do nothing. Duty of managing memory reply on outside code
	}else{
		checkCudaErrors(cudaMemcpy(h_alpha, d_alpha, k * sizeof(float3), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_beta, d_beta, k * sizeof(float3), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_q1));
		checkCudaErrors(cudaFree(d_qT));
		checkCudaErrors(cudaFree(d_p1));
		checkCudaErrors(cudaFree(d_alpha));
		checkCudaErrors(cudaFree(d_beta));
	}
}

void PointSetHamiltonianSystem_CUDA::FlowHamiltonianWithGradient_CUDA3D( std::vector<float3*> &Qt, std::vector<float3*> &Pt,
		float3 * h_q0, float3 * h_p0, float3 * h_q, float3 * h_p, float9 * h_grad_q, float9 * h_grad_p,
		int N, int k, bool dataInDevice){
}



float PointSetHamiltonianSystem_CUDA::FlowHamiltonian_CUDA3D(float3 * h_q0, float3 * h_p0,
		float3 * h_q1, float3 * h_p1,
		float3 * h_hq, float3 * h_hp,
		std::vector<float3*> &Qt, std::vector<float3*> &Pt,
		float sigma, int k, int N, bool saveIntermediate, bool dataInDevice){
	float dt = 1.0 / (float)(N-1);
	// Initialize q and p
	// The return value
	float H, H0;

	float3 * d_q_t;
	float3 * d_p_t;
	float3 * d_hq;
	float3 * d_hp;

	dim3 threads;
	dim3 blocks;

	checkCudaErrors(cudaMalloc((void **)&d_q_t, k*sizeof(float3)));
	checkCudaErrors(cudaMalloc((void **)&d_p_t, k*sizeof(float3)));

	// Some memory control stuff
	if (dataInDevice){
		d_hq = h_hq;
		d_hp = h_hp;

		checkCudaErrors(cudaMemcpy(d_q_t, h_q0, k * sizeof(float3), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_p_t, h_p0, k * sizeof(float3), cudaMemcpyDeviceToDevice));

		if (saveIntermediate){
			checkCudaErrors(cudaMemcpy(Qt[0], h_q0, k * sizeof(float3), cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMemcpy(Pt[0], h_p0, k * sizeof(float3), cudaMemcpyDeviceToDevice));
		}

	} else {
		checkCudaErrors(cudaMalloc((void **)&d_hq, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_hp, k*sizeof(float3)));

		checkCudaErrors(cudaMemcpy(d_q_t, h_q0, k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_p_t, h_p0, k * sizeof(float3), cudaMemcpyHostToDevice));

		if (saveIntermediate){
			checkCudaErrors(cudaMemcpy(Qt[0], h_q0, k * sizeof(float3), cudaMemcpyHostToHost));
			checkCudaErrors(cudaMemcpy(Pt[0], h_p0, k * sizeof(float3), cudaMemcpyHostToHost));
		}
	}

	// Flow over time
	for(int t = 1; t < N; t++){
		// Compute the hamiltonian

		H = PointSetHamiltonianSystem_CUDA::ComputeHamiltonianJet_CUDA3D(d_q_t, d_p_t,
				d_hq, d_hp, NULL, NULL, NULL,
				sigma, k, false, true);

		// Euler update
		threads = dim3(256, 1, 1);
		blocks = dim3( (k+255)/256, 1, 1);
		updateKernel1D <<< blocks, threads >>> (d_q_t, d_hp, dt, d_q_t, k);
		updateKernel1D <<< blocks, threads >>> (d_p_t, d_hq, -dt, d_p_t, k);

		// Save intermediate result if necessary
		if (saveIntermediate){
			if (dataInDevice){
				checkCudaErrors(cudaMemcpy(Qt[t], d_q_t, k * sizeof(float3), cudaMemcpyDeviceToDevice));
				checkCudaErrors(cudaMemcpy(Pt[t], d_p_t, k * sizeof(float3), cudaMemcpyDeviceToDevice));
			}else{
				checkCudaErrors(cudaMemcpy(Qt[t], d_q_t, k * sizeof(float3), cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaMemcpy(Pt[t], d_p_t, k * sizeof(float3), cudaMemcpyDeviceToHost));
			}
		}

		// store the first hamiltonian value
		if(t == 1)
			H0 = H;
	}

	// copy the final result out
	if (dataInDevice){
		checkCudaErrors(cudaMemcpy(h_q1, d_q_t, k * sizeof(float3), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(h_p1, d_p_t, k * sizeof(float3), cudaMemcpyDeviceToDevice));
	}else{
		checkCudaErrors(cudaMemcpy(h_q1, d_q_t, k * sizeof(float3), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_p1, d_p_t, k * sizeof(float3), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemcpy(h_hq, d_hq, k * sizeof(float3), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_hp, d_hp, k * sizeof(float3), cudaMemcpyDeviceToHost));
	}

	// Clean up
	if (dataInDevice){
		// Do nothing. Duty to manage mem replies on outside code
	}else{
		checkCudaErrors(cudaFree(d_hq));
		checkCudaErrors(cudaFree(d_hp));
	}

	checkCudaErrors(cudaFree(d_q_t));
	checkCudaErrors(cudaFree(d_p_t));

	return H0;
}

void PointSetHamiltonianSystem_CUDA::FlowGradientBackward_CUDA3D(
		std::vector<float3*> &Qt, std::vector<float3*> &Pt,
		const float3 * d_alpha, const float3 * d_beta, float3 * d_result,
		float sigma, int k, int N, bool dataInDevice){
	// Variables
	float3 * d_a;
	float3 * d_b;
	float3 * d_Da;
	float3 * d_Db;
	float3 * d_q;
	float3 * d_p;

	float dt = 1.0 / (float)(N-1);
	dim3 threads;
	dim3 blocks;

	// Some memory control stuff
	checkCudaErrors(cudaMalloc((void **)&d_a, k*sizeof(float3)));
	checkCudaErrors(cudaMalloc((void **)&d_b, k*sizeof(float3)));
	checkCudaErrors(cudaMalloc((void **)&d_Da, k*sizeof(float3)));
	checkCudaErrors(cudaMalloc((void **)&d_Db, k*sizeof(float3)));

	if (dataInDevice){
		checkCudaErrors(cudaMemcpy(d_a, d_alpha, k * sizeof(float3), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_b, d_beta, k * sizeof(float3), cudaMemcpyDeviceToDevice));
	}else{
		checkCudaErrors(cudaMemcpy(d_a, d_alpha, k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_b, d_beta, k * sizeof(float3), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMalloc((void **)&d_q, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_p, k*sizeof(float3)));
	}

	// Work our way backwards
	for(int t = N-1; t > 0; t--){
		// Load intermediate q and p
		if (dataInDevice){
			d_q = Qt[t - 1];
			d_p = Pt[t - 1];
		}else{
			checkCudaErrors(cudaMemcpy(d_q, Qt[t - 1], k * sizeof(float3), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_p, Pt[t - 1], k * sizeof(float3), cudaMemcpyHostToDevice));
		}

		// Apply Hamiltonian Hessian to get an update in alpha/beta
		PointSetHamiltonianSystem_CUDA::ApplyHamiltonianHessianToAlphaBeta_CUDA3D(d_q, d_p,
				d_a, d_b, d_Da, d_Db, sigma, k, true );

		// Update the vectors
		threads = dim3(256, 1, 1);
		blocks = dim3( (k+255)/256, 1, 1);
		updateKernel1D <<< blocks, threads >>> (d_a, d_Da, dt, d_a, k);
		updateKernel1D <<< blocks, threads >>> (d_b, d_Db, dt, d_b, k);
	}

	// Finally, what we are really after are the betas
	if (dataInDevice){
		checkCudaErrors(cudaMemcpy(d_result, d_b, k * sizeof(float3), cudaMemcpyDeviceToDevice));
	}else{
		checkCudaErrors(cudaMemcpy(d_result, d_b, k * sizeof(float3), cudaMemcpyDeviceToHost));
	}



	// Clean up
	if (dataInDevice){
		// Do nothing.
	}else{
		checkCudaErrors(cudaFree(d_q));
		checkCudaErrors(cudaFree(d_p));
	}

	checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaFree(d_b));
	checkCudaErrors(cudaFree(d_Da));
	checkCudaErrors(cudaFree(d_Db));
}

void PointSetHamiltonianSystem_CUDA::InterpolateVelocity_CUDA3D(int t, const float3 x, float3 &v,
		std::vector<float3*> &Qt, std::vector<float3*> &Pt,
		float sigma, int k, bool dataInDevice){
	// Variables
	float f = -0.5 / (sigma * sigma);
	dim3 threads;
	dim3 blocks;

	float3 * d_q;
	float3 * d_p;
	float * d_KqPt_x;
	float * d_KqPt_y;
	float * d_KqPt_z;

	// Some memory control stuff
	checkCudaErrors(cudaMalloc((void **)&d_KqPt_x, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_KqPt_y, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_KqPt_z, k*sizeof(float)));

	if (dataInDevice){
		d_q = Qt[t];
		d_p = Pt[t];
	}else{
		checkCudaErrors(cudaMalloc((void **)&d_q, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_p, k*sizeof(float3)));

		checkCudaErrors(cudaMemcpy(d_q, Qt[t], k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_p, Pt[t], k * sizeof(float3), cudaMemcpyHostToDevice));
	}

	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	KqPtKernel <<< blocks, threads >>> ( d_q, d_p, x, f, d_KqPt_x, d_KqPt_y, d_KqPt_z, k);

	v.x = Reducer::reduce_sum_wrapper(k, d_KqPt_x);
	v.y = Reducer::reduce_sum_wrapper(k, d_KqPt_y);
	v.z = Reducer::reduce_sum_wrapper(k, d_KqPt_z);

	// Clean up
	if (dataInDevice){
		// Do nothing
	} else {
		checkCudaErrors(cudaFree(d_q));
		checkCudaErrors(cudaFree(d_p));
	}

	checkCudaErrors(cudaFree(d_KqPt_x));
	checkCudaErrors(cudaFree(d_KqPt_y));
	checkCudaErrors(cudaFree(d_KqPt_z));
}

///////////////////////////////////////// 2D functions//////////////////////////////////////////////
float PointSetHamiltonianSystem_CUDA::ComputeHamiltonianJet_CUDA2D(float2 * h_q, float2 * h_p,
		float2 * h_hq, float2 * h_hp,
		float4 * h_hqq, float4 * h_hqp, float4 * h_hpp,
		float sigma, int k, bool flag_hessian, bool dataInDevice ){
	// Parameters
	float f = -0.5 / (sigma * sigma);
	long k2 = k*k;

	dim3 threads;
	dim3 blocks;

	// Initialize cublas
	cublasHandle_t handle;
	checkCublasErrors( cublasCreate(&handle) );
	//	cublasOperation_t trans = CUBLAS_OP_N; // AT  if transa == CUBLAS_OP_T
	cublasOperation_t trans = CUBLAS_OP_T; // AT  if transa == CUBLAS_OP_T
	float alf=1.0;
	float beta=0;

	// Some memory control stuff
	float2 * d_q;
	float2 * d_p;

	if (dataInDevice){
		d_q = h_q;
		d_p = h_p;
	} else {
		checkCudaErrors(cudaMalloc((void **)&d_q, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_p, k*sizeof(float2)));

		checkCudaErrors(cudaMemcpy(d_q, h_q, k*sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_p, h_p, k*sizeof(float2), cudaMemcpyHostToDevice));
	}


	// Start timer
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	// Initialize hamiltonian
	float H = 0.0;

	// allocate the memory
	float * d_pi_pj;
	float * d_pi_pj_g;
	float2 * d_dq;
	float * d_g;

	checkCudaErrors(cudaMalloc((void **)&d_pi_pj, k2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_pi_pj_g, k2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_dq, k2*sizeof(float2)));
	checkCudaErrors(cudaMalloc((void **)&d_g, k2*sizeof(float)));

	// Calculate the pi_pj, dq, g and (pi_pj * g)
	threads = dim3(16, 16, 1);
	blocks = dim3( (k+15)/16, (k+15)/16, 1);
	dqpipjKernel <<< blocks, threads >>> (d_q, d_dq, d_g, f, d_p, d_pi_pj, k);
	multiplyKernel2D <<< blocks, threads >>> (d_pi_pj, d_g, d_pi_pj_g, k, k);

	float * h_pi_pj_g = new float[k2];

	// Calculate H
	H = 0.5 * Reducer::reduce_sum_wrapper(k2, d_pi_pj_g);
	checkCudaErrors(cudaFree(d_pi_pj_g));

	// Calculate the 1st derivative
	//printf("Calculating 1st derivative...\n");
	float * d_pi_pj_g1_dq_x;
	float * d_pi_pj_g1_dq_y;

	float * d_p_g_x;
	float * d_p_g_y;

	checkCudaErrors(cudaMalloc((void **)&d_pi_pj_g1_dq_x, k2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_pi_pj_g1_dq_y, k2*sizeof(float)));

	checkCudaErrors(cudaMalloc((void **)&d_p_g_x, k2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_p_g_y, k2*sizeof(float)));

	// Precompute the terms that need to be added up
	threads = dim3(16, 16, 1);
	blocks = dim3( (k+15)/16, (k+15)/16, 1);
	hqhpPreComputeKernel <<< blocks, threads >>> ( d_pi_pj, d_dq, d_g, d_p, f,
			d_pi_pj_g1_dq_x, d_pi_pj_g1_dq_y,
			d_p_g_x, d_p_g_y, k);

	float * d_one;
	checkCudaErrors(cudaMalloc((void **)&d_one, k*sizeof(float)));
	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	onesKernel1D  <<< blocks, threads >>>  (d_one, k);

	// Allocate the memory
	float * d_hq_x;
	float * d_hq_y;
	float * d_hq_z;

	float * d_hp_x;
	float * d_hp_y;
	float * d_hp_z;

	checkCudaErrors(cudaMalloc((void **)&d_hq_x, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_hq_y, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_hq_z, k*sizeof(float)));

	checkCudaErrors(cudaMalloc((void **)&d_hp_x, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_hp_y, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_hp_z, k*sizeof(float)));

	// Use CUBLAS to multiply the terms by one vector to add up
	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_pi_pj_g1_dq_x, k, d_one, 1, &beta, d_hq_x, 1) );
	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_pi_pj_g1_dq_y, k, d_one, 1, &beta, d_hq_y, 1) );

	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_p_g_x, k, d_one, 1, &beta, d_hp_x, 1) );
	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_p_g_y, k, d_one, 1, &beta, d_hp_y, 1) );

	// clean up
	checkCudaErrors(cudaFree(d_pi_pj_g1_dq_x));
	checkCudaErrors(cudaFree(d_pi_pj_g1_dq_y));

	checkCudaErrors(cudaFree(d_p_g_x));
	checkCudaErrors(cudaFree(d_p_g_y));

	// TODO: copy the result back to host
	float2 * d_hq;
	float2 * d_hp;
	if (dataInDevice){
		d_hq = h_hq;
		d_hp = h_hp;
	} else {
		checkCudaErrors(cudaMalloc((void **)&d_hq, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_hp, k*sizeof(float2)));
	}

	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	Float2Float2Kernel1D  <<< blocks, threads >>>  ( d_hq_x, d_hq_y, d_hq, k);
	Float2Float2Kernel1D  <<< blocks, threads >>>  ( d_hp_x, d_hp_y, d_hp, k);

	checkCudaErrors(cudaFree(d_hq_x));
	checkCudaErrors(cudaFree(d_hq_y));

	checkCudaErrors(cudaFree(d_hp_x));
	checkCudaErrors(cudaFree(d_hp_y));

	// Some memory control stuff
	if (dataInDevice){
		// Do nothing. Duty to manage memory relies on outside code
	} else {
		checkCudaErrors(cudaMemcpy(h_hq, d_hq, k * sizeof(float2), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_hp, d_hp, k * sizeof(float2), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_hq));
		checkCudaErrors(cudaFree(d_hp));
	}


	//printf("Done 1st derivative.\n");

	// Calculate the 2nd derivatives
	if (flag_hessian){
		//printf("Calculating 2nd derivative...\n");
		//printf("Calculating hqq...\n");
		///////////////////////////////////////////////////////////////////////////////////////////////////////
		/* hqq */
		float * d_hqq_xx; float * d_hqq_xy;
		float * d_hqq_yx; float * d_hqq_yy;

		// Allocate memory
		//printf("hqq: Allocating mem...\n");
		checkCudaErrors(cudaMalloc((void **)&d_hqq_xx, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqq_xy, k2*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&d_hqq_yx, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqq_yy, k2*sizeof(float)));

		// Precompute the terms
		//printf("hqq: Precomputing...\n");
		threads = dim3(16, 16, 1);
		blocks = dim3( (k+15)/16, (k+15)/16, 1);
		hqqPreComputeKernel  <<< blocks, threads >>>  (d_pi_pj, d_g, d_dq, f,
				d_hqq_xx, d_hqq_xy,
				d_hqq_yx, d_hqq_yy, k);

		// The diagonal terms need sum - again use cublas
		float * d_hqq_diag_xx; float * d_hqq_diag_xy;
		float * d_hqq_diag_yx; float * d_hqq_diag_yy;

		checkCudaErrors(cudaMalloc((void **)&d_hqq_diag_xx, k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqq_diag_xy, k*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&d_hqq_diag_yx, k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqq_diag_yy, k*sizeof(float)));

		// cublas sum
		//printf("hqq: cublas sum...\n");
		float * d_mone;
		checkCudaErrors(cudaMalloc((void **)&d_mone, k*sizeof(float)));
		threads = dim3(256, 1, 1);
		blocks = dim3( (k+255)/256, 1, 1);
		fillKernel1D <<< blocks, threads >>> (d_mone, k, -1 );
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqq_xx, k, d_mone, 1, &beta, d_hqq_diag_xx, 1) );
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqq_xy, k, d_mone, 1, &beta, d_hqq_diag_xy, 1) );

		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqq_yx, k, d_mone, 1, &beta, d_hqq_diag_yx, 1) );
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqq_yy, k, d_mone, 1, &beta, d_hqq_diag_yy, 1) );

		checkCudaErrors(cudaFree(d_mone));

		// Copy the diagonal terms into the matrix
		//printf("hqq: copy diagonal term...\n");
		threads = dim3(256, 1, 1);
		blocks = dim3( (k+255)/256, 1, 1);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqq_xx, d_hqq_diag_xx, k);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqq_xy, d_hqq_diag_xy, k);

		copyToDiagonal  <<< blocks, threads >>>  (d_hqq_yx, d_hqq_diag_yx, k);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqq_yy, d_hqq_diag_yy, k);

		checkCudaErrors(cudaFree(d_hqq_diag_xx));
		checkCudaErrors(cudaFree(d_hqq_diag_xy));

		checkCudaErrors(cudaFree(d_hqq_diag_yx));
		checkCudaErrors(cudaFree(d_hqq_diag_yy));

		// copy the result back to host
		//printf("hqq: copy back the result...\n");
		float4 * d_hqq;
		if (dataInDevice){
			d_hqq = h_hqq;
		}else{
			checkCudaErrors(cudaMalloc((void **)&d_hqq, k2*sizeof(float4)));
		}
		//printf("hqq: Done allocate memory...\n");

		threads = dim3(16, 16, 1);
		blocks = dim3( (k+15)/16, (k+15)/16, 1);
		Float2Float4Kernel2D   <<< blocks, threads >>> (
				d_hqq_xx, d_hqq_xy,
				d_hqq_yx, d_hqq_yy, d_hqq, k);

		//printf("hqq: Done copy 9 float to float9...\n");

		checkCudaErrors(cudaFree(d_hqq_xx));
		checkCudaErrors(cudaFree(d_hqq_xy));

		checkCudaErrors(cudaFree(d_hqq_yx));
		checkCudaErrors(cudaFree(d_hqq_yy));

		if (dataInDevice){
			// Do nothing. Duty to manage memory relies on outside code
		}else{
			checkCudaErrors(cudaMemcpy(h_hqq, d_hqq, k2 * sizeof(float4), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaFree(d_hqq));
		}
		//printf("hqq: Done copy back to host...\n");

		//printf("Done hqq.\n");
		//printf("Calculating hqp...\n");
		////////////////////////////////////////////////////////////////////////////////////////////////////
		/* hqp */
		float * d_hqp_xx; float * d_hqp_xy;
		float * d_hqp_yx; float * d_hqp_yy;

		float * d_hqp_ii_xx; float * d_hqp_ii_xy;
		float * d_hqp_ii_yx; float * d_hqp_ii_yy;

		// Allocate memory
		checkCudaErrors(cudaMalloc((void **)&d_hqp_xx, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_xy, k2*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&d_hqp_yx, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_yy, k2*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&d_hqp_ii_xx, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_ii_xy, k2*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&d_hqp_ii_yx, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_ii_yy, k2*sizeof(float)));

		threads = dim3(16, 16, 1);
		blocks = dim3( (k+15)/16, (k+15)/16, 1);
		hqpPreComputeKernel <<< blocks, threads >>> (d_p, d_g, f, d_dq,
				d_hqp_xx, d_hqp_xy,
				d_hqp_yx, d_hqp_yy,
				d_hqp_ii_xx, d_hqp_ii_xy,
				d_hqp_ii_yx, d_hqp_ii_yy, k);

		// The diagonal terms need sum - again use cublas
		float * d_hqp_diag_xx;  float * d_hqp_diag_xy;
		float * d_hqp_diag_yx;  float * d_hqp_diag_yy;

		checkCudaErrors(cudaMalloc((void **)&d_hqp_diag_xx, k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_diag_xy, k*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&d_hqp_diag_yx, k*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hqp_diag_yy, k*sizeof(float)));

		// cublas sum
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqp_ii_xx, k, d_one, 1, &beta, d_hqp_diag_xx, 1) );
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqp_ii_xy, k, d_one, 1, &beta, d_hqp_diag_xy, 1) );

		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqp_ii_yx, k, d_one, 1, &beta, d_hqp_diag_yx, 1) );
		checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_hqp_ii_yy, k, d_one, 1, &beta, d_hqp_diag_yy, 1) );

		// Release
		checkCudaErrors(cudaFree(d_hqp_ii_xx));
		checkCudaErrors(cudaFree(d_hqp_ii_xy));

		checkCudaErrors(cudaFree(d_hqp_ii_yx));
		checkCudaErrors(cudaFree(d_hqp_ii_yy));

		// copy the diagonal terms into the matrix
		threads = dim3(256, 1, 1);
		blocks = dim3( (k+255)/256, 1, 1);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqp_xx, d_hqp_diag_xx, k);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqp_xy, d_hqp_diag_xy, k);

		copyToDiagonal  <<< blocks, threads >>>  (d_hqp_yx, d_hqp_diag_yx, k);
		copyToDiagonal  <<< blocks, threads >>>  (d_hqp_yy, d_hqp_diag_yy, k);

		checkCudaErrors(cudaFree(d_hqp_diag_xx));
		checkCudaErrors(cudaFree(d_hqp_diag_xy));

		checkCudaErrors(cudaFree(d_hqp_diag_yx));
		checkCudaErrors(cudaFree(d_hqp_diag_yy));

		// copy the result back to host
		float4 * d_hqp;
		if (dataInDevice){
			d_hqp = h_hqp;
		}else{
			checkCudaErrors(cudaMalloc((void **)&d_hqp, k2*sizeof(float4)));
		}

		threads = dim3(16, 16, 1);
		blocks = dim3( (k+15)/16, (k+15)/16, 1);
		Float2Float4Kernel2D   <<< blocks, threads >>> (
				d_hqp_xx, d_hqp_xy,
				d_hqp_yx, d_hqp_yy,
				d_hqp, k);

		checkCudaErrors(cudaFree(d_hqp_xx));
		checkCudaErrors(cudaFree(d_hqp_xy));

		checkCudaErrors(cudaFree(d_hqp_yx));
		checkCudaErrors(cudaFree(d_hqp_yy));

		if (dataInDevice){
			// Do nothing. Duty to manage memory relies on outside code
		}else{
			checkCudaErrors(cudaMemcpy(h_hqp, d_hqp, k2 * sizeof(float4), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaFree(d_hqp));
		}

		//printf("Done hqp.\n");
		//printf("Calculating hpp...\n");

		////////////////////////////////////////////////////////////////////////////////////////////
		/* hpp */
		float * d_hpp_xx; float * d_hpp_yy;
		checkCudaErrors(cudaMalloc((void **)&d_hpp_xx, k2*sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&d_hpp_yy, k2*sizeof(float)));
		threads = dim3(16, 16, 1);
		blocks = dim3( (k+15)/16, (k+15)/16, 1);
		hppPreComputeKernel  <<< blocks, threads >>>  (d_g, d_hpp_xx, d_hpp_yy, k);

		// copy the result back to host
		float * d_zero;
		checkCudaErrors(cudaMalloc((void **)&d_zero, k2*sizeof(float)));
		checkCudaErrors(cudaMemset(d_zero, 0, k2*sizeof(float)));

		float4 * d_hpp;
		if (dataInDevice){
			d_hpp = h_hpp;
		}else{
			checkCudaErrors(cudaMalloc((void **)&d_hpp, k2*sizeof(float4)));
		}

		Float2Float4Kernel2D   <<< blocks, threads >>> (
				d_hpp_xx, d_zero,
				d_zero, d_hpp_yy,
				d_hpp, k);

		checkCudaErrors(cudaFree(d_zero));

		if (dataInDevice){
			// Do nothing. Duty to manage memory relies on outside code
		}else{
			checkCudaErrors(cudaMemcpy(h_hpp, d_hpp, k2 * sizeof(float4), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaFree(d_hpp));
		}

		//printf("Done hpp.\n");
		//printf("Done 2nd derivative.\n");

	}

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);

	double hamiltonian_time = sdkGetTimerValue(&hTimer);
	//printf("Hamiltonian takes %f ms.\n", hamiltonian_time);

	// Clean up
	if (dataInDevice){
		// Do nothing. Duty to manage memory relies on outside code
	} else {
		checkCudaErrors( cudaFree(d_p) );
		checkCudaErrors( cudaFree(d_q) );
	}
	checkCudaErrors(cudaFree(d_pi_pj));
	checkCudaErrors(cudaFree(d_dq));
	checkCudaErrors(cudaFree(d_g));
	checkCudaErrors(cudaFree(d_one));

	checkCublasErrors( cublasDestroy(handle) );
	checkCudaErrors(cudaDeviceSynchronize());

	return H;
}

void PointSetHamiltonianSystem_CUDA::ApplyHamiltonianHessianToAlphaBeta_CUDA2D(float2 * h_q, float2 * h_p,
		float2 * h_alpha, float2 * h_beta,
		float2 * h_dalpha, float2 * h_dbeta,
		float sigma, int k, bool dataInDevice ){
	// Some variable
	float f = -0.5 / (sigma * sigma);
	long k2 = k*k;

	dim3 threads;
	dim3 blocks;

	float * d_one;
	checkCudaErrors(cudaMalloc((void **)&d_one, k*sizeof(float)));
	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	onesKernel1D  <<< blocks, threads >>>  (d_one, k);

	// Initialize cublas
	cublasHandle_t handle;
	checkCublasErrors( cublasCreate(&handle) );
	//	cublasOperation_t trans = CUBLAS_OP_N; // AT  if transa == CUBLAS_OP_T
	cublasOperation_t trans = CUBLAS_OP_T; // AT  if transa == CUBLAS_OP_T
	float alf=1.0;
	float beta=0;

	// Some memory control stuff
	float2 * d_q;
	float2 * d_p;
	float2 * d_alpha;
	float2 * d_beta;

	if (dataInDevice){
		d_q = h_q;
		d_p = h_p;
		d_alpha = h_alpha;
		d_beta = h_beta;
	} else {
		checkCudaErrors(cudaMalloc((void **)&d_q, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_p, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_alpha, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_beta, k*sizeof(float2)));

		checkCudaErrors(cudaMemcpy(d_q, h_q, k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_p, h_p, k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_alpha, h_alpha, k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_beta, h_beta, k * sizeof(float2), cudaMemcpyHostToDevice));
	}

	// Start timer
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	// allocate the memory
	float * d_pi_pj;
	float2 * d_dq;
	float * d_g;

	checkCudaErrors(cudaMalloc((void **)&d_pi_pj, k2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_dq, k2*sizeof(float2)));
	checkCudaErrors(cudaMalloc((void **)&d_g, k2*sizeof(float)));

	// Calculate the pi_pj, dq, g and (pi_pj * g)
	threads = dim3(16, 16, 1);
	blocks = dim3( (k+15)/16, (k+15)/16, 1);
	dqpipjKernel <<< blocks, threads >>> (d_q, d_dq, d_g, f, d_p, d_pi_pj, k);

	// Calculate the dj-di
	threads = dim3(16, 16, 1);
	blocks = dim3( (k+15)/16, (k+15)/16, 1);
	float2 * d_dbji;
	checkCudaErrors(cudaMalloc((void **)&d_dbji, k2*sizeof(float2)));
	dbjiKernel <<< blocks, threads >>> ( d_beta, d_dbji, k );

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/* dalpha */
	// Precompute for the da and aa terms
	float * d_da_pre_x;
	float * d_da_pre_y;

	checkCudaErrors(cudaMalloc((void **)&d_da_pre_x, k2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_da_pre_y, k2*sizeof(float)));

	threads = dim3(16, 16, 1);
	blocks = dim3( (k+15)/16, (k+15)/16, 1);
	dalphaPrecomputeKernel <<< blocks, threads >>> (d_pi_pj, d_dq, d_g, d_dbji, f, k,
			d_da_pre_x, d_da_pre_y,
			d_p, d_alpha);

	// Use cublas to sum
	float * d_da_x;
	float * d_da_y;

	checkCudaErrors(cudaMalloc((void **)&d_da_x, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_da_y, k*sizeof(float)));

	// cublas m * v
	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_da_pre_x, k, d_one, 1, &beta, d_da_x, 1) );
	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_da_pre_y, k, d_one, 1, &beta, d_da_y, 1) );

	checkCudaErrors( cudaFree(d_da_pre_x) );
	checkCudaErrors( cudaFree(d_da_pre_y) );

	// 2 float to float2
	float2 * d_dalpha;
	if (dataInDevice){
		d_dalpha = h_dalpha;
	} else {
		checkCudaErrors(cudaMalloc((void **)&d_dalpha, k*sizeof(float2)));
	}

	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	Float2Float2Kernel1D <<< blocks, threads >>> ( d_da_x, d_da_y, d_dalpha, k);

	// copy the result back to host mem
	if (dataInDevice){
		// Do nothing. Duty to manage memory relies on outside code
	} else {
		checkCudaErrors(cudaMemcpy(h_dalpha, d_dalpha, k * sizeof(float2), cudaMemcpyDeviceToHost));
		checkCudaErrors( cudaFree(d_dalpha) );
	}

	checkCudaErrors( cudaFree(d_da_x) );
	checkCudaErrors( cudaFree(d_da_y) );

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/* dbeta */

	// precompute
	float * d_db_pre_x;
	float * d_db_pre_y;

	checkCudaErrors(cudaMalloc((void **)&d_db_pre_x, k2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_db_pre_y, k2*sizeof(float)));

	threads = dim3(16, 16, 1);
	blocks = dim3( (k+15)/16, (k+15)/16, 1);
	dbetaPrecomputeKernel <<< blocks, threads >>> ( d_p, d_dq, d_g, d_dbji, f, k,
			d_db_pre_x, d_db_pre_y, d_alpha);

	// Use cublas to sum
	float * d_db_x;
	float * d_db_y;

	checkCudaErrors(cudaMalloc((void **)&d_db_x, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_db_y, k*sizeof(float)));

	// cublas m * v
	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_db_pre_x, k, d_one, 1, &beta, d_db_x, 1) );
	checkCublasErrors( cublasSgemv(handle, trans, k, k, &alf, d_db_pre_y, k, d_one, 1, &beta, d_db_y, 1) );

	checkCudaErrors( cudaFree(d_db_pre_x) );
	checkCudaErrors( cudaFree(d_db_pre_y) );

	// 3 float to float3
	float2 * d_dbeta;
	if (dataInDevice){
		d_dbeta = h_dbeta;
	} else {
		checkCudaErrors(cudaMalloc((void **)&d_dbeta, k*sizeof(float2)));
	}

	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	Float2Float2Kernel1D <<< blocks, threads >>> ( d_db_x, d_db_y, d_dbeta, k);

	// add the alpha term
	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	addKernel1D <<< blocks, threads >>> (d_dbeta, d_alpha, d_dbeta, k );

	// copy the result back to host mem
	if (dataInDevice){
		// Do nothing. Duty to manage memory relies on outside code
	} else {
		checkCudaErrors(cudaMemcpy(h_dbeta, d_dbeta, k * sizeof(float2), cudaMemcpyDeviceToHost));
		checkCudaErrors( cudaFree(d_dbeta) );
	}

	checkCudaErrors( cudaFree(d_db_x) );
	checkCudaErrors( cudaFree(d_db_y) );

	// stop timer
	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);

	double alpha_beta_time = sdkGetTimerValue(&hTimer);
	//printf("Alpha_beta takes %f ms.\n", alpha_beta_time);

	// clean up
	if (dataInDevice){
		// Do nothing. Duty to manage memory relies on outside code
	} else {
		checkCudaErrors( cudaFree(d_p) );
		checkCudaErrors( cudaFree(d_q) );
		checkCudaErrors( cudaFree(d_alpha) );
		checkCudaErrors( cudaFree(d_beta) );
	}

	checkCudaErrors(cudaFree(d_pi_pj));
	checkCudaErrors(cudaFree(d_dq));
	checkCudaErrors(cudaFree(d_g));
	checkCudaErrors(cudaFree(d_dbji));
	checkCudaErrors(cudaFree(d_one));

	checkCublasErrors( cublasDestroy(handle) );
	checkCudaErrors(cudaDeviceSynchronize());
}

float PointSetHamiltonianSystem_CUDA::landmarkError_CUDA2D(float2 * h_q1, float2 * h_qT, float2 * h_alpha, int k, bool dataInDevice){
	// Variables
	float2 * d_q1;
	float2 * d_qT;
	float2 * d_alpha;
	float * d_alpha_mag;
	dim3 threads;
	dim3 blocks;

	// Some memory control stuff
	checkCudaErrors(cudaMalloc((void **)&d_alpha_mag, k*sizeof(float)));
	if (dataInDevice){
		d_q1 = h_q1;
		d_qT = h_qT;
		d_alpha = h_alpha;
	}else{
		checkCudaErrors(cudaMalloc((void **)&d_q1, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_qT, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_alpha, k*sizeof(float2)));

		checkCudaErrors(cudaMemcpy(d_q1, h_q1, k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_qT, h_qT, k * sizeof(float2), cudaMemcpyHostToDevice));
	}

	// Calculate the difference and the magnitude
	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	minusAndMagKernel1D <<< blocks, threads >>> (d_q1, d_qT, d_alpha, d_alpha_mag, k);
	float fnorm_sq = Reducer::reduce_sum_wrapper(k, d_alpha_mag);

	checkCudaErrors(cudaFree(d_alpha_mag));

	if (dataInDevice){
		// Do nothing. Memory control relies on outside code
	}else{
		checkCudaErrors(cudaMemcpy(h_alpha, d_alpha, k * sizeof(float2), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_q1));
		checkCudaErrors(cudaFree(d_qT));
		checkCudaErrors(cudaFree(d_alpha));
	}

	return fnorm_sq;
}

void PointSetHamiltonianSystem_CUDA::combineGradient_CUDA2D(float2 * h_grad, float2 * h_hp, int k, float lambda, bool dataInDevice){
	// Variables
	float2 * d_grad;
	float2 * d_hp;
	dim3 threads;
	dim3 blocks;

	// Some memory control stuff
	if (dataInDevice){
		d_grad = h_grad;
		d_hp = h_hp;
	}else{
		checkCudaErrors(cudaMalloc((void **)&d_grad, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_hp, k*sizeof(float2)));

		checkCudaErrors(cudaMemcpy(d_grad, h_grad, k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_hp, h_grad, k * sizeof(float2), cudaMemcpyHostToDevice));
	}

	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	updateKernel1D <<< blocks, threads >>> (d_hp, d_grad, lambda, d_grad, k);

	// Clean up
	if (dataInDevice){
		// Do nothing. Memory control relies on outside code
	}else{
		checkCudaErrors(cudaMemcpy(h_grad, d_grad, k * sizeof(float2), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(d_grad));
		checkCudaErrors(cudaFree(d_hp));
	}
}

void PointSetHamiltonianSystem_CUDA::initP_CUDA2D(float2 * h_q0, float2 * h_qT, float2 * h_p0, int N, int k, bool dataInDevice){
	// Variables
	float2 * d_q0;
	float2 * d_qT;
	float2 * d_p0;
	dim3 threads;
	dim3 blocks;

	// Some memory control stuff
	if (dataInDevice){
		d_q0 = h_q0;
		d_qT = h_qT;
		d_p0 = h_p0;
	}else{
		checkCudaErrors(cudaMalloc((void **)&d_q0, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_qT, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_p0, k*sizeof(float2)));

		checkCudaErrors(cudaMemcpy(d_q0, h_q0, k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_qT, h_qT, k * sizeof(float2), cudaMemcpyHostToDevice));
	}

	// Calculate the difference and the magnitude
	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	minusAndDivideKernel1D <<< blocks, threads >>> (d_qT, d_q0, d_p0, (float) N, k );

	// Clean up
	if (dataInDevice){
		// Do nothing. Memory control relies on outside code
	}else{
		checkCudaErrors(cudaMemcpy(h_p0, d_p0, k * sizeof(float2), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_q0));
		checkCudaErrors(cudaFree(d_qT));
		checkCudaErrors(cudaFree(d_p0));
	}

}

void PointSetHamiltonianSystem_CUDA::GAlphaBeta_CUDA2D(float2 * h_q1, float2 * h_qT, float2 * h_p1,
		float2 * h_alpha, float2 * h_beta, float &Gnorm_sq, float &dsq, float lambda, int k, bool dataInDevice ){
	// Variables
	float2 * d_q1;
	float2 * d_qT;
	float2 * d_p1;
	float2 * d_alpha;
	float2 * d_beta;

	float * d_gnsq;
	float * d_dsq;

	dim3 threads;
	dim3 blocks;

	// Some memory control stuff
	checkCudaErrors(cudaMalloc((void **)&d_gnsq, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_dsq, k*sizeof(float)));
	if (dataInDevice){
		d_q1 = h_q1;
		d_qT = h_qT;
		d_p1 = h_p1;
		d_alpha = h_alpha;
		d_beta = h_beta;
	}else{
		checkCudaErrors(cudaMalloc((void **)&d_q1, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_qT, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_p1, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_alpha, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_beta, k*sizeof(float2)));

		checkCudaErrors(cudaMemcpy(d_q1, h_q1, k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_qT, h_qT, k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_p1, h_p1, k * sizeof(float2), cudaMemcpyHostToDevice));
	}

	// Compute G and alpha/beta
	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	GAlphaBetaKernel <<< blocks, threads >>> (d_q1, d_qT, d_p1, d_alpha, d_beta, d_gnsq, d_dsq, lambda, k);

	Gnorm_sq = Reducer::reduce_sum_wrapper(k, d_gnsq);
	dsq = Reducer::reduce_sum_wrapper(k, d_dsq);

	// Clean up
	checkCudaErrors(cudaFree(d_gnsq));
	checkCudaErrors(cudaFree(d_dsq));

	if (dataInDevice){
		// Do nothing. Duty of managing memory reply on outside code
	}else{
		checkCudaErrors(cudaMemcpy(h_alpha, d_alpha, k * sizeof(float2), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_beta, d_beta, k * sizeof(float2), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_q1));
		checkCudaErrors(cudaFree(d_qT));
		checkCudaErrors(cudaFree(d_p1));
		checkCudaErrors(cudaFree(d_alpha));
		checkCudaErrors(cudaFree(d_beta));
	}
}

void PointSetHamiltonianSystem_CUDA::FlowHamiltonianWithGradient_CUDA2D( std::vector<float2*> &Qt, std::vector<float2*> &Pt,
		float2 * h_q0, float2 * h_p0, float2 * h_q, float2 * h_p, float4 * h_grad_q, float4 * h_grad_p,
		int N, int k, bool dataInDevice){
}




float PointSetHamiltonianSystem_CUDA::FlowHamiltonian_CUDA2D(float2 * h_q0, float2 * h_p0,
		float2 * h_q1, float2 * h_p1,
		float2 * h_hq, float2 * h_hp,
		std::vector<float2*> &Qt, std::vector<float2*> &Pt,
		float sigma, int k, int N, bool saveIntermediate, bool dataInDevice){

	float dt = 1.0 / (float)(N-1);
	// Initialize q and p
	// The return value
	float H, H0;

	float2 * d_q_t;
	float2 * d_p_t;
	float2 * d_hq;
	float2 * d_hp;

	dim3 threads;
	dim3 blocks;

	checkCudaErrors(cudaMalloc((void **)&d_q_t, k*sizeof(float2)));
	checkCudaErrors(cudaMalloc((void **)&d_p_t, k*sizeof(float2)));
	checkCudaErrors(cudaMalloc((void **)&d_hq, k*sizeof(float2)));
	checkCudaErrors(cudaMalloc((void **)&d_hp, k*sizeof(float2)));

	// Some memory control stuff
	if (dataInDevice){
		checkCudaErrors(cudaMemcpy(d_q_t, h_q0, k * sizeof(float2), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_p_t, h_p0, k * sizeof(float2), cudaMemcpyDeviceToDevice));

		if (saveIntermediate){
			checkCudaErrors(cudaMemcpy(Qt[0], h_q0, k * sizeof(float2), cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMemcpy(Pt[0], h_p0, k * sizeof(float2), cudaMemcpyDeviceToDevice));
		}

	} else {
		checkCudaErrors(cudaMemcpy(d_q_t, h_q0, k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_p_t, h_p0, k * sizeof(float2), cudaMemcpyHostToDevice));

		if (saveIntermediate){
			checkCudaErrors(cudaMemcpy(Qt[0], h_q0, k * sizeof(float2), cudaMemcpyHostToHost));
			checkCudaErrors(cudaMemcpy(Pt[0], h_p0, k * sizeof(float2), cudaMemcpyHostToHost));
		}
	}

	// Flow over time
	for(int t = 1; t < N; t++){
		// Compute the hamiltonian
		H = PointSetHamiltonianSystem_CUDA::ComputeHamiltonianJet_CUDA2D(d_q_t, d_p_t,
				d_hq, d_hp, NULL, NULL, NULL,
				sigma, k, false, true);

		// Euler update
		threads = dim3(256, 1, 1);
		blocks = dim3( (k+255)/256, 1, 1);
		updateKernel1D <<< blocks, threads >>> (d_q_t, d_hp, dt, d_q_t, k);
		updateKernel1D <<< blocks, threads >>> (d_p_t, d_hq, -dt, d_p_t, k);

		// Save intermediate result if necessary
		if (saveIntermediate){
			if (dataInDevice){
				checkCudaErrors(cudaMemcpy(Qt[t], d_q_t, k * sizeof(float2), cudaMemcpyDeviceToDevice));
				checkCudaErrors(cudaMemcpy(Pt[t], d_p_t, k * sizeof(float2), cudaMemcpyDeviceToDevice));
			}else{
				checkCudaErrors(cudaMemcpy(Qt[t], d_q_t, k * sizeof(float2), cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaMemcpy(Pt[t], d_p_t, k * sizeof(float2), cudaMemcpyDeviceToHost));
			}
		}

		// store the first hamiltonian value
		if(t == 1)
			H0 = H;
	}

	// copy the final result out
	if (dataInDevice){
		checkCudaErrors(cudaMemcpy(h_q1, d_q_t, k * sizeof(float2), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(h_p1, d_p_t, k * sizeof(float2), cudaMemcpyDeviceToDevice));
	}else{
		checkCudaErrors(cudaMemcpy(h_q1, d_q_t, k * sizeof(float2), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_p1, d_p_t, k * sizeof(float2), cudaMemcpyDeviceToHost));
	}

	// Clean up
	checkCudaErrors(cudaFree(d_q_t));
	checkCudaErrors(cudaFree(d_p_t));
	checkCudaErrors(cudaFree(d_hq));
	checkCudaErrors(cudaFree(d_hp));

	return H0;
}

void PointSetHamiltonianSystem_CUDA::FlowGradientBackward_CUDA2D(std::vector<float2*> &Qt, std::vector<float2*> &Pt,
		const float2 * d_alpha, const float2 * d_beta, float2 * d_result,
		float sigma, int k, int N, bool dataInDevice){
	// Variables
	float2 * d_a;
	float2 * d_b;
	float2 * d_Da;
	float2 * d_Db;
	float2 * d_q;
	float2 * d_p;

	float dt = 1.0 / (float)(N-1);
	dim3 threads;
	dim3 blocks;

	// Some memory control stuff
	checkCudaErrors(cudaMalloc((void **)&d_a, k*sizeof(float2)));
	checkCudaErrors(cudaMalloc((void **)&d_b, k*sizeof(float2)));
	checkCudaErrors(cudaMalloc((void **)&d_Da, k*sizeof(float2)));
	checkCudaErrors(cudaMalloc((void **)&d_Db, k*sizeof(float2)));

	if (dataInDevice){
		checkCudaErrors(cudaMemcpy(d_a, d_alpha, k * sizeof(float2), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_b, d_beta, k * sizeof(float2), cudaMemcpyDeviceToDevice));
	}else{
		checkCudaErrors(cudaMemcpy(d_a, d_alpha, k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_b, d_beta, k * sizeof(float2), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMalloc((void **)&d_q, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_p, k*sizeof(float2)));
	}

	// Work our way backwards
	for(int t = N-1; t > 0; t--){
		// Load intermediate q and p
		if (dataInDevice){
			d_q = Qt[t - 1];
			d_p = Pt[t - 1];
		}else{
			checkCudaErrors(cudaMemcpy(d_q, Qt[t - 1], k * sizeof(float2), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_p, Pt[t - 1], k * sizeof(float2), cudaMemcpyHostToDevice));
		}

		// Apply Hamiltonian Hessian to get an update in alpha/beta
		PointSetHamiltonianSystem_CUDA::ApplyHamiltonianHessianToAlphaBeta_CUDA2D(d_q, d_p,
				d_a, d_b, d_Da, d_Db, sigma, k, true );

		// Update the vectors
		threads = dim3(256, 1, 1);
		blocks = dim3( (k+255)/256, 1, 1);
		updateKernel1D <<< blocks, threads >>> (d_a, d_Da, dt, d_a, k);
		updateKernel1D <<< blocks, threads >>> (d_b, d_Db, dt, d_b, k);
	}

	// Finally, what we are really after are the betas
	if (dataInDevice){
		checkCudaErrors(cudaMemcpy(d_result, d_b, k * sizeof(float2), cudaMemcpyDeviceToDevice));
	}else{
		checkCudaErrors(cudaMemcpy(d_result, d_b, k * sizeof(float2), cudaMemcpyDeviceToHost));
	}

	// Clean up
	if (dataInDevice){
		// Do nothing.
	}else{
		checkCudaErrors(cudaFree(d_q));
		checkCudaErrors(cudaFree(d_p));
	}

	checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaFree(d_b));
	checkCudaErrors(cudaFree(d_Da));
	checkCudaErrors(cudaFree(d_Db));
}

void PointSetHamiltonianSystem_CUDA::InterpolateVelocity_CUDA2D(unsigned int t, const float2 x, float2 &v,
		std::vector<float2*> &Qt, std::vector<float2*> &Pt,
		float sigma, int k, bool dataInDevice){
	// Variables
	float f = -0.5 / (sigma * sigma);
	dim3 threads;
	dim3 blocks;

	float2 * d_q;
	float2 * d_p;
	float * d_KqPt_x;
	float * d_KqPt_y;

	// Some memory control stuff
	checkCudaErrors(cudaMalloc((void **)&d_KqPt_x, k*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_KqPt_y, k*sizeof(float)));

	if (dataInDevice){
		d_q = Qt[t];
		d_p = Pt[t];
	}else{
		checkCudaErrors(cudaMalloc((void **)&d_q, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_p, k*sizeof(float2)));

		checkCudaErrors(cudaMemcpy(d_q, Qt[t], k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_p, Pt[t], k * sizeof(float2), cudaMemcpyHostToDevice));
	}

	threads = dim3(256, 1, 1);
	blocks = dim3( (k+255)/256, 1, 1);
	KqPtKernel <<< blocks, threads >>> ( d_q, d_p, x, f, d_KqPt_x, d_KqPt_y, k);

	v.x = Reducer::reduce_sum_wrapper(k, d_KqPt_x);
	v.y = Reducer::reduce_sum_wrapper(k, d_KqPt_y);

	// Clean up
	if (dataInDevice){
		// Do nothing
	} else {
		checkCudaErrors(cudaFree(d_q));
		checkCudaErrors(cudaFree(d_p));
	}

	checkCudaErrors(cudaFree(d_KqPt_x));
	checkCudaErrors(cudaFree(d_KqPt_y));
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Accelerated hqhp and alphabeta
void hqhpRestrictedKernel_wrapper(float3 * d_q, float3 * d_p,
		float3 * d_hq, float3 * d_hp, float * d_ham, float f, int k, int blockSize){
	dim3 dimBlock( blockSize, 1, 1 );
	dim3 dimGrid( (k+2*blockSize-1) / (2*blockSize), k, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = 7 * blockSize * sizeof(float);

	// Debug: check the cover- cover is actually normal. Then the reduce process must have bug
	switch (blockSize){
	case 1024:
		hqhpRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, f, d_hq, d_hp,  d_ham, k, blockSize);
		break;
	case 512:
		hqhpRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, f, d_hq, d_hp,  d_ham, k, blockSize);
		break;
	case 256:
		hqhpRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, f, d_hq, d_hp,  d_ham, k, blockSize);
		break;
	case 128:
		hqhpRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, f, d_hq, d_hp,  d_ham, k, blockSize);
		break;
	case 64:
		hqhpRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, f, d_hq, d_hp,  d_ham, k, blockSize);
		break;
	default:
		printf("From function hqhpRestrictedKernel_wrapper: The number of thread is not a power of 2 or it is"
				" smaller than 64 or larger than 1024. Check your code!\n");
	}
}

void hqhpRestrictedKernel_wrapper(float2 * d_q, float2 * d_p,
		float2 * d_hq, float2 * d_hp, float * d_ham, float f, int k, int blockSize){
	dim3 dimBlock( blockSize, 1, 1 );
	dim3 dimGrid( (k+2*blockSize-1) / (2*blockSize), k, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = 5 * blockSize * sizeof(float);

	// Debug: check the cover- cover is actually normal. Then the reduce process must have bug
	switch (blockSize){
	case 1024:
		hqhpRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, f, d_hq, d_hp,  d_ham, k, blockSize);
		break;
	case 512:
		hqhpRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, f, d_hq, d_hp,  d_ham, k, blockSize);
		break;
	case 256:
		hqhpRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, f, d_hq, d_hp,  d_ham, k, blockSize);
		break;
	case 128:
		hqhpRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, f, d_hq, d_hp,  d_ham, k, blockSize);
		break;
	case 64:
		hqhpRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, f, d_hq, d_hp,  d_ham, k, blockSize);
		break;
	default:
		printf("From function hqhpRestrictedKernel_wrapper: The number of thread is not a power of 2 or it is"
				" smaller than 64 or larger than 1024. Check your code!\n");
	}
}

float PointSetHamiltonianSystem_CUDA::ComputeHamiltonianJet_CUDA3D_Restricted(float3 * h_q, float3 * h_p,
		float3 * h_hq, float3 * h_hp,
		float sigma, int k, int blockSize, bool dataInDevice ){
	// Variables
	float f = -0.5 / (sigma * sigma);

	float3 * d_q;
	float3 * d_p;
	float3 * d_hq;
	float3 * d_hp;
	float * d_ham;

	// Some memory control stuff
	if (dataInDevice){
		d_q = h_q;
		d_p = h_p;
		d_hq = h_hq;
		d_hp = h_hp;
	} else {
		checkCudaErrors(cudaMalloc((void **)&d_q, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_p, k*sizeof(float3)));

		checkCudaErrors(cudaMalloc((void **)&d_hq, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_hp, k*sizeof(float3)));

		checkCudaErrors(cudaMemcpy(d_q, h_q, k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_p, h_p, k * sizeof(float3), cudaMemcpyHostToDevice));
	}
	checkCudaErrors(cudaMalloc((void **)&d_ham, k*sizeof(float)));

	// Start timer
//	StopWatchInterface *hTimer = NULL;
//	sdkCreateTimer(&hTimer);
//	sdkResetTimer(&hTimer);
//	sdkStartTimer(&hTimer);
//	sdkStopTimer(&hTimer);
//	double hamiltonian_time = sdkGetTimerValue(&hTimer);
//	//printf("Restricted Hamiltonian takes %f ms.\n", hamiltonian_time);
//	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemset(d_hq, 0, k*sizeof(float3)));
	checkCudaErrors(cudaMemset(d_hp, 0, k*sizeof(float3)));
	checkCudaErrors(cudaMemset(d_ham, 0, k*sizeof(float)));

	hqhpRestrictedKernel_wrapper(d_q, d_p, d_hq, d_hp, d_ham, f, k, blockSize);

	// Calculate hamiltonian
	float H = 0.5 * Reducer::reduce_sum_wrapper(k, d_ham);

	// Clean up
	if (dataInDevice){
		// Do nothing. Duty to manage memory relies on outside code
	} else {
		checkCudaErrors(cudaMemcpy(h_hq, d_hq, k * sizeof(float3), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_hp, d_hp, k * sizeof(float3), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_p));
		checkCudaErrors(cudaFree(d_q));
		checkCudaErrors(cudaFree(d_hq));
		checkCudaErrors(cudaFree(d_hp));
	}
	checkCudaErrors(cudaFree(d_ham));

	checkCudaErrors(cudaDeviceSynchronize());

	return H;
}

float PointSetHamiltonianSystem_CUDA::ComputeHamiltonianJet_CUDA2D_Restricted(float2 * h_q, float2 * h_p,
		float2 * h_hq, float2 * h_hp,
		float sigma, int k, int blockSize, bool dataInDevice ){
	// Variables
	float f = -0.5 / (sigma * sigma);

	float2 * d_q;
	float2 * d_p;
	float2 * d_hq;
	float2 * d_hp;
	float * d_ham;

	// Some memory control stuff
	if (dataInDevice){
		d_q = h_q;
		d_p = h_p;
		d_hq = h_hq;
		d_hp = h_hp;
	} else {
		checkCudaErrors(cudaMalloc((void **)&d_q, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_p, k*sizeof(float2)));

		checkCudaErrors(cudaMalloc((void **)&d_hq, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_hp, k*sizeof(float2)));

		checkCudaErrors(cudaMemcpy(d_q, h_q, k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_p, h_p, k * sizeof(float2), cudaMemcpyHostToDevice));
	}
	checkCudaErrors(cudaMalloc((void **)&d_ham, k*sizeof(float)));

	// Start timer
//	StopWatchInterface *hTimer = NULL;
//	sdkCreateTimer(&hTimer);
//	sdkResetTimer(&hTimer);
//	sdkStartTimer(&hTimer);
//	sdkStopTimer(&hTimer);
//	double hamiltonian_time = sdkGetTimerValue(&hTimer);
//	//printf("Restricted Hamiltonian takes %f ms.\n", hamiltonian_time);
//	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemset(d_hq, 0, k*sizeof(float2)));
	checkCudaErrors(cudaMemset(d_hp, 0, k*sizeof(float2)));
	checkCudaErrors(cudaMemset(d_ham, 0, k*sizeof(float)));

	hqhpRestrictedKernel_wrapper(d_q, d_p, d_hq, d_hp, d_ham, f, k, blockSize);

	// Calculate hamiltonian
	float H = 0.5 * Reducer::reduce_sum_wrapper(k, d_ham);

	// Clean up
	if (dataInDevice){
		// Do nothing. Duty to manage memory relies on outside code
	} else {
		checkCudaErrors(cudaMemcpy(h_hq, d_hq, k * sizeof(float2), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_hp, d_hp, k * sizeof(float2), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaFree(d_p));
		checkCudaErrors(cudaFree(d_q));
		checkCudaErrors(cudaFree(d_hq));
		checkCudaErrors(cudaFree(d_hp));
	}
	checkCudaErrors(cudaFree(d_ham));

	checkCudaErrors(cudaDeviceSynchronize());

	return H;
}


void alphaBetaRestrictedKernel_wrapper(float3 * d_q, float3 * d_p,
		float3 * d_alpha, float3 * d_beta,
		float3 * d_dalpha, float3 * d_dbeta, float f, int k, int blockSize){
	dim3 dimBlock( blockSize, 1, 1 );
	dim3 dimGrid( (k+2*blockSize-1) / (2*blockSize), k, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = 6 * blockSize * sizeof(float);

	// Debug: check the cover- cover is actually normal. Then the reduce process must have bug
	switch (blockSize){
	case 1024:
		alphaBetaRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, d_alpha, d_beta, d_dalpha, d_dbeta, f, k, blockSize);
		break;
	case 512:
		alphaBetaRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, d_alpha, d_beta, d_dalpha, d_dbeta, f, k, blockSize);
		break;
	case 256:
		alphaBetaRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, d_alpha, d_beta, d_dalpha, d_dbeta, f, k, blockSize);
		break;
	case 128:
		alphaBetaRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, d_alpha, d_beta, d_dalpha, d_dbeta, f, k, blockSize);
		break;
	case 64:
		alphaBetaRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, d_alpha, d_beta, d_dalpha, d_dbeta, f, k, blockSize);
		break;
	default:
		printf("From function hqhpRestrictedKernel_wrapper: The number of thread is not a power of 2 or it is"
				" smaller than 64 or larger than 1024. Check your code!\n");
	}
}

void alphaBetaRestrictedKernel_wrapper(float2 * d_q, float2 * d_p,
		float2 * d_alpha, float2 * d_beta,
		float2 * d_dalpha, float2 * d_dbeta, float f, int k, int blockSize){
	dim3 dimBlock( blockSize, 1, 1 );
	dim3 dimGrid( (k+2*blockSize-1) / (2*blockSize), k, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = 4 * blockSize * sizeof(float);

	// Debug: check the cover- cover is actually normal. Then the reduce process must have bug
	switch (blockSize){
	case 1024:
		alphaBetaRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, d_alpha, d_beta, d_dalpha, d_dbeta, f, k, blockSize);
		break;
	case 512:
		alphaBetaRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, d_alpha, d_beta, d_dalpha, d_dbeta, f, k, blockSize);
		break;
	case 256:
		alphaBetaRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, d_alpha, d_beta, d_dalpha, d_dbeta, f, k, blockSize);
		break;
	case 128:
		alphaBetaRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, d_alpha, d_beta, d_dalpha, d_dbeta, f, k, blockSize);
		break;
	case 64:
		alphaBetaRestrictedKernel <<< dimGrid, dimBlock, smemSize >>> ( d_q, d_p, d_alpha, d_beta, d_dalpha, d_dbeta, f, k, blockSize);
		break;
	default:
		printf("From function hqhpRestrictedKernel_wrapper: The number of thread is not a power of 2 or it is"
				" smaller than 64 or larger than 1024. Check your code!\n");
	}
}



void PointSetHamiltonianSystem_CUDA::ApplyHamiltonianHessianToAlphaBeta_CUDA3D_Restricted(float3 * h_q, float3 * h_p,
		float3 * h_alpha, float3 * h_beta,
		float3 * h_dalpha, float3 * h_dbeta,
		float sigma, int k, int blockSize, bool dataInDevice ){
	// Some variable
	float f = -0.5 / (sigma * sigma);
	long k2 = k*k;

	dim3 threads;
	dim3 blocks;

	// variables
	float3 * d_q;
	float3 * d_p;
	float3 * d_alpha;
	float3 * d_beta;
	float3 * d_dalpha;
	float3 * d_dbeta;

	// Some memory control stuff
	if (dataInDevice){
		d_q = h_q;
		d_p = h_p;
		d_alpha = h_alpha;
		d_beta = h_beta;
		d_dalpha = h_dalpha;
		d_dbeta = h_dbeta;

	} else {
		checkCudaErrors(cudaMalloc((void **)&d_q, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_p, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_alpha, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_beta, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_dalpha, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_dbeta, k*sizeof(float3)));

		checkCudaErrors(cudaMemcpy(d_q, h_q, k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_p, h_p, k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_alpha, h_alpha, k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_beta, h_beta, k * sizeof(float3), cudaMemcpyHostToDevice));
	}

	// Start timer
//	StopWatchInterface *hTimer = NULL;
//	sdkCreateTimer(&hTimer);
//	sdkResetTimer(&hTimer);
//	sdkStartTimer(&hTimer);
//	sdkStopTimer(&hTimer);
//	double alpha_beta_time = sdkGetTimerValue(&hTimer);
	//printf("Restricted alpha_beta takes %f ms.\n", alpha_beta_time);
//	// stop timer
//	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemset(d_dalpha, 0, k*sizeof(float3)));
	checkCudaErrors(cudaMemset(d_dbeta, 0, k*sizeof(float3)));

	/////////////////////////////////////////////////////////////////////////////////////////
	// run the wrapper
	alphaBetaRestrictedKernel_wrapper(d_q, d_p, d_alpha, d_beta, d_dalpha, d_dbeta, f, k, blockSize);

	// clean up
	if (dataInDevice){
		// Do nothing. Duty to manage memory relies on outside code
	} else {
		// copy the result back
		checkCudaErrors(cudaMemcpy(h_dalpha, d_dalpha, k * sizeof(float3), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_dbeta, d_dbeta, k * sizeof(float3), cudaMemcpyDeviceToHost));

		checkCudaErrors( cudaFree(d_p) );
		checkCudaErrors( cudaFree(d_q) );
		checkCudaErrors( cudaFree(d_alpha) );
		checkCudaErrors( cudaFree(d_beta) );
		checkCudaErrors( cudaFree(d_dalpha) );
		checkCudaErrors( cudaFree(d_dbeta) );
	}
	checkCudaErrors(cudaDeviceSynchronize());
}

void PointSetHamiltonianSystem_CUDA::ApplyHamiltonianHessianToAlphaBeta_CUDA2D_Restricted(float2 * h_q, float2 * h_p,
		float2 * h_alpha, float2 * h_beta,
		float2 * h_dalpha, float2 * h_dbeta,
		float sigma, int k, int blockSize, bool dataInDevice ){
	// Some variable
	float f = -0.5 / (sigma * sigma);
	long k2 = k*k;

	dim3 threads;
	dim3 blocks;

	// variables
	float2 * d_q;
	float2 * d_p;
	float2 * d_alpha;
	float2 * d_beta;
	float2 * d_dalpha;
	float2 * d_dbeta;

	// Some memory control stuff
	if (dataInDevice){
		d_q = h_q;
		d_p = h_p;
		d_alpha = h_alpha;
		d_beta = h_beta;
		d_dalpha = h_dalpha;
		d_dbeta = h_dbeta;

	} else {
		checkCudaErrors(cudaMalloc((void **)&d_q, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_p, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_alpha, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_beta, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_dalpha, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_dbeta, k*sizeof(float2)));

		checkCudaErrors(cudaMemcpy(d_q, h_q, k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_p, h_p, k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_alpha, h_alpha, k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_beta, h_beta, k * sizeof(float2), cudaMemcpyHostToDevice));
	}

	// Start timer
//	StopWatchInterface *hTimer = NULL;
//	sdkCreateTimer(&hTimer);
//	sdkResetTimer(&hTimer);
//	sdkStartTimer(&hTimer);
//	sdkStopTimer(&hTimer);
//	double alpha_beta_time = sdkGetTimerValue(&hTimer);
	//printf("Restricted alpha_beta takes %f ms.\n", alpha_beta_time);
//	// stop timer
//	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemset(d_dalpha, 0, k*sizeof(float2)));
	checkCudaErrors(cudaMemset(d_dbeta, 0, k*sizeof(float2)));

	/////////////////////////////////////////////////////////////////////////////////////////
	// run the wrapper
	alphaBetaRestrictedKernel_wrapper(d_q, d_p, d_alpha, d_beta, d_dalpha, d_dbeta, f, k, blockSize);

	// clean up
	if (dataInDevice){
		// Do nothing. Duty to manage memory relies on outside code
	} else {
		// copy the result back
		checkCudaErrors(cudaMemcpy(h_dalpha, d_dalpha, k * sizeof(float2), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_dbeta, d_dbeta, k * sizeof(float2), cudaMemcpyDeviceToHost));

		checkCudaErrors( cudaFree(d_p) );
		checkCudaErrors( cudaFree(d_q) );
		checkCudaErrors( cudaFree(d_alpha) );
		checkCudaErrors( cudaFree(d_beta) );
		checkCudaErrors( cudaFree(d_dalpha) );
		checkCudaErrors( cudaFree(d_dbeta) );
	}
	checkCudaErrors(cudaDeviceSynchronize());
}


float PointSetHamiltonianSystem_CUDA::FlowHamiltonian_CUDA3D_Restricted(float3 * h_q0, float3 * h_p0,
		float3 * h_q1, float3 * h_p1,
		float3 * h_hq, float3 * h_hp,
		std::vector<float3*> &Qt, std::vector<float3*> &Pt,
		float sigma, int k, int N, int blockSize, bool saveIntermediate, bool dataInDevice){
	float dt = 1.0 / (float)(N-1);
	// Initialize q and p
	// The return value
	float H, H0;

	float3 * d_q_t;
	float3 * d_p_t;
	float3 * d_hq;
	float3 * d_hp;

	dim3 threads;
	dim3 blocks;

	checkCudaErrors(cudaMalloc((void **)&d_q_t, k*sizeof(float3)));
	checkCudaErrors(cudaMalloc((void **)&d_p_t, k*sizeof(float3)));

	// Some memory control stuff
	if (dataInDevice){
		d_hq = h_hq;
		d_hp = h_hp;

		checkCudaErrors(cudaMemcpy(d_q_t, h_q0, k * sizeof(float3), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_p_t, h_p0, k * sizeof(float3), cudaMemcpyDeviceToDevice));

		if (saveIntermediate){
			checkCudaErrors(cudaMemcpy(Qt[0], h_q0, k * sizeof(float3), cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMemcpy(Pt[0], h_p0, k * sizeof(float3), cudaMemcpyDeviceToDevice));
		}

	} else {
		checkCudaErrors(cudaMalloc((void **)&d_hq, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_hp, k*sizeof(float3)));

		checkCudaErrors(cudaMemcpy(d_q_t, h_q0, k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_p_t, h_p0, k * sizeof(float3), cudaMemcpyHostToDevice));

		if (saveIntermediate){
			checkCudaErrors(cudaMemcpy(Qt[0], h_q0, k * sizeof(float3), cudaMemcpyHostToHost));
			checkCudaErrors(cudaMemcpy(Pt[0], h_p0, k * sizeof(float3), cudaMemcpyHostToHost));
		}
	}

	// Flow over time
	for(int t = 1; t < N; t++){
		// Compute the hamiltonian

		H = PointSetHamiltonianSystem_CUDA::ComputeHamiltonianJet_CUDA3D_Restricted(d_q_t, d_p_t,
				d_hq, d_hp, sigma, k, blockSize, true);

		// Euler update
		threads = dim3(256, 1, 1);
		blocks = dim3( (k+255)/256, 1, 1);
		updateKernel1D <<< blocks, threads >>> (d_q_t, d_hp, dt, d_q_t, k);
		updateKernel1D <<< blocks, threads >>> (d_p_t, d_hq, -dt, d_p_t, k);

		// Save intermediate result if necessary
		if (saveIntermediate){
			if (dataInDevice){
				checkCudaErrors(cudaMemcpy(Qt[t], d_q_t, k * sizeof(float3), cudaMemcpyDeviceToDevice));
				checkCudaErrors(cudaMemcpy(Pt[t], d_p_t, k * sizeof(float3), cudaMemcpyDeviceToDevice));
			}else{
				checkCudaErrors(cudaMemcpy(Qt[t], d_q_t, k * sizeof(float3), cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaMemcpy(Pt[t], d_p_t, k * sizeof(float3), cudaMemcpyDeviceToHost));
			}
		}

		// store the first hamiltonian value
		if(t == 1)
			H0 = H;
	}

	// copy the final result out
	if (dataInDevice){
		checkCudaErrors(cudaMemcpy(h_q1, d_q_t, k * sizeof(float3), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(h_p1, d_p_t, k * sizeof(float3), cudaMemcpyDeviceToDevice));
	}else{
		checkCudaErrors(cudaMemcpy(h_q1, d_q_t, k * sizeof(float3), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_p1, d_p_t, k * sizeof(float3), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemcpy(h_hq, d_hq, k * sizeof(float3), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_hp, d_hp, k * sizeof(float3), cudaMemcpyDeviceToHost));
	}

	// Clean up
	if (dataInDevice){
		// Do nothing. Duty to manage mem replies on outside code
	}else{
		checkCudaErrors(cudaFree(d_hq));
		checkCudaErrors(cudaFree(d_hp));
	}

	checkCudaErrors(cudaFree(d_q_t));
	checkCudaErrors(cudaFree(d_p_t));

	checkCudaErrors(cudaDeviceSynchronize());

	return H0;
}

float PointSetHamiltonianSystem_CUDA::FlowHamiltonian_CUDA2D_Restricted(float2 * h_q0, float2 * h_p0,
		float2 * h_q1, float2 * h_p1,
		float2 * h_hq, float2 * h_hp,
		std::vector<float2*> &Qt, std::vector<float2*> &Pt,
		float sigma, int k, int N, int blockSize, bool saveIntermediate, bool dataInDevice){
	float dt = 1.0 / (float)(N-1);
	// Initialize q and p
	// The return value
	float H, H0;

	float2 * d_q_t;
	float2 * d_p_t;
	float2 * d_hq;
	float2 * d_hp;

	dim3 threads;
	dim3 blocks;

	checkCudaErrors(cudaMalloc((void **)&d_q_t, k*sizeof(float2)));
	checkCudaErrors(cudaMalloc((void **)&d_p_t, k*sizeof(float2)));

	// Some memory control stuff
	if (dataInDevice){
		d_hq = h_hq;
		d_hp = h_hp;

		checkCudaErrors(cudaMemcpy(d_q_t, h_q0, k * sizeof(float2), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_p_t, h_p0, k * sizeof(float2), cudaMemcpyDeviceToDevice));

		if (saveIntermediate){
			checkCudaErrors(cudaMemcpy(Qt[0], h_q0, k * sizeof(float2), cudaMemcpyDeviceToDevice));
			checkCudaErrors(cudaMemcpy(Pt[0], h_p0, k * sizeof(float2), cudaMemcpyDeviceToDevice));
		}

	} else {
		checkCudaErrors(cudaMalloc((void **)&d_hq, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_hp, k*sizeof(float2)));

		checkCudaErrors(cudaMemcpy(d_q_t, h_q0, k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_p_t, h_p0, k * sizeof(float2), cudaMemcpyHostToDevice));

		if (saveIntermediate){
			checkCudaErrors(cudaMemcpy(Qt[0], h_q0, k * sizeof(float2), cudaMemcpyHostToHost));
			checkCudaErrors(cudaMemcpy(Pt[0], h_p0, k * sizeof(float2), cudaMemcpyHostToHost));
		}
	}

	// Flow over time
	for(int t = 1; t < N; t++){
		// Compute the hamiltonian

		H = PointSetHamiltonianSystem_CUDA::ComputeHamiltonianJet_CUDA2D_Restricted(d_q_t, d_p_t,
				d_hq, d_hp, sigma, k, blockSize, true);

		// Euler update
		threads = dim3(256, 1, 1);
		blocks = dim3( (k+255)/256, 1, 1);
		updateKernel1D <<< blocks, threads >>> (d_q_t, d_hp, dt, d_q_t, k);
		updateKernel1D <<< blocks, threads >>> (d_p_t, d_hq, -dt, d_p_t, k);

		// Save intermediate result if necessary
		if (saveIntermediate){
			if (dataInDevice){
				checkCudaErrors(cudaMemcpy(Qt[t], d_q_t, k * sizeof(float2), cudaMemcpyDeviceToDevice));
				checkCudaErrors(cudaMemcpy(Pt[t], d_p_t, k * sizeof(float2), cudaMemcpyDeviceToDevice));
			}else{
				checkCudaErrors(cudaMemcpy(Qt[t], d_q_t, k * sizeof(float2), cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaMemcpy(Pt[t], d_p_t, k * sizeof(float2), cudaMemcpyDeviceToHost));
			}
		}

		// store the first hamiltonian value
		if(t == 1)
			H0 = H;
	}

	// copy the final result out
	if (dataInDevice){
		checkCudaErrors(cudaMemcpy(h_q1, d_q_t, k * sizeof(float2), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(h_p1, d_p_t, k * sizeof(float2), cudaMemcpyDeviceToDevice));
	}else{
		checkCudaErrors(cudaMemcpy(h_q1, d_q_t, k * sizeof(float2), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_p1, d_p_t, k * sizeof(float2), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemcpy(h_hq, d_hq, k * sizeof(float2), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_hp, d_hp, k * sizeof(float2), cudaMemcpyDeviceToHost));
	}

	// Clean up
	if (dataInDevice){
		// Do nothing. Duty to manage mem replies on outside code
	}else{
		checkCudaErrors(cudaFree(d_hq));
		checkCudaErrors(cudaFree(d_hp));
	}

	checkCudaErrors(cudaFree(d_q_t));
	checkCudaErrors(cudaFree(d_p_t));

	checkCudaErrors(cudaDeviceSynchronize());

	return H0;
}

void PointSetHamiltonianSystem_CUDA::FlowGradientBackward_CUDA3D_Restricted(
		std::vector<float3*> &Qt, std::vector<float3*> &Pt,
		const float3 * d_alpha, const float3 * d_beta, float3 * d_result,
		float sigma, int k, int N, int blockSize, bool dataInDevice){
	// Variables
	float3 * d_a;
	float3 * d_b;
	float3 * d_Da;
	float3 * d_Db;
	float3 * d_q;
	float3 * d_p;

	float dt = 1.0 / (float)(N-1);
	dim3 threads;
	dim3 blocks;

	// Some memory control stuff
	checkCudaErrors(cudaMalloc((void **)&d_a, k*sizeof(float3)));
	checkCudaErrors(cudaMalloc((void **)&d_b, k*sizeof(float3)));
	checkCudaErrors(cudaMalloc((void **)&d_Da, k*sizeof(float3)));
	checkCudaErrors(cudaMalloc((void **)&d_Db, k*sizeof(float3)));

	if (dataInDevice){
		checkCudaErrors(cudaMemcpy(d_a, d_alpha, k * sizeof(float3), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_b, d_beta, k * sizeof(float3), cudaMemcpyDeviceToDevice));
	}else{
		checkCudaErrors(cudaMemcpy(d_a, d_alpha, k * sizeof(float3), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_b, d_beta, k * sizeof(float3), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMalloc((void **)&d_q, k*sizeof(float3)));
		checkCudaErrors(cudaMalloc((void **)&d_p, k*sizeof(float3)));
	}

	// Work our way backwards
	for(int t = N-1; t > 0; t--){
		// Load intermediate q and p
		if (dataInDevice){
			d_q = Qt[t - 1];
			d_p = Pt[t - 1];
		}else{
			checkCudaErrors(cudaMemcpy(d_q, Qt[t - 1], k * sizeof(float3), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_p, Pt[t - 1], k * sizeof(float3), cudaMemcpyHostToDevice));
		}

		// Apply Hamiltonian Hessian to get an update in alpha/beta
		PointSetHamiltonianSystem_CUDA::ApplyHamiltonianHessianToAlphaBeta_CUDA3D_Restricted(d_q, d_p,
				d_a, d_b, d_Da, d_Db, sigma, k, blockSize, true );

		// Update the vectors
		threads = dim3(256, 1, 1);
		blocks = dim3( (k+255)/256, 1, 1);
		updateKernel1D <<< blocks, threads >>> (d_a, d_Da, dt, d_a, k);
		updateKernel1D <<< blocks, threads >>> (d_b, d_Db, dt, d_b, k);
	}

	// Finally, what we are really after are the betas
	if (dataInDevice){
		checkCudaErrors(cudaMemcpy(d_result, d_b, k * sizeof(float3), cudaMemcpyDeviceToDevice));
	}else{
		checkCudaErrors(cudaMemcpy(d_result, d_b, k * sizeof(float3), cudaMemcpyDeviceToHost));
	}

	// Clean up
	if (dataInDevice){
		// Do nothing.
	}else{
		checkCudaErrors(cudaFree(d_q));
		checkCudaErrors(cudaFree(d_p));
	}

	checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaFree(d_b));
	checkCudaErrors(cudaFree(d_Da));
	checkCudaErrors(cudaFree(d_Db));

	checkCudaErrors(cudaDeviceSynchronize());
}

void PointSetHamiltonianSystem_CUDA::FlowGradientBackward_CUDA2D_Restricted(
		std::vector<float2*> &Qt, std::vector<float2*> &Pt,
		const float2 * d_alpha, const float2 * d_beta, float2 * d_result,
		float sigma, int k, int N, int blockSize, bool dataInDevice){
	// Variables
	float2 * d_a;
	float2 * d_b;
	float2 * d_Da;
	float2 * d_Db;
	float2 * d_q;
	float2 * d_p;

	float dt = 1.0 / (float)(N-1);
	dim3 threads;
	dim3 blocks;

	// Some memory control stuff
	checkCudaErrors(cudaMalloc((void **)&d_a, k*sizeof(float2)));
	checkCudaErrors(cudaMalloc((void **)&d_b, k*sizeof(float2)));
	checkCudaErrors(cudaMalloc((void **)&d_Da, k*sizeof(float2)));
	checkCudaErrors(cudaMalloc((void **)&d_Db, k*sizeof(float2)));

	if (dataInDevice){
		checkCudaErrors(cudaMemcpy(d_a, d_alpha, k * sizeof(float2), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_b, d_beta, k * sizeof(float2), cudaMemcpyDeviceToDevice));
	}else{
		checkCudaErrors(cudaMemcpy(d_a, d_alpha, k * sizeof(float2), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_b, d_beta, k * sizeof(float2), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMalloc((void **)&d_q, k*sizeof(float2)));
		checkCudaErrors(cudaMalloc((void **)&d_p, k*sizeof(float2)));
	}

	// Work our way backwards
	for(int t = N-1; t > 0; t--){
		// Load intermediate q and p
		if (dataInDevice){
			d_q = Qt[t - 1];
			d_p = Pt[t - 1];
		}else{
			checkCudaErrors(cudaMemcpy(d_q, Qt[t - 1], k * sizeof(float2), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_p, Pt[t - 1], k * sizeof(float2), cudaMemcpyHostToDevice));
		}

		// Apply Hamiltonian Hessian to get an update in alpha/beta
		PointSetHamiltonianSystem_CUDA::ApplyHamiltonianHessianToAlphaBeta_CUDA2D_Restricted(d_q, d_p,
				d_a, d_b, d_Da, d_Db, sigma, k, blockSize, true );

		// Update the vectors
		threads = dim3(256, 1, 1);
		blocks = dim3( (k+255)/256, 1, 1);
		updateKernel1D <<< blocks, threads >>> (d_a, d_Da, dt, d_a, k);
		updateKernel1D <<< blocks, threads >>> (d_b, d_Db, dt, d_b, k);
	}

	// Finally, what we are really after are the betas
	if (dataInDevice){
		checkCudaErrors(cudaMemcpy(d_result, d_b, k * sizeof(float2), cudaMemcpyDeviceToDevice));
	}else{
		checkCudaErrors(cudaMemcpy(d_result, d_b, k * sizeof(float2), cudaMemcpyDeviceToHost));
	}

	// Clean up
	if (dataInDevice){
		// Do nothing.
	}else{
		checkCudaErrors(cudaFree(d_q));
		checkCudaErrors(cudaFree(d_p));
	}

	checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaFree(d_b));
	checkCudaErrors(cudaFree(d_Da));
	checkCudaErrors(cudaFree(d_Db));

	checkCudaErrors(cudaDeviceSynchronize());
}

