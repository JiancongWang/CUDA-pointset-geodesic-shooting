#ifndef HAMILTONIAN_KERNEL_CU_
#define HAMILTONIAN_KERNEL_CU_

#include "../include/hamiltonian.h"

__global__ void onesKernel1D(float * input, int col ){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= col)
		return;
	input[c] = 1.0;
}

__global__ void onesKernel2D(float * input, int col, int row ){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if ((c >= col) || (r >=row))
		return;
	int idx = c + r * col;
	input[idx] = (float)c/15.0;
}

__global__ void fillKernel1D(float * input, int col, float filler ){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= col)
		return;
	input[c] = filler;
}

__global__ void addKernel1D(float * input1, float * input2, float * output, int col ){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c>=col)
		return;
	output[c] = input1[c] + input2[c];
}

__global__ void addKernel1D(float3 * input1, float3 * input2, float3 * output, int col ){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c>=col)
		return;
	float3 i1 = input1[c];
	float3 i2 = input2[c];

	output[c].x = i1.x + i2.x;
	output[c].y = i1.y + i2.y;
	output[c].z = i1.z + i2.z;
}

__global__ void addKernel1D(float2 * input1, float2 * input2, float2 * output, int col ){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c>=col)
		return;
	float2 i1 = input1[c];
	float2 i2 = input2[c];

	output[c].x = i1.x + i2.x;
	output[c].y = i1.y + i2.y;
}

__global__ void addKernel2D(float * input1, float * input2, float * output, int col, int row ){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;

	if ((c >= col) || (r >= row))
		return;
	output[c + r * col ] = input1[c + r * col ] + input2[c + r * col ];
}


__global__ void minusAndMagKernel1D(float2 * input1, float2 * input2, float2 * output, float * mag, int col ){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c>=col)
		return;

	float2 in1 = input1[c];
	float2 in2 = input2[c];
	float2 diff;
	diff.x = in1.x - in2.x;
	diff.y = in1.y - in2.y;

	output[c] = diff;
	mag[c] = diff.x * diff.x + diff.y * diff.y;
}

__global__ void minusAndMagKernel1D(float3 * input1, float3 * input2, float3 * output, float * mag, int col ){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c>=col)
		return;

	float3 in1 = input1[c];
	float3 in2 = input2[c];
	float3 diff;
	diff.x = in1.x - in2.x;
	diff.y = in1.y - in2.y;
	diff.z = in1.z - in2.z;

	output[c] = diff;
	mag[c] = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

__global__ void minusAndDivideKernel1D(float2 * input1, float2 * input2, float2 * output, float N, int col ){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c>=col)
		return;

	float2 in1 = input1[c];
	float2 in2 = input2[c];
	float2 diff;
	diff.x = (in1.x - in2.x)/N;
	diff.y = (in1.y - in2.y)/N;

	output[c] = diff;
}

__global__ void minusAndDivideKernel1D(float3 * input1, float3 * input2, float3 * output, float N, int col ){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c>=col)
		return;

	float3 in1 = input1[c];
	float3 in2 = input2[c];
	float3 diff;
	diff.x = (in1.x - in2.x)/N;
	diff.y = (in1.y - in2.y)/N;
	diff.z = (in1.z - in2.z)/N;

	output[c] = diff;
}

__global__ void magnitiudeKernel1D (float3 * input, float * magnitude, int col){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c>=col)
		return;

	float3 v = input[c];
	magnitude[c] = v.x * v.x + v.y * v.y + v.z * v.z;
}

__global__ void magnitiudeKernel2D (float3 * input, float * magnitude, int col, int row){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if ((c >= col) || (r >= row))
		return;
	int idx = c + r * col;

	float3 v = input[idx];
	magnitude[idx] = v.x * v.x + v.y * v.y + v.z * v.z;
}


__global__ void initializeHessianDiagonal (float9 * input, float init, int col, int row){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if ((c >= col) || (r >= row))
		return;

	int idx = c + r * col;

	input[idx].e00 = init;
	input[idx].e11 = init;
	input[idx].e22 = init;
}

__global__ void gaussianAndDerivative2D(float3 * dq, float f, float * g, float * g1, float * g2, int col, int row){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if ((c >= col) || (r >= row))
		return;

	int idx = c + r * col;

	float3 ddq = dq[idx];
	float ddq_mag = ddq.x * ddq.x + ddq.y * ddq.y + ddq.z * ddq.z;

	float gg = exp(f*ddq_mag);
	float gg1 = gg * f;
	float gg2 = gg1 * f;

	g[idx] = gg;
	g1[idx] = gg1;
	g2[idx] = gg2;
}

__global__ void multiplyKernel2D(float * input1, float * input2, float * output, int col, int row ){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;

	if ((c >= col) || (r >= row))
		return;
	output[c + r * col ] = input1[c + r * col ] * input2[c + r * col ];
}


__global__ void dqpipjKernel (float2 * q, float2 * dq, float * g, float f, float2 * p, float * pi_pj, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;

	if ((c >= k) || (r >= k))
		return;

	// Calculate dq and g
	float2 qr = q[r];
	float2 qc = q[c];
	float2 ddq;
	ddq.x = qr.x - qc.x;
	ddq.y = qr.y - qc.y;

	dq[ c + r * k ] = ddq;
	float gg = exp ( (ddq.x * ddq.x + ddq.y * ddq.y) * f);
	g[ c + r * k ] = gg;

	// Calculate pi_pj and g*pi_pj
	float2 pr = p[r];
	float2 pc = p[c];

	float pipj = (pr.x * pc.x + pr.y * pc.y);
	pi_pj[ c + r * k ] = pipj;

}


__global__ void dqpipjKernel (float3 * q, float3 * dq, float * g, float f, float3 * p, float * pi_pj, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;

	if ((c >= k) || (r >= k))
		return;

	// Calculate dq and g
	float3 qr = q[r];
	float3 qc = q[c];
	float3 ddq;
	ddq.x = qr.x - qc.x;
	ddq.y = qr.y - qc.y;
	ddq.z = qr.z - qc.z;

	dq[ c + r * k ] = ddq;
	float gg = exp ( (ddq.x * ddq.x + ddq.y * ddq.y + ddq.z * ddq.z) * f);
	g[ c + r * k ] = gg;

	// Calculate pi_pj and g*pi_pj
	float3 pr = p[r];
	float3 pc = p[c];

	float pipj = (pr.x * pc.x + pr.y * pc.y + pr.z * pc.z);
	pi_pj[ c + r * k ] = pipj;
}


__global__ void hqhpPreComputeKernel ( float * pi_pj, float2 * dq, float * g, float2 * p, float f,
		float * pi_pj_g1_dq_x, float * pi_pj_g1_dq_y,
		float * p_g_x, float * p_g_y,
		int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;

	if ((c >= k) || (r >= k))
		return;

	int idx = c + r * k;
	float pipj = pi_pj[idx];
	float2 ddq = dq[idx];
	float gg = g[idx];
	float gg1 = gg * f;
	float2 pj = p[c];

	float factor = 2 * pipj * gg1;

	pi_pj_g1_dq_x[idx] = factor * ddq.x;
	pi_pj_g1_dq_y[idx] = factor * ddq.y;

	p_g_x[idx] = gg * pj.x;
	p_g_y[idx] = gg * pj.y;

}

__global__ void hqhpPreComputeKernel ( float * pi_pj, float3 * dq, float * g, float3 * p, float f,
		float * pi_pj_g1_dq_x, float * pi_pj_g1_dq_y, float * pi_pj_g1_dq_z,
		float * p_g_x, float * p_g_y, float * p_g_z , int k){

	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;

	if ((c >= k) || (r >= k))
		return;

	int idx = c + r * k;
	float pipj = pi_pj[idx];
	float3 ddq = dq[idx];
	float gg = g[idx];
	float gg1 = gg * f;
	float3 pj = p[c];

	float factor = 2 * pipj * gg1;

	pi_pj_g1_dq_x[idx] = factor * ddq.x;
	pi_pj_g1_dq_y[idx] = factor * ddq.y;
	pi_pj_g1_dq_z[idx] = factor * ddq.z;

	p_g_x[idx] = gg * pj.x;
	p_g_y[idx] = gg * pj.y;
	p_g_z[idx] = gg * pj.z;

}

__global__ void hqqPreComputeKernel(float * pi_pj, float * g, float2 * dq, float f,
		float * hqq_xx, float * hqq_xy,
		float * hqq_yx, float * hqq_yy, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if ((c >= k) || (r >= k))
		return;

	int idx = c + r * k;
	float pipj = pi_pj[idx];
	float gg = g[idx];
	float gg1 = gg * f;
	float gg2 = gg1 * f;
	float2 ddq = dq[idx];

	float xx = -2.0 * pipj * (2.0 * gg2 * ddq.x * ddq.x + gg1);
	float yy = -2.0 * pipj * (2.0 * gg2 * ddq.y * ddq.y + gg1);

	float xy, yx;
	xy = yx = -2.0 * pipj * (2.0 * gg2 * ddq.x * ddq.y);

	// Must fill 0 here to make the sum correct
	hqq_xx[idx] = ( c==r? 0:xx );
	hqq_xy[idx] = ( c==r? 0:xy );

	hqq_yx[idx] = ( c==r? 0:yx );
	hqq_yy[idx] = ( c==r? 0:yy );
}

__global__ void hqqPreComputeKernel(float * pi_pj, float * g, float3 * dq, float f,
		float * hqq_xx, float * hqq_xy, float * hqq_xz,
		float * hqq_yx, float * hqq_yy, float * hqq_yz,
		float * hqq_zx, float * hqq_zy, float * hqq_zz, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if ((c >= k) || (r >= k))
		return;

	int idx = c + r * k;
	float pipj = pi_pj[idx];
	float gg = g[idx];
	float gg1 = gg * f;
	float gg2 = gg1 * f;
	float3 ddq = dq[idx];

	float xx = -2.0 * pipj * (2.0 * gg2 * ddq.x * ddq.x + gg1);
	float yy = -2.0 * pipj * (2.0 * gg2 * ddq.y * ddq.y + gg1);
	float zz = -2.0 * pipj * (2.0 * gg2 * ddq.z * ddq.z + gg1);

	float xy, yx;
	float xz, zx;
	float yz, zy;

	xy = yx = -2.0 * pipj * (2.0 * gg2 * ddq.x * ddq.y);
	xz = zx = -2.0 * pipj * (2.0 * gg2 * ddq.x * ddq.z);
	yz = zy = -2.0 * pipj * (2.0 * gg2 * ddq.y * ddq.z);

	// Must fill 0 here to make the sum correct
	hqq_xx[idx] = ( c==r? 0:xx );
	hqq_xy[idx] = ( c==r? 0:xy );
	hqq_xz[idx] = ( c==r? 0:xz );

	hqq_yx[idx] = ( c==r? 0:yx );
	hqq_yy[idx] = ( c==r? 0:yy );
	hqq_yz[idx] = ( c==r? 0:yz );

	hqq_zx[idx] = ( c==r? 0:zx );
	hqq_zy[idx] = ( c==r? 0:zy );
	hqq_zz[idx] = ( c==r? 0:zz );

}

__global__ void hqpPreComputeKernel(float2 * p, float * g, float f, float2 * dq,
		float * d_hqp_xx, float * d_hqp_xy,
		float * d_hqp_yx, float * d_hqp_yy,
		float * d_hqp_ii_xx, float * d_hqp_ii_xy,
		float * d_hqp_ii_yx, float * d_hqp_ii_yy, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if ((c >= k) || (r >= k))
		return;

	int idx = c + r * k;

	float2 pi = p[r];
	float2 pj = p[c];
	float gg = g[idx];
	float gg1 = gg * f;
	float2 ddq = dq[idx];

	float xx = 2.0 * gg1 * ddq.x * pi.x;
	float xy = 2.0 * gg1 * ddq.x * pi.y;

	float yx = 2.0 * gg1 * ddq.y * pi.x;
	float yy = 2.0 * gg1 * ddq.y * pi.y;

	d_hqp_xx[idx] = xx;
	d_hqp_xy[idx] = xy;

	d_hqp_yx[idx] = yx;
	d_hqp_yy[idx] = yy;

	xx = 2.0 * gg1 * ddq.x * pj.x;
	xy = 2.0 * gg1 * ddq.x * pj.y;

	yx = 2.0 * gg1 * ddq.y * pj.x;
	yy = 2.0 * gg1 * ddq.y * pj.y;

	d_hqp_ii_xx[idx] = xx;
	d_hqp_ii_xy[idx] = xy;

	d_hqp_ii_yx[idx] = yx;
	d_hqp_ii_yy[idx] = yy;
}

__global__ void hqpPreComputeKernel(float3 * p, float * g, float f, float3 * dq,
		float * d_hqp_xx, float * d_hqp_xy, float * d_hqp_xz,
		float * d_hqp_yx, float * d_hqp_yy, float * d_hqp_yz,
		float * d_hqp_zx, float * d_hqp_zy, float * d_hqp_zz,
		float * d_hqp_ii_xx, float * d_hqp_ii_xy, float * d_hqp_ii_xz,
		float * d_hqp_ii_yx, float * d_hqp_ii_yy, float * d_hqp_ii_yz,
		float * d_hqp_ii_zx, float * d_hqp_ii_zy, float * d_hqp_ii_zz, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if ((c >= k) || (r >= k))
		return;

	int idx = c + r * k;

	float3 pi = p[r];
	float3 pj = p[c];
	float gg = g[idx];
	float gg1 = gg * f;
	float3 ddq = dq[idx];

	float xx = 2.0 * gg1 * ddq.x * pi.x;
	float xy = 2.0 * gg1 * ddq.x * pi.y;
	float xz = 2.0 * gg1 * ddq.x * pi.z;

	float yx = 2.0 * gg1 * ddq.y * pi.x;
	float yy = 2.0 * gg1 * ddq.y * pi.y;
	float yz = 2.0 * gg1 * ddq.y * pi.z;

	float zx = 2.0 * gg1 * ddq.z * pi.x;
	float zy = 2.0 * gg1 * ddq.z * pi.y;
	float zz = 2.0 * gg1 * ddq.z * pi.z;

	d_hqp_xx[idx] = xx;
	d_hqp_xy[idx] = xy;
	d_hqp_xz[idx] = xz;

	d_hqp_yx[idx] = yx;
	d_hqp_yy[idx] = yy;
	d_hqp_yz[idx] = yz;

	d_hqp_zx[idx] = zx;
	d_hqp_zy[idx] = zy;
	d_hqp_zz[idx] = zz;

	xx = 2.0 * gg1 * ddq.x * pj.x;
	xy = 2.0 * gg1 * ddq.x * pj.y;
	xz = 2.0 * gg1 * ddq.x * pj.z;

	yx = 2.0 * gg1 * ddq.y * pj.x;
	yy = 2.0 * gg1 * ddq.y * pj.y;
	yz = 2.0 * gg1 * ddq.y * pj.z;

	zx = 2.0 * gg1 * ddq.z * pj.x;
	zy = 2.0 * gg1 * ddq.z * pj.y;
	zz = 2.0 * gg1 * ddq.z * pj.z;

	d_hqp_ii_xx[idx] = xx;
	d_hqp_ii_xy[idx] = xy;
	d_hqp_ii_xz[idx] = xz;

	d_hqp_ii_yx[idx] = yx;
	d_hqp_ii_yy[idx] = yy;
	d_hqp_ii_yz[idx] = yz;

	d_hqp_ii_zx[idx] = zx;
	d_hqp_ii_zy[idx] = zy;
	d_hqp_ii_zz[idx] = zz;
}

__global__ void hppPreComputeKernel(float * g,
		float * hpp_xx, float * hpp_yy, float * hpp_zz, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if ((c >= k) || (r >= k))
		return;

	int idx = c + r * k;
	float gg = g[idx];

	hpp_xx[idx] = gg;
	hpp_yy[idx] = gg;
	hpp_zz[idx] = gg;
}

__global__ void hppPreComputeKernel(float * g,
		float * hpp_xx, float * hpp_yy, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if ((c >= k) || (r >= k))
		return;

	int idx = c + r * k;
	float gg = g[idx];

	hpp_xx[idx] = gg;
	hpp_yy[idx] = gg;
}


__global__ void copyToDiagonal(float* matrix, float * diag, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= k)
		return;

	matrix[c + c * k] = diag[c];
}

__global__ void Float2Float2Kernel1D( float * input1, float * input2, float2 * output, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= k)
		return;

	output[c].x = input1[c];
	output[c].y = input2[c];
}

__global__ void Float2Float3Kernel1D( float * input1, float * input2, float * input3, float3 * output, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= k)
		return;

	output[c].x = input1[c];
	output[c].y = input2[c];
	output[c].z = input3[c];
}

__global__ void Float2Float4Kernel2D (
		float * inputxx, float * inputxy,
		float * inputyx, float * inputyy,
		float4 * output, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if ((c >= k) || (r >= k))
		return;
	int idx = c + r * k;

	output[idx].x = inputxx[idx];
	output[idx].y = inputxy[idx];
	output[idx].z = inputyx[idx];
	output[idx].w = inputyy[idx];

}

__global__ void Float2Float9Kernel2D (
		float * inputxx, float * inputxy, float * inputxz,
		float * inputyx, float * inputyy, float * inputyz,
		float * inputzx, float * inputzy, float * inputzz,
		float9 * output, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if ((c >= k) || (r >= k))
		return;
	int idx = c + r * k;

	output[idx].e00 = inputxx[idx];
	output[idx].e01 = inputxy[idx];
	output[idx].e02 = inputxz[idx];

	output[idx].e10 = inputyx[idx];
	output[idx].e11 = inputyy[idx];
	output[idx].e12 = inputyz[idx];

	output[idx].e20 = inputzx[idx];
	output[idx].e21 = inputzy[idx];
	output[idx].e22 = inputzz[idx];
}


__global__ void dbjiKernel( float3 * beta, float3 * db, int k ){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if ((c >= k) || (r >= k))
		return;
	int idx = c + r * k;

	float3 bi = beta[r];
	float3 bj = beta[c];
	float3 ddb;

	ddb.x = bj.x - bi.x;
	ddb.y = bj.y - bi.y;
	ddb.z = bj.z - bi.z;

	db[idx] = ddb;
}

__global__ void dbjiKernel( float2 * beta, float2 * db, int k ){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if ((c >= k) || (r >= k))
		return;
	int idx = c + r * k;

	float2 bi = beta[r];
	float2 bj = beta[c];
	float2 ddb;

	ddb.x = bj.x - bi.x;
	ddb.y = bj.y - bi.y;

	db[idx] = ddb;
}


__global__ void dalphaPrecomputeKernel(float * pi_pj, float3 * dq, float * g, float3 * dbji, float f, int k,
		float * da_pre_x, float * da_pre_y, float * da_pre_z,
		float3 * p, float3 * alpha ){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if ((c >= k) || (r >= k))
		return;
	int idx = c + r * k;

	float pipj = pi_pj[idx];
	float3 ddq = dq[idx];
	float gg = g[idx];
	float gg1 = gg * f;
	float gg2 = gg1 * f;

	float3 ddbji = dbji[idx];

	//the da terms
	float val_qq_xx = 2.0 * pipj * (2 * gg2 * ddq.x * ddq.x + gg1);
	float val_qq_xy = 2.0 * pipj * (2 * gg2 * ddq.x * ddq.y);
	float val_qq_xz = 2.0 * pipj * (2 * gg2 * ddq.x * ddq.z);

	float val_qq_yx = val_qq_xy;
	float val_qq_yy = 2.0 * pipj * (2 * gg2 * ddq.y * ddq.y + gg1);
	float val_qq_yz = 2.0 * pipj * (2 * gg2 * ddq.y * ddq.z);

	float val_qq_zx = val_qq_xz;
	float val_qq_zy = val_qq_yz;
	float val_qq_zz = 2.0 * pipj * (2 * gg2 * ddq.z * ddq.z + gg1);

	float dda_pre_x = ddbji.x * val_qq_xx + ddbji.y * val_qq_yx + ddbji.z * val_qq_zx;
	float dda_pre_y = ddbji.x * val_qq_xy + ddbji.y * val_qq_yy + ddbji.z * val_qq_zy;
	float dda_pre_z = ddbji.x * val_qq_xz + ddbji.y * val_qq_yz + ddbji.z * val_qq_zz;

	// the alpha * alpha terms
	float3 ai = alpha[r];
	float3 aj = alpha[c];

	float3 pi = p[r];
	float3 pj = p[c];

	float alpha_alpha =	aj.x * pi.x + ai.x * pj.x + aj.y * pi.y + ai.y * pj.y + aj.z * pi.z + ai.z * pj.z ;

	float aaa_pre_x = 2.0 * gg1 * ddq.x * alpha_alpha;
	float aaa_pre_y = 2.0 * gg1 * ddq.y * alpha_alpha;
	float aaa_pre_z = 2.0 * gg1 * ddq.z * alpha_alpha;

	da_pre_x[idx] = c==r?0: (dda_pre_x + aaa_pre_x);
	da_pre_y[idx] = c==r?0: (dda_pre_y + aaa_pre_y);
	da_pre_z[idx] = c==r?0: (dda_pre_z + aaa_pre_z);
}

__global__ void dalphaPrecomputeKernel(float * pi_pj, float2 * dq, float * g, float2 * dbji, float f, int k,
		float * da_pre_x, float * da_pre_y,
		float2 * p, float2 * alpha ){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if ((c >= k) || (r >= k))
		return;
	int idx = c + r * k;

	float pipj = pi_pj[idx];
	float2 ddq = dq[idx];
	float gg = g[idx];
	float gg1 = gg * f;
	float gg2 = gg1 * f;

	float2 ddbji = dbji[idx];

	//the da terms
	float val_qq_xx = 2.0 * pipj * (2 * gg2 * ddq.x * ddq.x + gg1);
	float val_qq_xy = 2.0 * pipj * (2 * gg2 * ddq.x * ddq.y);

	float val_qq_yx = val_qq_xy;
	float val_qq_yy = 2.0 * pipj * (2 * gg2 * ddq.y * ddq.y + gg1);

	float dda_pre_x = ddbji.x * val_qq_xx + ddbji.y * val_qq_yx;
	float dda_pre_y = ddbji.x * val_qq_xy + ddbji.y * val_qq_yy;

	// the alpha * alpha terms
	float2 ai = alpha[r];
	float2 aj = alpha[c];

	float2 pi = p[r];
	float2 pj = p[c];

	float alpha_alpha =	aj.x * pi.x + ai.x * pj.x + aj.y * pi.y + ai.y * pj.y;

	float aaa_pre_x = 2.0 * gg1 * ddq.x * alpha_alpha;
	float aaa_pre_y = 2.0 * gg1 * ddq.y * alpha_alpha;

	da_pre_x[idx] = c==r?0: (dda_pre_x + aaa_pre_x);
	da_pre_y[idx] = c==r?0: (dda_pre_y + aaa_pre_y);
}


__global__ void dbetaPrecomputeKernel( float3 * p, float3 * dq, float * g, float3 * dbji, float f, int k,
		float * db_pre_x, float * db_pre_y, float * db_pre_z,
		float3 * alpha){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if ((c >= k) || (r >= k))
		return;
	int idx = c + r * k;

	float3 ddbji = dbji[idx];

	float3 pj = p[c];

	float3 ddq = dq[idx];
	float gg = g[idx];
	float gg1 = gg * f;

	// The db term
	float factor = 2.0 * gg1 * (ddbji.x * ddq.x + ddbji.y * ddq.y + ddbji.z * ddq.z);

	float ddb_pre_x = factor * pj.x;
	float ddb_pre_y = factor * pj.y;
	float ddb_pre_z = factor * pj.z;

	// The alpha * g term
	float3 aa = alpha[c];
	float aag_pre_x = gg * aa.x;
	float aag_pre_y = gg * aa.y;
	float aag_pre_z = gg * aa.z;

	db_pre_x[idx] = c==r?0:(ddb_pre_x + aag_pre_x);
	db_pre_y[idx] = c==r?0:(ddb_pre_y + aag_pre_y);
	db_pre_z[idx] = c==r?0:(ddb_pre_z + aag_pre_z);

}

__global__ void dbetaPrecomputeKernel( float2 * p, float2 * dq, float * g, float2 * dbji, float f, int k,
		float * db_pre_x, float * db_pre_y,
		float2 * alpha){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if ((c >= k) || (r >= k))
		return;
	int idx = c + r * k;

	float2 ddbji = dbji[idx];

	float2 pj = p[c];

	float2 ddq = dq[idx];
	float gg = g[idx];
	float gg1 = gg * f;

	// The db term
	float factor = 2.0 * gg1 * (ddbji.x * ddq.x + ddbji.y * ddq.y);

	float ddb_pre_x = factor * pj.x;
	float ddb_pre_y = factor * pj.y;

	// The alpha * g term
	float2 aa = alpha[c];
	float aag_pre_x = gg * aa.x;
	float aag_pre_y = gg * aa.y;

	db_pre_x[idx] = c==r?0:(ddb_pre_x + aag_pre_x);
	db_pre_y[idx] = c==r?0:(ddb_pre_y + aag_pre_y);
}


__global__ void updateKernel1D(float2 * input, float2 * gradient, float epsilon, float2 * output, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= k)
		return;
	float2 in = input[c];
	float2 g = gradient[c];
	float2 out;

	out.x = in.x + epsilon * g.x;
	out.y = in.y + epsilon * g.y;

	output[c] = out;
}

__global__ void updateKernel1D(float3 * input, float3 * gradient, float epsilon, float3 * output, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= k)
		return;
	float3 in = input[c];
	float3 g = gradient[c];
	float3 out;

	out.x = in.x + epsilon * g.x;
	out.y = in.y + epsilon * g.y;
	out.z = in.z + epsilon * g.z;

	output[c] = out;
}

__global__ void KqPtKernel(float3 * q, float3 * p, float3 x, float f,
		float * KqPt_x, float * KqPt_y, float * KqPt_z, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= k)
		return;
	float3 qq = q[c];
	float3 pp = p[c];

	float dsq = (qq.x - x.x)*(qq.x - x.x) +
			(qq.y - x.y)*(qq.y - x.y) +
			(qq.z - x.z)*(qq.z - x.z);
	float Kq = exp(dsq * f);

	KqPt_x[c] =  Kq * pp.x ;
	KqPt_y[c] =  Kq * pp.y ;
	KqPt_z[c] =  Kq * pp.z ;
}

__global__ void KqPtKernel(float2 * q, float2 * p, float2 x, float f,
		float * KqPt_x, float * KqPt_y, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= k)
		return;
	float2 qq = q[c];
	float2 pp = p[c];

	float dsq = (qq.x - x.x)*(qq.x - x.x) +
				(qq.y - x.y)*(qq.y - x.y);
	float Kq = exp(dsq * f);

	KqPt_x[c] =  Kq * pp.x ;
	KqPt_y[c] =  Kq * pp.y ;
}

__global__ void GAlphaBetaKernel(float2 * q1, float2 * qT, float2 * p1,
		float2 * alpha, float2 * beta, float * gnsq, float * dsq,
		float lambda, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= k)
		return;

	float2 qq1 = q1[c];
	float2 qqT = qT[c];
	float2 pp1 = p1[c];

	float2 a;
	float2 G;
	float2 dq;

	dq.x = qq1.x - qqT.x;
	dq.y = qq1.y - qqT.y;

	G.x = pp1.x + lambda* dq.x;
	G.y = pp1.y + lambda* dq.y;

	gnsq[c] = G.x * G.x + G.y * G.y;
	dsq[c] = dq.x * dq.x + dq.y * dq.y;

	a.x = lambda * G.x;
	a.y = lambda * G.y;

	alpha[c] = a;
	beta[c] = G;
}

__global__ void GAlphaBetaKernel(float3 * q1, float3 * qT, float3 * p1,
		float3 * alpha, float3 * beta, float * gnsq, float * dsq,
		float lambda, int k){
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	if (c >= k)
		return;

	float3 qq1 = q1[c];
	float3 qqT = qT[c];
	float3 pp1 = p1[c];

	float3 a;
	float3 G;
	float3 dq;

	dq.x = qq1.x - qqT.x;
	dq.y = qq1.y - qqT.y;
	dq.z = qq1.z - qqT.z;

	G.x = pp1.x + lambda* dq.x;
	G.y = pp1.y + lambda* dq.y;
	G.z = pp1.z + lambda* dq.z;

	gnsq[c] = G.x * G.x + G.y * G.y + G.z * G.z;
	dsq[c] = dq.x * dq.x + dq.y * dq.y + dq.z * dq.z;

	a.x = lambda * G.x;
	a.y = lambda * G.y;
	a.z = lambda * G.z;

	alpha[c] = a;
	beta[c] = G;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernals for Accelerated hqhp and alphabeta
__device__ __inline__ void hqhpSingleTermCompute(float3 qi, float3 qj, float3 pi, float3 pj,
		float &hq_x, float &hq_y, float &hq_z,
		float &hp_x, float &hp_y, float &hp_z,
		float &ham, float f){
	float pipj;
	pipj = pi.x * pj.x + pi.y * pj.y + pi.z * pj.z;

	float3 dq;
	dq.x = qi.x - qj.x;
	dq.y = qi.y - qj.y;
	dq.z = qi.z - qj.z;

	float g = exp ( (dq.x * dq.x + dq.y * dq.y + dq.z * dq.z) * f);
	float g1 = g*f;
	float factor = 2 * pipj * g1;

	ham += g * pipj;

	hq_x += factor * dq.x;
	hq_y += factor * dq.y;
	hq_z += factor * dq.z;

	hp_x += g * pj.x;
	hp_y += g * pj.y;
	hp_z += g * pj.z;
}

__device__ __inline__ void hqhpSingleTermCompute(float2 qi, float2 qj, float2 pi, float2 pj,
		float &hq_x, float &hq_y,
		float &hp_x, float &hp_y,
		float &ham, float f){
	float pipj;
	pipj = pi.x * pj.x + pi.y * pj.y;

	float2 dq;
	dq.x = qi.x - qj.x;
	dq.y = qi.y - qj.y;

	float g = exp ( (dq.x * dq.x + dq.y * dq.y ) * f);
	float g1 = g*f;
	float factor = 2 * pipj * g1;

	ham += g * pipj;

	hq_x += factor * dq.x;
	hq_y += factor * dq.y;

	hp_x += g * pj.x;
	hp_y += g * pj.y;
}


__device__ __inline__ void alphaBetaSingleTermCompute(float3 qi, float3 pi, float3 qj, float3 pj,
		float3 ai, float3 bi, float3 aj, float3 bj, int c, int r, float f,
		float &da_x, float &da_y, float &da_z,
		float &db_x, float &db_y, float &db_z){
	/***** Precompute common terms ******/
	float pipj;
	pipj = pi.x * pj.x + pi.y * pj.y + pi.z * pj.z;

	float3 dq;
	dq.x = qi.x - qj.x;
	dq.y = qi.y - qj.y;
	dq.z = qi.z - qj.z;

	float g = exp ( (dq.x * dq.x + dq.y * dq.y + dq.z * dq.z) * f);
	float g1 = g*f;
	float g2 = g1*f;

	float3 bdji;

	bdji.x = bj.x - bi.x;
	bdji.y = bj.y - bi.y;
	bdji.z = bj.z - bi.z;

	/***** alpha terms ******/
	float val_qq_xx = 2.0 * pipj * (2 * g2 * dq.x * dq.x + g1);
	float val_qq_xy = 2.0 * pipj * (2 * g2 * dq.x * dq.y);
	float val_qq_xz = 2.0 * pipj * (2 * g2 * dq.x * dq.z);

	float val_qq_yx = val_qq_xy;
	float val_qq_yy = 2.0 * pipj * (2 * g2 * dq.y * dq.y + g1);
	float val_qq_yz = 2.0 * pipj * (2 * g2 * dq.y * dq.z);

	float val_qq_zx = val_qq_xz;
	float val_qq_zy = val_qq_yz;
	float val_qq_zz = 2.0 * pipj * (2 * g2 * dq.z * dq.z + g1);

	float dda_pre_x = bdji.x * val_qq_xx + bdji.y * val_qq_yx + bdji.z * val_qq_zx;
	float dda_pre_y = bdji.x * val_qq_xy + bdji.y * val_qq_yy + bdji.z * val_qq_zy;
	float dda_pre_z = bdji.x * val_qq_xz + bdji.y * val_qq_yz + bdji.z * val_qq_zz;

	// the alpha * alpha terms
	float alpha_alpha =	aj.x * pi.x + ai.x * pj.x + aj.y * pi.y + ai.y * pj.y + aj.z * pi.z + ai.z * pj.z ;

	float aaa_pre_x = 2.0 * g1 * dq.x * alpha_alpha;
	float aaa_pre_y = 2.0 * g1 * dq.y * alpha_alpha;
	float aaa_pre_z = 2.0 * g1 * dq.z * alpha_alpha;

	da_x += (c==r)? 0: (dda_pre_x + aaa_pre_x);
	da_y += (c==r)? 0: (dda_pre_y + aaa_pre_y);
	da_z += (c==r)? 0: (dda_pre_z + aaa_pre_z);

	/***** beta terms ******/
	// The db term
	float factor = 2.0 * g1 * (bdji.x * dq.x + bdji.y * dq.y + bdji.z * dq.z);

	float ddb_pre_x = factor * pj.x;
	float ddb_pre_y = factor * pj.y;
	float ddb_pre_z = factor * pj.z;

	// The alpha * g term
	float aag_pre_x = g * aj.x;
	float aag_pre_y = g * aj.y;
	float aag_pre_z = g * aj.z;

	db_x += (c==r)? ai.x:(ddb_pre_x + aag_pre_x);
	db_y += (c==r)? ai.y:(ddb_pre_y + aag_pre_y);
	db_z += (c==r)? ai.z:(ddb_pre_z + aag_pre_z);
}

__device__ __inline__ void alphaBetaSingleTermCompute(float2 qi, float2 pi, float2 qj, float2 pj,
		float2 ai, float2 bi, float2 aj, float2 bj, int c, int r, float f,
		float &da_x, float &da_y,
		float &db_x, float &db_y){
	/***** Precompute common terms ******/
	float pipj;
	pipj = pi.x * pj.x + pi.y * pj.y;

	float2 dq;
	dq.x = qi.x - qj.x;
	dq.y = qi.y - qj.y;

	float g = exp ( (dq.x * dq.x + dq.y * dq.y) * f);
	float g1 = g*f;
	float g2 = g1*f;

	float2 bdji;

	bdji.x = bj.x - bi.x;
	bdji.y = bj.y - bi.y;

	/***** alpha terms ******/
	float val_qq_xx = 2.0 * pipj * (2 * g2 * dq.x * dq.x + g1);
	float val_qq_xy = 2.0 * pipj * (2 * g2 * dq.x * dq.y);

	float val_qq_yx = val_qq_xy;
	float val_qq_yy = 2.0 * pipj * (2 * g2 * dq.y * dq.y + g1);

	float dda_pre_x = bdji.x * val_qq_xx + bdji.y * val_qq_yx;
	float dda_pre_y = bdji.x * val_qq_xy + bdji.y * val_qq_yy;

	// the alpha * alpha terms
	float alpha_alpha =	aj.x * pi.x + ai.x * pj.x + aj.y * pi.y + ai.y * pj.y;

	float aaa_pre_x = 2.0 * g1 * dq.x * alpha_alpha;
	float aaa_pre_y = 2.0 * g1 * dq.y * alpha_alpha;

	da_x += (c==r)? 0: (dda_pre_x + aaa_pre_x);
	da_y += (c==r)? 0: (dda_pre_y + aaa_pre_y);

	/***** beta terms ******/
	// The db term
	float factor = 2.0 * g1 * (bdji.x * dq.x + bdji.y * dq.y);

	float ddb_pre_x = factor * pj.x;
	float ddb_pre_y = factor * pj.y;

	// The alpha * g term
	float aag_pre_x = g * aj.x;
	float aag_pre_y = g * aj.y;

	db_x += (c==r)? ai.x:(ddb_pre_x + aag_pre_x);
	db_y += (c==r)? ai.y:(ddb_pre_y + aag_pre_y);
}

// One thread, 2 elements, 2 * blocksize get reduced
__global__ void hqhpRestrictedKernel ( float3 * q, float3 * p, float f,
		float3 * hq, float3 * hp,  float * ham, int k, int blockSize){
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if (r>=k)
		return;

	int tid = threadIdx.x;
	int c = 2 * blockDim.x * blockIdx.x + threadIdx.x;

	// Declare shared mem used in the reduce phase. Use float instead of float3 because in reduce phase
	// can use register shuffle trick
	extern __shared__ float sharedMem[];
	float * hq_local_x = &sharedMem[0];
	float * hq_local_y = &sharedMem[blockSize];
	float * hq_local_z = &sharedMem[2*blockSize];
	float * hp_local_x = &sharedMem[3*blockSize];
	float * hp_local_y = &sharedMem[4*blockSize];
	float * hp_local_z = &sharedMem[5*blockSize];
	float * ham_local = &sharedMem[6*blockSize];

	// Init variable
	float hq_pre_x;
	float hq_pre_y;
	float hq_pre_z;
	float hp_pre_x;
	float hp_pre_y;
	float hp_pre_z;
	float ham_pre;
	hq_pre_x = 0;
	hq_pre_y = 0;
	hq_pre_z = 0;
	hp_pre_x = 0;
	hp_pre_y = 0;
	hp_pre_z = 0;
	ham_pre = 0;

	/************* Precompute phase ******************/
	if (c<k){
		// 4 global read
		float3 qi = q[r];
		float3 pi = p[r];

		float3 qj = q[c];
		float3 pj = p[c];
		hqhpSingleTermCompute(qi, qj, pi, pj,
				hq_pre_x, hq_pre_y, hq_pre_z,
				hp_pre_x, hp_pre_y, hp_pre_z,
				ham_pre, f);

		if ((c+blockSize)<k){
			// Another 2 global read
			float3 qk = q[c+blockSize];
			float3 pk = p[c+blockSize];
			hqhpSingleTermCompute(qi, qk, pi, pk,
					hq_pre_x, hq_pre_y, hq_pre_z,
					hp_pre_x, hp_pre_y, hp_pre_z,
					ham_pre, f);

		}
	}
	// Put pre-compute result into shared mem
	hq_local_x[tid] = hq_pre_x;
	hq_local_y[tid] = hq_pre_y;
	hq_local_z[tid] = hq_pre_z;

	hp_local_x[tid] = hp_pre_x;
	hp_local_y[tid] = hp_pre_y;
	hp_local_z[tid] = hp_pre_z;

	ham_local[tid] = ham_pre;

	__syncthreads(); // wait till all threads in the same block finish

	/************* Reduce phase *********************/
	// do reduction in shared mem. When reduced down to single warp shift to
	// warp shuffling.
	for (unsigned int s=blockDim.x/2; s>32; s>>=1){
		if (tid < s){
			hq_local_x[tid] = hq_pre_x = hq_pre_x + hq_local_x[tid + s];
			hq_local_y[tid] = hq_pre_y = hq_pre_y + hq_local_y[tid + s];
			hq_local_z[tid] = hq_pre_z = hq_pre_z + hq_local_z[tid + s];

			hp_local_x[tid] = hp_pre_x = hp_pre_x + hp_local_x[tid + s];
			hp_local_y[tid] = hp_pre_y = hp_pre_y + hp_local_y[tid + s];
			hp_local_z[tid] = hp_pre_z = hp_pre_z + hp_local_z[tid + s];

			ham_local[tid] = ham_pre = ham_pre + ham_local[tid + s];
		}
		__syncthreads();
	}

	if ( tid < 32 ){
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64) {
			hq_pre_x += hq_local_x[tid + 32];
			hq_pre_y += hq_local_y[tid + 32];
			hq_pre_z += hq_local_z[tid + 32];

			hp_pre_x += hp_local_x[tid + 32];
			hp_pre_y += hp_local_y[tid + 32];
			hp_pre_z += hp_local_z[tid + 32];

			ham_pre += ham_local[tid + 32];
		}

		for (int offset = warpSize/2; offset > 0; offset /= 2){
			hq_pre_x += __shfl_down(hq_pre_x, offset);
			hq_pre_y += __shfl_down(hq_pre_y, offset);
			hq_pre_z += __shfl_down(hq_pre_z, offset);

			hp_pre_x += __shfl_down(hp_pre_x, offset);
			hp_pre_y += __shfl_down(hp_pre_y, offset);
			hp_pre_z += __shfl_down(hp_pre_z, offset);

			ham_pre += __shfl_down(ham_pre, offset);
		}
	}

	// TODO: need to try which is faster: this or do a reduction later or implement a atomicadd for float3
	if (tid == 0){
		atomicAdd( &hq[r].x, hq_pre_x);
		atomicAdd( &hq[r].y, hq_pre_y);
		atomicAdd( &hq[r].z, hq_pre_z);

		atomicAdd( &hp[r].x, hp_pre_x);
		atomicAdd( &hp[r].y, hp_pre_y);
		atomicAdd( &hp[r].z, hp_pre_z);

		atomicAdd( &ham[r], ham_pre );
	}
}

__global__ void hqhpRestrictedKernel ( float2 * q, float2 * p, float f,
		float2 * hq, float2 * hp,  float * ham, int k, int blockSize){
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if (r>=k)
		return;

	int tid = threadIdx.x;
	int c = 2 * blockDim.x * blockIdx.x + threadIdx.x;

	// Declare shared mem used in the reduce phase. Use float instead of float3 because in reduce phase
	// can use register shuffle trick
	extern __shared__ float sharedMem[];
	float * hq_local_x = &sharedMem[0];
	float * hq_local_y = &sharedMem[blockSize];
	float * hp_local_x = &sharedMem[2*blockSize];
	float * hp_local_y = &sharedMem[3*blockSize];
	float * ham_local = &sharedMem[4*blockSize];

	// Init variable
	float hq_pre_x;
	float hq_pre_y;
	float hp_pre_x;
	float hp_pre_y;
	float ham_pre;
	hq_pre_x = 0;
	hq_pre_y = 0;
	hp_pre_x = 0;
	hp_pre_y = 0;
	ham_pre = 0;

	/************* Precompute phase ******************/
	if (c<k){
		// 4 global read
		float2 qi = q[r];
		float2 pi = p[r];

		float2 qj = q[c];
		float2 pj = p[c];
		hqhpSingleTermCompute(qi, qj, pi, pj,
				hq_pre_x, hq_pre_y,
				hp_pre_x, hp_pre_y,
				ham_pre, f);

		if ((c+blockSize)<k){
			// Another 2 global read
			float2 qk = q[c+blockSize];
			float2 pk = p[c+blockSize];
			hqhpSingleTermCompute(qi, qk, pi, pk,
					hq_pre_x, hq_pre_y,
					hp_pre_x, hp_pre_y,
					ham_pre, f);

		}
	}
	// Put pre-compute result into shared mem
	hq_local_x[tid] = hq_pre_x;
	hq_local_y[tid] = hq_pre_y;

	hp_local_x[tid] = hp_pre_x;
	hp_local_y[tid] = hp_pre_y;

	ham_local[tid] = ham_pre;

	__syncthreads(); // wait till all threads in the same block finish

	/************* Reduce phase *********************/
	// do reduction in shared mem. When reduced down to single warp shift to
	// warp shuffling.
	for (unsigned int s=blockDim.x/2; s>32; s>>=1){
		if (tid < s){
			hq_local_x[tid] = hq_pre_x = hq_pre_x + hq_local_x[tid + s];
			hq_local_y[tid] = hq_pre_y = hq_pre_y + hq_local_y[tid + s];

			hp_local_x[tid] = hp_pre_x = hp_pre_x + hp_local_x[tid + s];
			hp_local_y[tid] = hp_pre_y = hp_pre_y + hp_local_y[tid + s];

			ham_local[tid] = ham_pre = ham_pre + ham_local[tid + s];
		}
		__syncthreads();
	}

	if ( tid < 32 ){
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64) {
			hq_pre_x += hq_local_x[tid + 32];
			hq_pre_y += hq_local_y[tid + 32];

			hp_pre_x += hp_local_x[tid + 32];
			hp_pre_y += hp_local_y[tid + 32];

			ham_pre += ham_local[tid + 32];
		}

		for (int offset = warpSize/2; offset > 0; offset /= 2){
			hq_pre_x += __shfl_down(hq_pre_x, offset);
			hq_pre_y += __shfl_down(hq_pre_y, offset);

			hp_pre_x += __shfl_down(hp_pre_x, offset);
			hp_pre_y += __shfl_down(hp_pre_y, offset);

			ham_pre += __shfl_down(ham_pre, offset);
		}
	}

	// TODO: need to try which is faster: this or do a reduction later or implement a atomicadd for float3
	if (tid == 0){
		atomicAdd( &hq[r].x, hq_pre_x);
		atomicAdd( &hq[r].y, hq_pre_y);

		atomicAdd( &hp[r].x, hp_pre_x);
		atomicAdd( &hp[r].y, hp_pre_y);

		atomicAdd( &ham[r], ham_pre );
	}
}

// One thread, 2 elements, 2 * blocksize get reduced
__global__ void alphaBetaRestrictedKernel ( float3 * q, float3 * p,
		float3 * alpha, float3 * beta,
		float3 * dalpha, float3 * dbeta,
		float f, int k, int blockSize){
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if (r>=k)
		return;

	int tid = threadIdx.x;
	int c = 2 * blockDim.x * blockIdx.x + threadIdx.x;

	// Declare shared mem used in the reduce phase. Use float instead of float3 because in reduce phase
	// can use register shuffle trick
	extern __shared__ float sharedMem[];
	float * da_local_x = &sharedMem[0];
	float * da_local_y = &sharedMem[blockSize];
	float * da_local_z = &sharedMem[2*blockSize];
	float * db_local_x = &sharedMem[3*blockSize];
	float * db_local_y = &sharedMem[4*blockSize];
	float * db_local_z = &sharedMem[5*blockSize];

	// Init variable
	float da_pre_x;
	float da_pre_y;
	float da_pre_z;
	float db_pre_x;
	float db_pre_y;
	float db_pre_z;
	da_pre_x = 0;
	da_pre_y = 0;
	da_pre_z = 0;
	db_pre_x = 0;
	db_pre_y = 0;
	db_pre_z = 0;

	/************* Precompute phase ******************/
	if (c<k){
		// 8 global read
		float3 qi = q[r];
		float3 pi = p[r];
		float3 ai = alpha[r];
		float3 bi = beta[r];

		float3 qj = q[c];
		float3 pj = p[c];
		float3 aj = alpha[c];
		float3 bj = beta[c];

		alphaBetaSingleTermCompute(qi, pi, qj, pj,
				ai, bi, aj, bj, c, r, f,
				da_pre_x, da_pre_y, da_pre_z,
				db_pre_x, db_pre_y, db_pre_z);

		if ((c+blockSize)<k){
			// Another 4 global read
			float3 qk = q[c+blockSize];
			float3 pk = p[c+blockSize];
			float3 ak = alpha[c+blockSize];
			float3 bk = beta[c+blockSize];

			alphaBetaSingleTermCompute(qi, pi, qk, pk,
							ai, bi, ak, bk, c+blockSize, r, f,
							da_pre_x, da_pre_y, da_pre_z,
							db_pre_x, db_pre_y, db_pre_z);

		}
	}

	// Put pre-compute result into shared mem
	da_local_x[tid] = da_pre_x;
	da_local_y[tid] = da_pre_y;
	da_local_z[tid] = da_pre_z;

	db_local_x[tid] = db_pre_x;
	db_local_y[tid] = db_pre_y;
	db_local_z[tid] = db_pre_z;

	__syncthreads(); // wait till all threads in the same block finish

	/************* Reduce phase *********************/
	// do reduction in shared mem. When reduced down to single warp shift to
	// warp shuffling.
	for (unsigned int s=blockDim.x/2; s>32; s>>=1){
		if (tid < s){
			da_local_x[tid] = da_pre_x = da_pre_x + da_local_x[tid + s];
			da_local_y[tid] = da_pre_y = da_pre_y + da_local_y[tid + s];
			da_local_z[tid] = da_pre_z = da_pre_z + da_local_z[tid + s];

			db_local_x[tid] = db_pre_x = db_pre_x + db_local_x[tid + s];
			db_local_y[tid] = db_pre_y = db_pre_y + db_local_y[tid + s];
			db_local_z[tid] = db_pre_z = db_pre_z + db_local_z[tid + s];
		}
		__syncthreads();
	}

	if ( tid < 32 ){
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64) {
			da_pre_x += da_local_x[tid + 32];
			da_pre_y += da_local_y[tid + 32];
			da_pre_z += da_local_z[tid + 32];

			db_pre_x += db_local_x[tid + 32];
			db_pre_y += db_local_y[tid + 32];
			db_pre_z += db_local_z[tid + 32];
		}

		for (int offset = warpSize/2; offset > 0; offset /= 2){
			da_pre_x += __shfl_down(da_pre_x, offset);
			da_pre_y += __shfl_down(da_pre_y, offset);
			da_pre_z += __shfl_down(da_pre_z, offset);

			db_pre_x += __shfl_down(db_pre_x, offset);
			db_pre_y += __shfl_down(db_pre_y, offset);
			db_pre_z += __shfl_down(db_pre_z, offset);
		}
	}

	// TODO: need to try which is faster: this or do a reduction later or implement a atomicadd for float3
	if (tid == 0){
		atomicAdd( &dalpha[r].x, da_pre_x);
		atomicAdd( &dalpha[r].y, da_pre_y);
		atomicAdd( &dalpha[r].z, da_pre_z);

		atomicAdd( &dbeta[r].x, db_pre_x);
		atomicAdd( &dbeta[r].y, db_pre_y);
		atomicAdd( &dbeta[r].z, db_pre_z);
	}
}

__global__ void alphaBetaRestrictedKernel ( float2 * q, float2 * p,
		float2 * alpha, float2 * beta,
		float2 * dalpha, float2 * dbeta,
		float f, int k, int blockSize){
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	if (r>=k)
		return;

	int tid = threadIdx.x;
	int c = 2 * blockDim.x * blockIdx.x + threadIdx.x;

	// Declare shared mem used in the reduce phase. Use float instead of float3 because in reduce phase
	// can use register shuffle trick
	extern __shared__ float sharedMem[];
	float * da_local_x = &sharedMem[0];
	float * da_local_y = &sharedMem[blockSize];
	float * db_local_x = &sharedMem[2*blockSize];
	float * db_local_y = &sharedMem[3*blockSize];

	// Init variable
	float da_pre_x;
	float da_pre_y;
	float db_pre_x;
	float db_pre_y;
	da_pre_x = 0;
	da_pre_y = 0;
	db_pre_x = 0;
	db_pre_y = 0;

	/************* Precompute phase ******************/
	if (c<k){
		// 8 global read
		float2 qi = q[r];
		float2 pi = p[r];
		float2 ai = alpha[r];
		float2 bi = beta[r];

		float2 qj = q[c];
		float2 pj = p[c];
		float2 aj = alpha[c];
		float2 bj = beta[c];

		alphaBetaSingleTermCompute(qi, pi, qj, pj,
				ai, bi, aj, bj, c, r, f,
				da_pre_x, da_pre_y,
				db_pre_x, db_pre_y);

		if ((c+blockSize)<k){
			// Another 4 global read
			float2 qk = q[c+blockSize];
			float2 pk = p[c+blockSize];
			float2 ak = alpha[c+blockSize];
			float2 bk = beta[c+blockSize];

			alphaBetaSingleTermCompute(qi, pi, qk, pk,
							ai, bi, ak, bk, c+blockSize, r, f,
							da_pre_x, da_pre_y,
							db_pre_x, db_pre_y);

		}
	}

	// Put pre-compute result into shared mem
	da_local_x[tid] = da_pre_x;
	da_local_y[tid] = da_pre_y;

	db_local_x[tid] = db_pre_x;
	db_local_y[tid] = db_pre_y;

	__syncthreads(); // wait till all threads in the same block finish

	/************* Reduce phase *********************/
	// do reduction in shared mem. When reduced down to single warp shift to
	// warp shuffling.
	for (unsigned int s=blockDim.x/2; s>32; s>>=1){
		if (tid < s){
			da_local_x[tid] = da_pre_x = da_pre_x + da_local_x[tid + s];
			da_local_y[tid] = da_pre_y = da_pre_y + da_local_y[tid + s];

			db_local_x[tid] = db_pre_x = db_pre_x + db_local_x[tid + s];
			db_local_y[tid] = db_pre_y = db_pre_y + db_local_y[tid + s];
		}
		__syncthreads();
	}

	if ( tid < 32 ){
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >=  64) {
			da_pre_x += da_local_x[tid + 32];
			da_pre_y += da_local_y[tid + 32];

			db_pre_x += db_local_x[tid + 32];
			db_pre_y += db_local_y[tid + 32];
		}

		for (int offset = warpSize/2; offset > 0; offset /= 2){
			da_pre_x += __shfl_down(da_pre_x, offset);
			da_pre_y += __shfl_down(da_pre_y, offset);

			db_pre_x += __shfl_down(db_pre_x, offset);
			db_pre_y += __shfl_down(db_pre_y, offset);
		}
	}

	// TODO: need to try which is faster: this or do a reduction later or implement a atomicadd for float3
	if (tid == 0){
		atomicAdd( &dalpha[r].x, da_pre_x);
		atomicAdd( &dalpha[r].y, da_pre_y);

		atomicAdd( &dbeta[r].x, db_pre_x);
		atomicAdd( &dbeta[r].y, db_pre_y);
	}
}







#endif
