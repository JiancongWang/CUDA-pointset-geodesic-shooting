#ifndef HAMILTONIAN_H_
#define HAMILTONIAN_H_

#include "../include/float9.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

class PointSetHamiltonianSystem_CUDA{
public:
	// 3D functions
	static float ComputeHamiltonianJet_CUDA3D(float3 * h_q, float3 * h_p, float3 * h_hq, float3 * h_hp, float9 * h_hqq, float9 * h_hqp, float9 * h_hpp,
			float sigma, int k, bool flag_hessian, bool dataInDevice );
	static void ApplyHamiltonianHessianToAlphaBeta_CUDA3D(float3 * h_q, float3 * h_p,
			float3 * h_alpha, float3 * h_beta,
			float3 * h_dalpha, float3 * h_dbeta,
			float sigma, int k, bool dataInDevice );
	static float FlowHamiltonian_CUDA3D(float3 * h_q0, float3 * h_p0, float3 * h_q1, float3 * h_p1, float3 * h_hq, float3 * h_hp, std::vector<float3*> &Qt, std::vector<float3*> &Pt, float sigma, int k, int N, bool saveIntermediate, bool dataInDevice);
	static void FlowGradientBackward_CUDA3D(std::vector<float3*> &Qt, std::vector<float3*> &Pt, const float3 * h_alpha, const float3 * h_beta, float3 * h_result, float sigma, int k, int N, bool dataInDevice);

	// The unrestricted version of the two functions
	static float ComputeHamiltonianJet_CUDA3D_Restricted(float3 * h_q, float3 * h_p,
			float3 * h_hq, float3 * h_hp,
			float sigma, int k, int blockSize, bool dataInDevice );
	static void ApplyHamiltonianHessianToAlphaBeta_CUDA3D_Restricted(float3 * h_q, float3 * h_p,
			float3 * h_alpha, float3 * h_beta,
			float3 * h_dalpha, float3 * h_dbeta,
			float sigma, int k, int blockSize, bool dataInDevice );
	static float FlowHamiltonian_CUDA3D_Restricted(float3 * h_q0, float3 * h_p0, float3 * h_q1, float3 * h_p1, float3 * h_hq, float3 * h_hp, std::vector<float3*> &Qt, std::vector<float3*> &Pt,
			float sigma, int k, int N, int blockSize,
			bool saveIntermediate, bool dataInDevice);
	static void FlowGradientBackward_CUDA3D_Restricted(std::vector<float3*> &Qt, std::vector<float3*> &Pt, const float3 * h_alpha, const float3 * h_beta, float3 * h_result,
			float sigma, int k, int N, int blockSize, bool dataInDevice);


	static void InterpolateVelocity_CUDA3D(int t, const float3 x, float3 &v, std::vector<float3*> &Qt, std::vector<float3*> &Pt, float sigma, int k, bool dataInDevice);

	// Auxillary functions used in the LBFS optimizer
	static float landmarkError_CUDA3D(float3 * h_q1, float3 * h_qT, float3 * h_alpha, int k, bool dataInDevice);
	static void combineGradient_CUDA3D(float3 * h_grad, float3 * h_hp, int k, float lambda, bool dataInDevice);
	static void initP_CUDA3D(float3 * h_q0, float3 * h_qT, float3 * h_p0, int N, int k, bool dataInDevice);
	static void GAlphaBeta_CUDA3D(float3 * h_q1, float3 * h_qT, float3 * h_p1, float3 * h_alpha, float3 * h_beta, float &Gnorm_sq, float &dsq, float lambda, int k, bool dataInDevice );

	//TODO: finish this
	// Note this function is quite vram demanding to run! - 3D is a nightmare
	static void FlowHamiltonianWithGradient_CUDA3D( std::vector<float3*> &Qt, std::vector<float3*> &Pt,
			float3 * h_q0, float3 * h_p0, float3 * h_q, float3 * h_p, float9 * h_grad_q, float9 * h_grad_p,
			int N, int k, bool dataInDevice);

	// 2D functions
	static float ComputeHamiltonianJet_CUDA2D(float2 * h_q, float2 * h_p, float2 * h_hq, float2 * h_hp, float4 * h_hqq, float4 * h_hqp, float4 * h_hpp,
			float sigma, int k, bool flag_hessian, bool dataInDevice );
	static void ApplyHamiltonianHessianToAlphaBeta_CUDA2D(float2 * h_q, float2 * h_p,
			float2 * h_alpha, float2 * h_beta,
			float2 * h_dalpha, float2 * h_dbeta,
			float sigma, int k, bool dataInDevice );
	static float FlowHamiltonian_CUDA2D(float2 * h_q0, float2 * h_p0, float2 * h_q1, float2 * h_p1, float2 * h_hq, float2 * h_hp, std::vector<float2*> &Qt, std::vector<float2*> &Pt, float sigma, int k, int N, bool saveIntermediate, bool dataInDevice);
	static void FlowGradientBackward_CUDA2D(std::vector<float2*> &Qt, std::vector<float2*> &Pt, const float2 * d_alpha, const float2 * d_beta, float2 * d_result, float sigma, int k, int N, bool dataInDevice);


	// The unrestricted version of the two functions
	static float ComputeHamiltonianJet_CUDA2D_Restricted(float2 * h_q, float2 * h_p,
			float2 * h_hq, float2 * h_hp,
			float sigma, int k, int blockSize, bool dataInDevice );
	static void ApplyHamiltonianHessianToAlphaBeta_CUDA2D_Restricted(float2 * h_q, float2 * h_p,
			float2 * h_alpha, float2 * h_beta,
			float2 * h_dalpha, float2 * h_dbeta,
			float sigma, int k, int blockSize, bool dataInDevice );
	static float FlowHamiltonian_CUDA2D_Restricted(float2 * h_q0, float2 * h_p0, float2 * h_q1, float2 * h_p1,
			float2 * h_hq, float2 * h_hp, std::vector<float2*> &Qt, std::vector<float2*> &Pt,
			float sigma, int k, int N, int blockSize,
			bool saveIntermediate, bool dataInDevice);
	static void FlowGradientBackward_CUDA2D_Restricted(std::vector<float2*> &Qt, std::vector<float2*> &Pt,
			const float2 * h_alpha, const float2 * h_beta, float2 * h_result,
			float sigma, int k, int N, int blockSize, bool dataInDevice);



	static void InterpolateVelocity_CUDA2D(unsigned int t, const float2 x, float2 &v, std::vector<float2*> &Qt, std::vector<float2*> &Pt, float sigma, int k, bool dataInDevice);


	// Auxillary functions used in the LBFS optimizer
	static float landmarkError_CUDA2D(float2 * h_q1, float2 * h_qT, float2 * h_alpha, int k, bool dataInDevice);
	static void combineGradient_CUDA2D(float2 * h_grad, float2 * h_hp, int k, float lambda, bool dataInDevice);
	static void initP_CUDA2D(float2 * h_q0, float2 * h_qT, float2 * h_p0, int N, int k, bool dataInDevice);
	static void GAlphaBeta_CUDA2D(float2 * h_q1, float2 * h_qT, float2 * h_p1, float2 * h_alpha, float2 * h_beta, float &Gnorm_sq, float &dsq, float lambda, int k, bool dataInDevice );

	//TODO: finish this
	// Note this function is quite vram demanding to run! - 2D should be fine.
	static void FlowHamiltonianWithGradient_CUDA2D( std::vector<float2*> &Qt, std::vector<float2*> &Pt,
			float2 * h_q0, float2 * h_p0, float2 * h_q, float2 * h_p, float4 * h_grad_q, float4 * h_grad_p,
			int N, int k, bool dataInDevice);

private:
};

#endif
