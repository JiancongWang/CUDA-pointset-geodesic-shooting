class PointSetShootingCostFunction_CUDA : public vnl_cost_function{
public:
	typedef PointSetHamiltonianSystem_CUDA HSystem;
	typedef typename HSystem::Vector Vector;
	typedef typename HSystem::Matrix Matrix;

	// Separate type because vnl optimizer is double-only
	typedef vnl_vector<double> DVector;

	PointSetShootingCostFunction_CUDA(
			const ShootingParameters &param, const Matrix &q0, const Matrix &qT)
	: vnl_cost_function(), hsys(q0, param.sigma, param.N){

		this->param = param;

		// TODO: put all these assignment code outside the scope.
		this->p0 = (qT - q0) / param.N;
		this->qT = qT;
		this->k = q0.rows();
		this->p1.set_size(k, VDim);
		this->q1.set_size(k, VDim);

		for(unsigned int a = 0; a < VDim; a++){
			alpha[a].set_size(k);
			beta[a].set_size(k); beta[a].fill(0.0);
			grad_f[a].set_size(k);
		}
	}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	int PointSetShootingProblem_CUDA::minimize(const ShootingParameters &param, ShootingData &data){
		/* Read the datasets */
		vtkPolyData *pTemplate = ReadVTKData(param.fnTemplate);
		vtkPolyData *pTarget = ReadVTKData(param.fnTarget);

		// Get the number of vertices and dimensionality
		check(pTemplate->GetNumberOfPoints() == pTarget->GetNumberOfPoints(), "Meshes don't match");

		unsigned int np = pTemplate->GetNumberOfPoints();
		unsigned int k = np;

		// Mapping of matrix index to vertex - start with identity mapping
		std::vector<unsigned int> index(np);
		for(int i = 0; i < np; i++)
			index[i] = i;

		// Downsample meshes if needed
		if(param.downsample < 1.0){
			// Downsample by random shuffle - not very efficient, just lazy
			k = (unsigned int)(param.downsample * np);
			std::random_shuffle(index.begin(), index.end());
		}
		else if(param.qcdiv.size()){
			// Store the index of each input point in VTK mesh
			vtkSmartPointer<vtkIntArray> arr_idx = vtkIntArray::New();
			arr_idx->SetNumberOfComponents(1);
			arr_idx->SetNumberOfTuples(np);
			arr_idx->SetName("Index");
			for(int i = 0; i < np; i++)
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
			k = qc_result->GetNumberOfPoints();
			for(int i = 0; i < k; i++)
				index[i] = qc_index->GetTuple1(i);
		}

		// Report number of actual vertices being used
		cout << "Performing geodesic shooting with " << k << " landmarks" << endl;

		// Landmarks and initial momentum
		// TODO: change the input format here
		Matrix q0(k, VDim), qT(k, VDim), p0(k, VDim);

		// Initialize the meshes
		for(unsigned int i = 0; i < k; i++){
			for(unsigned int a = 0; a < VDim; a++){
				q0(i,a) = pTemplate->GetPoint(index[i])[a];
				qT(i,a) = pTarget->GetPoint(index[i])[a];
				p0(i,a) = (qT(i,a) - q0(i,a)) / param.N;
			}
		}

		/* Actual gradient descend */
		// Run some iterations of gradient descent
		if(param.iter_grad > 0){
			minimize_gradient(param, q0, qT, p0);
		}

		if(param.iter_newton > 0){
			minimize_Allassonniere(param, q0, qT, p0);
		}

		// Genererate the momentum map
		vtkDoubleArray *arr_p = vtkDoubleArray::New();
		arr_p->SetNumberOfComponents(VDim);
		arr_p->SetNumberOfTuples(np);
		arr_p->SetName("InitialMomentum");
		for(unsigned int a = 0; a < VDim; a++)
			arr_p->FillComponent(a, 0);

		for(unsigned int a = 0; a < VDim; a++){
			for(unsigned int i = 0; i < k; i++){
				arr_p->SetComponent(index[i],a,p0(i,a));
			}
		}

		pTemplate->GetPointData()->AddArray(arr_p);

		/* Output the final result */
		WriteVTKData(pTemplate, param.fnOutput);

		// If saving paths requested
		if(param.fnOutputPaths.size()){
			// Create and flow a system
			HSystem hsys(q0, param.sigma, param.N);
			Matrix q1, p1;
			hsys.FlowHamiltonian(p0, q1, p1);

			// Apply the flow to the points in the rest of the mesh
			vtkDoubleArray *arr_v = vtkDoubleArray::New();
			arr_v->SetNumberOfComponents(VDim);
			arr_v->SetNumberOfTuples(np);
			arr_v->SetName("Velocity");
			pTemplate->GetPointData()->AddArray(arr_v);

			// Apply Euler method to the mesh points
			double dt = hsys.GetDeltaT();
			for(unsigned int t = 1; t < param.N; t++){

				for(unsigned int i = 0; i < np; i++){
					float qi[VDim], vi[VDim];

					for(unsigned int a = 0; a < VDim; a++)
						qi[a] = pTemplate->GetPoint(i)[a];

					// Interpolate the velocity at each mesh point
					hsys.InterpolateVelocity(t-1, qi, vi);

					// Update the position using Euler's method
					for(unsigned int a = 0; a < VDim; a++)
						qi[a] += dt * vi[a];

					for(unsigned int a = 0; a < VDim; a++)
						pTemplate->GetPoints()->SetPoint(i, qi);
				}

				// Output the intermediate mesh
				char buffer[1024];
				sprintf(buffer, param.fnOutputPaths.c_str(), t);
				WriteVTKData(pTemplate, buffer);
			}
		}

		return 0;
	}
////////////////////////////////////////////////////////////////////////////////////////////////////////////
