#include <iostream>
#include <stdio.h>
#include <algorithm>

#include "../include/CommandLineHelper.h"
#include "../include/PointSetShootingProblem.h"
#include "../include/check.h"

using namespace std;

int usage(){
	cout << "lmshoot ver CUDA: Geodesic shooting for landmarks implemented with CUDA" << endl;
	cout << "(Considering most of you will run this with GTX cards, it is available in float32 only )" << endl;
	cout << "Usage:" << endl;
	cout << "  lmshoot [options]" << endl;
	cout << "Required Options:" << endl;
	cout << "  -m momemtum.vtk : input meshes" << endl;
	cout << "  -o q1.vtk              : (final points position)" << endl;
	cout << "  -s sigma                   : kernel standard deviation" << endl;
	cout << "  -l lambda                  : weight of landmark distance term" << endl;
	cout << "Additional Options" << endl;
	cout << "  -d dim                     : problem dimension (3)" << endl;
	cout << "  -n N                       : number of time steps (100)" << endl;
	cout << "  -i iter_grad iter_newt     : max iterations for optimization for gradient descent and newton's" << endl;
	cout << "  -r fraction                : randomly downsample mesh by factor (e.g. 0.01)" << endl;
	cout << "  -q vector                  : apply quadric clustering to mesh with specified" <<  endl;
	cout << "                               number of bins per dimension (e.g. 40x40x30)" << endl;
	//  cout << "  -a <A|G>                   : algorithm to use: A: Allassonniere; G: GradDescent (deflt)" << endl;
	cout << "  -O filepattern             : pattern for saving traced landmark paths (e.g., path%04d.vtk)" << endl;
	return -1;
}


int main(int argc, char *argv[]){
	if(argc < 2)
		return usage();

	ShootingParameters param;
	string n;

	// Process parameters
	CommandLineHelper cl(argc, argv);
	while(!cl.is_at_end()){
		// Read the next command
		std::string arg = cl.read_command();

		if(arg == "-m"){
			param.momemtumPaths = cl.read_existing_filename();
		}
		else if(arg == "-o"){
			param.q1outPaths = cl.read_output_filename();
		}
		else if(arg == "-O"){
			param.fnOutputPaths = cl.read_string();
		}
		else if(arg == "-s"){
			param.sigma = cl.read_double();
		}
		else if(arg == "-l"){
			param.lambda = cl.read_double();
		}
		else if(arg == "-r"){
			param.downsample = cl.read_double();
		}
		else if(arg == "-n"){
			param.N = (unsigned int) cl.read_integer();
		}
		else if(arg == "-d"){
			param.dim = (unsigned int) cl.read_integer();
		}
		else if(arg == "-i"){
			param.iter_grad = (unsigned int) cl.read_integer();
			param.iter_newton = (unsigned int) cl.read_integer();
		}
		else if(arg == "-q"){
			param.qcdiv = cl.read_int_vector();
		}
		else if (arg == "-a"){
			string alg = cl.read_string();
			if(alg == "a" || alg == "A" || alg == "Allassonniere")
				param.alg = ShootingParameters::Allassonniere;
			else if(alg == "q" || alg == "Q" || alg == "QuasiAllassonniere")
				param.alg = ShootingParameters::QuasiAllassonniere;
			else
				param.alg = ShootingParameters::GradDescent;
		}
		else if(arg == "-h"){
			return usage();
		}
		else{
			cerr << "Unknown option " << arg << endl;
			return -1;
		}
	}

	// Check parameters
	Check::check(param.sigma > 0, "Missing or negative sigma parameter");
	Check::check(param.N > 0 && param.N < 10000, "Incorrect N parameter");
	Check::check(param.dim >= 2 && param.dim <= 3, "Incorrect N parameter");

	// Run the actual program
	PointSetShootingProblem_CUDA::CalculateForward(param);

	return 0;
}
