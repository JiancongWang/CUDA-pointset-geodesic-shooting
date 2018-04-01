# CUDA-pointset-geodesic-shooting
This repository contains code for CUDA implementation of pointset geodesic shooting. The original CPU implementation is kindly
provided by Dr. Yushkevich in the cmrep toolkit, which is also available on github with following link.
https://github.com/pyushkevich/cmrep

Compared to the CPU version, this CUDA implementation uses CUDA parallel reduction to compute the pairwise energy. It is 
considerably faster (60-70X on CUDA implementation on a GTX 980Ti vs CPU implementation on E5 2643 v3) and produce numerically more accurate registration. 

This code is tested under Ubuntu 16.04, with cmake 3.2, CUDA 8.0, gcc 5.4, VTK 7.0 (for reading mesh file). To compile, simply 
cd to the folder and run

cmake ./

make -j4

After compiling, 2 command line executable will be generated: lmshoot_cuda and shoot_forward.  Given two set of points, the executable lmshoot_cuda will generate the initial momentum used by the geodesic shooting algorithm. To compute the final registration position, please use the executable shoot_forward.
To test the executable, one can run ./test_lmshoot.sh then ./test_shootforward.sh, which will test the registration on set of test data in folder /data. The output VTK mesh file can be visualized in 3D slicer as a movie.

To view all command line options for the two executable, check out file ./src/PointSetGeodesicShooting_CUDA.cpp (lmshoot_cuda) and ./src/ShootForward.cpp (shoot_forward). To view the actual reduction implementation, check out file 
./src/hamiltonian.cu and ./src/hamiltonian_kernel.cu. 

To use this code in academic purpose, please cite paper "Fast geodesic shooting for landmark matching using CUDA". 
Thanks. 
