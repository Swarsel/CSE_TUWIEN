
// The following three defines are necessary to pick the correct OpenCL version on the machine:
#define VEXCL_HAVE_OPENCL_HPP
#define CL_HPP_TARGET_OPENCL_VERSION  120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include <iostream>
#include <stdexcept>
#include <vexcl/vexcl.hpp>
#include "timer.hpp"

int main(int argc, char *argv[]) {
 vex::Context ctx(vex::Filter::GPU && vex::Filter::DoublePrecision);

 //std::cout << ctx << std::endl; // print list of selected devices

 int N = std::atoi(argv[1]);
 std::vector<double> a(N, 1.0), b(N, 2.0);

 vex::vector<double> A(ctx, a);
 vex::vector<double> B(ctx, b);

 vex::Reductor<double, vex::SUM> sum(ctx);

 double time = 0;
 Timer timer;
 double result = 0;
 timer.reset();
 result = sum((A + B) * (A - B));
 time = timer.get();
 std::cout << time << std::endl;
 return 0; }
