//
// Tutorial for demonstrating a simple OpenCL vector addition kernel
//
// Author: Karl Rupp    rupp@iue.tuwien.ac.at
//

typedef double       ScalarType;

// Specify the OpenCL target language version:
#define CL_TARGET_OPENCL_VERSION  220

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Helper include file for error checking
#include "ocl-error.hpp"
#include "timer.hpp"


const char *mSizeProgram(int M) {
  const char *init = ""
                     "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
                     "";
  const char *kernel_name = "__kernel void dot";
  const char *kernel_body = "(__global double *x,\n"
                            "                      __global double *y,\n"
                            "                      __global double *z,\n"
                            "                      unsigned int N\n)"
                            "{\n"
                            " __local double shared[128];\n"
                            " double sum = 0;\n"
                            " size_t id = get_local_id(0);\n"
                            "  for (unsigned int i  = get_global_id(0);\n"
                            "                    i  < N;\n"
                            "                    i += get_global_size(0))\n"
                            " sum += y[i] * x[i];\n"
                            " shared[id] = sum;\n"
                            " for (int stride  = get_local_size(0)/2;\n"
                            "                    stride  > 0;\n"
                            "                    stride /= 2)\n"
                            "   {\n"
                            " barrier(CLK_GLOBAL_MEM_FENCE);\n"
                            " if (id<stride) shared[id]+=shared[id+stride];\n"
                            "}\n"
                            " barrier(CLK_GLOBAL_MEM_FENCE);\n"
                            " if (id==0) z[get_group_id(0)]=shared[0];"
                            ""
                            "}";
  char *var = (char *)malloc( sizeof(char) * (std::string(init).length() + M * (4 + std::string(kernel_body).length() + std::string(kernel_name).length())));
  strcpy(var, init);

  for (int i = 0; i < M; i++) {
    strcat(var, kernel_name);
    std::string no = std::to_string(i);
    char const *no_char = no.c_str();
    strcat(var, no_char);
    strcat(var, kernel_body);
  }
  const char *out = var;
  return out;
}

const char *my_opencl_program = ""
                                "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"    // required to enable 'double' inside OpenCL programs
                                ""
                                "__kernel void dot(__global double *x,\n"
                                "                      __global double *y,\n"
                                "                      __global double *z,\n"
                                "                      unsigned int N\n)"
                                "{\n"
                                " __local double shared[128];\n"
                                " double sum = 0;\n"
                                " size_t id = get_local_id(0);\n"
                                "  for (unsigned int i  = get_global_id(0);\n"
                                "                    i  < N;\n"
                                "                    i += get_global_size(0))\n"
                                " sum += y[i] * x[i];\n"
                                " shared[id] = sum;\n"
                                " for (int stride  = get_local_size(0)/2;\n"
                                "                    stride  > 0;\n"
                                "                    stride /= 2)\n"
                                "   {\n"
                                " barrier(CLK_GLOBAL_MEM_FENCE);\n"
                                " if (id<stride) shared[id]+=shared[id+stride];\n"
                                "}\n"
                                " barrier(CLK_GLOBAL_MEM_FENCE);\n"
                                " if (id==0) z[get_group_id(0)]=shared[0];"
                                ""
                                "}";  // you can have multiple kernels within a single OpenCL program. For simplicity, this OpenCL program contains only a single kernel.


int main(int argc, char *argv[]) {
  cl_int err;
  int M = std::atoi(argv[1]);

  //
  /////////////////////////// Part 1: Set up an OpenCL context with one device ///////////////////////////////////
  //

  //
  // Query platform:
  //
  cl_uint num_platforms;
  cl_platform_id platform_ids[42];   //no more than 42 platforms supported...
  err = clGetPlatformIDs(42, platform_ids, &num_platforms); OPENCL_ERR_CHECK(err);
  // std::cout << "# Platforms found: " << num_platforms << std::endl;
  cl_platform_id my_platform = platform_ids[0];


  //
  // Query devices:
  //
  cl_device_id device_ids[42];
  cl_uint num_devices;
  err = clGetDeviceIDs(my_platform, CL_DEVICE_TYPE_ALL, 42, device_ids, &num_devices); OPENCL_ERR_CHECK(err);
  // std::cout << "# Devices found: " << num_devices << std::endl;
  cl_device_id my_device_id = device_ids[0];

  char device_name[64];
  size_t device_name_len = 0;
  err = clGetDeviceInfo(my_device_id, CL_DEVICE_NAME, sizeof(char)*63, device_name, &device_name_len); OPENCL_ERR_CHECK(err);
  // std::cout << "Using the following device: " << device_name << std::endl;

  //
  // Create context:
  //
  cl_context my_context = clCreateContext(0, 1, &my_device_id, NULL, NULL, &err); OPENCL_ERR_CHECK(err);


  //
  // create a command queue for the device:
  //
  cl_command_queue my_queue = clCreateCommandQueueWithProperties(my_context, my_device_id, 0, &err); OPENCL_ERR_CHECK(err);



  //
  /////////////////////////// Part 2: Create a program and extract kernels ///////////////////////////////////
  //

  Timer timer;
  timer.reset();

  //
  // Build the program:
  //
  const char *program = mSizeProgram(M);
  size_t source_len = std::string(program).length();
  timer.reset();
  cl_program prog = clCreateProgramWithSource(my_context, 1, &program, &source_len, &err);OPENCL_ERR_CHECK(err);
  err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
  std::cout << timer.get();

  //
  // Print compiler errors if there was a problem:
  //
  if (err != CL_SUCCESS) {

    char *build_log;
    size_t ret_val_size;
    err = clGetProgramBuildInfo(prog, my_device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
    build_log = (char *)malloc(sizeof(char) * (ret_val_size+1));
    err = clGetProgramBuildInfo(prog, my_device_id, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
    build_log[ret_val_size] = '\0'; // terminate string
    // std::cout << "Log: " << build_log << std::endl;
    free(build_log);
    // std::cout << "OpenCL program sources: " << std::endl << my_opencl_program << std::endl;
    return EXIT_FAILURE;
  }

  //
  // Extract the only kernel in the program:
  //
  cl_kernel my_kernel = clCreateKernel(prog, "dot0", &err); OPENCL_ERR_CHECK(err);

  // std::cout << "Time to compile and create kernel: " << timer.get() << std::endl;


  //
  /////////////////////////// Part 3: Create memory buffers ///////////////////////////////////
  //

  //
  // Set up buffers on host:
  //
  cl_uint vector_size = 128;
  std::vector<ScalarType> x(vector_size, 1.0);
  std::vector<ScalarType> y(vector_size, 2.0);
  std::vector<ScalarType> z(vector_size, 0.0);

  // std::cout << std::endl;
  // std::cout << "Vectors before kernel launch:" << std::endl;
  // std::cout << "x: " << x[0] << " " << x[1] << " " << x[2] << " ..." << std::endl;
  // std::cout << "y: " << y[0] << " " << y[1] << " " << y[2] << " ..." << std::endl;

  //
  // Now set up OpenCL buffers:
  //
  cl_mem ocl_x = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(ScalarType), &(x[0]), &err); OPENCL_ERR_CHECK(err);
  cl_mem ocl_y = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(ScalarType), &(y[0]), &err); OPENCL_ERR_CHECK(err);
  cl_mem ocl_z = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(ScalarType), &(z[0]), &err); OPENCL_ERR_CHECK(err);


  //
  /////////////////////////// Part 4: Run kernel ///////////////////////////////////
  //
  size_t  local_size = 128;
  size_t global_size = 128*128;

  //
  // Set kernel arguments:
  //
  err = clSetKernelArg(my_kernel, 0, sizeof(cl_mem),  (void*)&ocl_x); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 1, sizeof(cl_mem),  (void*)&ocl_y); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 2, sizeof(cl_mem),  (void*)&ocl_z); OPENCL_ERR_CHECK(err);
  err = clSetKernelArg(my_kernel, 3, sizeof(cl_uint), (void*)&vector_size); OPENCL_ERR_CHECK(err);

  //
  // Enqueue kernel in command queue:
  //
  timer.reset();
  err = clEnqueueNDRangeKernel(my_queue, my_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL); OPENCL_ERR_CHECK(err);

  // wait for all operations in queue to finish:
  err = clFinish(my_queue); OPENCL_ERR_CHECK(err);


  //
  /////////////////////////// Part 5: Get data from OpenCL buffer ///////////////////////////////////
  //

  err = clEnqueueReadBuffer(my_queue, ocl_z, CL_TRUE, 0, sizeof(ScalarType) * z.size(), &(z[0]), 0, NULL, NULL); OPENCL_ERR_CHECK(err);

  // std::cout << std::endl;
  // std::cout << "Vectors after kernel execution:" << std::endl;
  // std::cout << "z: " << z[0] << " " << z[1] << " " << z[2] << " ..." << std::endl;

  double dot_sum = 0;
  // this is crude, I know :sweat:
  for (int i = 0; i < 128; i++) {
    dot_sum += z[i];
  }
  // std::cout << dot_sum;
  // std::cout << timer.get();
  // std::cout << "y: " << y[0] << " " << y[1] << " " << y[2] << " ..." << std::endl;

  //
  // cleanup
  //
  clReleaseMemObject(ocl_x);
  clReleaseMemObject(ocl_y);
  clReleaseMemObject(ocl_z);
  clReleaseProgram(prog);
  clReleaseCommandQueue(my_queue);
  clReleaseContext(my_context);

  // std::cout << std::endl;
  // std::cout << "#" << std::endl;
  // std::cout << "# My first OpenCL application finished successfully!" << std::endl;
  // std::cout << "#" << std::endl;
  return EXIT_SUCCESS;
}