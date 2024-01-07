    // specify use of OpenCL 1.2:
    #define CL_TARGET_OPENCL_VERSION  120
    #define CL_MINIMUM_OPENCL_VERSION 120
#include <iostream>
    #include <vector>
    #include <algorithm>
#include "timer.hpp"
    #include <boost/compute/algorithm/transform.hpp>
    #include <boost/compute/container/vector.hpp>
    #include <boost/compute/functional/math.hpp>
#include <boost/compute/algorithm/reduce.hpp>

    namespace compute = boost::compute;


BOOST_COMPUTE_FUNCTION(double, plus, (double x, double y),
                       {
                           return x + y;
                       });
BOOST_COMPUTE_FUNCTION(double, minus, (double x, double y),
                       {
                           return x - y;
                       });

BOOST_COMPUTE_FUNCTION(double, mult, (double x, double y),
                       {
                           return x * y;
                       });

int main(int argc, char *argv[])
    {
      int N = std::atoi(argv[1]);
      double time = 0;
      Timer timer;

      // get default device and setup context
        compute::device device = compute::system::default_device();
        compute::context context(device);
        compute::command_queue queue(context, device);

        // generate random data on the host
        std::vector<double> host_vector_x(N);
        std::vector<double> host_vector_y(N);
        std::fill(host_vector_x.begin(), host_vector_x.end(), 1.0);
        std::fill(host_vector_y.begin(), host_vector_y.end(), 2.0);

        // create a vector on the device
        compute::vector<double> device_vector_x(host_vector_x.size(), context);
        compute::vector<double> device_vector_y(host_vector_y.size(), context);
        compute::vector<double> device_plus(host_vector_x.size(), context);
        compute::vector<double> device_minus(host_vector_x.size(), context);
        compute::vector<double> device_dot(host_vector_x.size(), context);

        // transfer data from the host to the device
        compute::copy(
            host_vector_x.begin(), host_vector_x.end(), device_vector_x.begin(), queue
        );

        compute::copy(
            host_vector_y.begin(), host_vector_y.end(), device_vector_y.begin(), queue
        );

        double result = 0;
        timer.reset();
        // calculate the square-root of each element in-place
        compute::transform(
                           device_vector_x.begin(),
                           device_vector_x.end(),
                           device_vector_y.begin(),
                           device_plus.begin(),
                           plus,
                           queue
        );

        compute::transform(
                           device_vector_x.begin(),
                           device_vector_x.end(),
                           device_vector_y.begin(),
                           device_minus.begin(),
                           minus,
                           queue
                           );

        compute::transform(
                           device_plus.begin(),
                           device_plus.end(),
                           device_minus.begin(),
                           device_dot.begin(),
                           mult,
                           queue
                           );
        compute::reduce(device_dot.begin(), device_dot.end(), &result, queue);

        time = timer.get();
        std::cout << time << std::endl;

        return 0;
    }
