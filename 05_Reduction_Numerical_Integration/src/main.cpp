/**
 * Pi program in OpenCL
 * Numerical integration of 4/(1+x*x) from 0 to 1
 *
 * Cpp code style
 */

// enable opencl exceptions
#define __CL_ENABLE_EXCEPTIONS

#include "CL/cl.hpp"    // Khronos C++ Wrapper API

#include <chrono> 

#define _USE_MATH_DEFINES // for C++
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "filesystem.h"
#include "util.hpp"

#include <iostream>
#include <fstream>

#include "config.h"

// device index
#define DEVICE_INDEX 0

// flag if CPU Matrix multiplication shall be run
#define RUN_CPU 1

//------------------------------------------------------------------------------

#define INSTEPS (512*512*512)
#define ITERS (262144)

static long num_steps = 100000000;
double step;

// --------------------------------------------------------------------------------------

/// <summary>
/// Gets the device list.
/// </summary>
/// <param name="devices">The devices.</param>
/// <returns></returns>
unsigned getDeviceList(std::vector<cl::Device>& devices)
{
    cl_int err;

    // Get list of platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // Enumerate devices
    for (int i = 0; i < platforms.size(); i++)
    {
        cl_uint num = 0;
        std::vector<cl::Device> plat_devices;
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &plat_devices);
        devices.insert(devices.end(), plat_devices.begin(), plat_devices.end());
    }

    return devices.size();
}


int main(void)
{

    // Print Programm Infos
    std::cout << "OpenCL integral of pi - Version " << 
       VERSION_MAJOR << "." << VERSION_MINOR << std::endl;

    // declare variables
    int i;
    double x, pi, sum = 0.0;

    step = 1.0 / (double)num_steps;

    // Run sequential mat mult on CPU
#if RUN_CPU
    {

        // start timepoint
        auto start = std::chrono::high_resolution_clock::now();

        for (i = 1; i <= num_steps; i++) {
            x = (i - 0.5) * step;
            sum = sum + 4.0 / (1.0 + x * x);
        }
        pi = step * sum;

        // end time stopping
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        std::cout << " pi with " << num_steps << " steps is " << pi << " in " << duration.count() / 1000 << " milliseconds" << std::endl;

        auto error = pi - CL_M_PI;
    }
#endif

    float* h_psum;                  // vector to hold partial sum
    int in_nsteps = INSTEPS;        // default number of steps (updated later to device preferable)
    int niters = ITERS;             // number of iterations
    int nsteps;
    float step_size;
    ::size_t nwork_groups;
    ::size_t max_size, work_group_size = 8;
    float pi_res;

    cl::Buffer d_partial_sums;

    try 
    {        
        // Get list of devices
        std::vector<cl::Device> devices;
        unsigned numDevices = getDeviceList(devices);

        // check if device indes is in range
        if (DEVICE_INDEX >= numDevices)
        {
            std::cout << "Invalid device index \n" << std::endl;
            return EXIT_FAILURE;
        }

        cl::Device device = devices[DEVICE_INDEX];
        
        // print device name of the chosen device 
        std::string name = device.getInfo<CL_DEVICE_NAME>();
        std::cout << "\nUsing OpenCL Device " << name << std::endl;

        // chosen device needs to be pushed in as an array
        std::vector<cl::Device> chosen_device;
        chosen_device.push_back(device);

        // create a context
        cl::Context context(chosen_device);
        // Get the command queue
        cl::CommandQueue queue(context, device);

 
        // Load in kernel source, creating a program object for the context

        // the 3rd parameter specifis that the programm should be build
        // if build flags needs to be specified use program.build()
        cl::Program program(context, util::loadProgram(FileSystem::getPath("kernel/numIntegration.cl")), true);
       
        // create the kernel functor
        cl::Kernel ko_pi(program, "pi");

        // get work group size
        work_group_size = ko_pi.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

        std::cout << "wgroup_size = " << work_group_size << std::endl;
              
        cl::make_kernel<int, float, cl::LocalSpaceArg, cl::Buffer> pi(program, "pi");

        // set the number of work groups, the actual number of steps and step size
        nwork_groups = in_nsteps / (work_group_size * niters);

        if (nwork_groups < 1) {
            nwork_groups = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
            std::cout << " MAX COMPUTE UNITS " << nwork_groups << std::endl;
            work_group_size = in_nsteps / (nwork_groups * niters);
        }

        nsteps = work_group_size * niters * nwork_groups;
        step_size = 1.0f / static_cast<float>(nsteps);
        std::vector<float> h_psum(nwork_groups);

        std::cout << (int)nwork_groups << " work groups of size " << (int)work_group_size << ". " << nsteps << " Integration steps" << std::endl;

        // initialize buffer
        d_partial_sums = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * nwork_groups);

        // start timepoint
        auto start = std::chrono::high_resolution_clock::now();

        // execute the kernel over the entire range of our 1d input data set
        // using the max number of work group items for this device
        pi(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(nsteps / niters),
                cl::NDRange(work_group_size)),
            niters,
            step_size,
            cl::Local(sizeof(float)* work_group_size),
            d_partial_sums);

        // copy partial sum back to cpu
        cl::copy(queue, d_partial_sums, h_psum.begin(), h_psum.end());

        // complete the sum and compute final integral value
        pi_res = 0.0f;
        for (unsigned int i = 0; i < nwork_groups; i++)
        {   
            pi_res += h_psum[i];
        }

        pi_res = pi_res * step_size;

        // end time stopping
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        auto error = pi_res - CL_M_PI;

        std::cout << "The calculation ran in " << duration.count() / 1000 << " milliseconds";
        std::cout << " pi = " << pi_res << " for " << nsteps << " steps.";
        std::cout << " Error: " << error << std::endl;

    }
    // catch opencl error
    catch (cl::Error err) {
        // catch errors and print error data
        std::cout << "OpenCL Error:" << err.what() << " returned " << std::endl;
        std::cout << "Check cl.h for error codes." << std::endl;

        exit(-1);
    } 

    return 0;
   
}