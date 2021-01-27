/**
 * Vadd_kernel
 *
 * Element wise addition of two vectors (c = a + b)
 *
 * In plain C code style
 *
 */

// enable opencl exceptions
#define __CL_ENABLE_EXCEPTIONS

#include "CL/cl.hpp"    // Khronos C++ Wrapper API

#include "filesystem.h"
#include "util.hpp"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>

#include "config.h"

 //  define VERBOSE if you want to print info about work groups sizes
 //#define  VERBOSE 1

 // pick up device type from compiler command line or from
 // the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c


// --------------------------------------------------------------------------------------
int main(void)
{
    // Print Programm Infos
    std::cout << "OpenCL Vadd_Kernel CPP - Version " << 
        Vadd_Kernel_cpp_VERSION_MAJOR << "." << Vadd_Kernel_cpp_VERSION_MINOR << std::endl;

    // declare variables
    std::vector<float> h_a(LENGTH);                 // a vector
    std::vector<float> h_b(LENGTH);                 // b vector
    std::vector<float> h_c(LENGTH, 0xdeadbeef);     // c vector (a+b) returned from the compute device

    cl::Buffer d_a;                 // device memory used for the input  a vector
    cl::Buffer d_b;                 // device memory used for the input  b vector
    cl::Buffer d_c;                 // device memory used for the output c vector

    // fill vectors a and b with random float values
    int count = LENGTH;

    // create input vectors and assign values on the host
    for (int i = 0; i < count; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    try 
    {
        // create a context
        cl::Context context(DEVICE);

        // Load in kernel source, creating a program object for the context

        cl::Program program(context, util::loadProgram(FileSystem::getPath("kernel/vadd.cl")), true);


        // Get the command queue
        cl::CommandQueue queue(context);

        // create the kernel functor
        auto vadd = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "vadd");
              

        // buffer construction
        // - this is a blcoking call
        // - constructor will enqueue a copy to the first device in the context
        // - ocl runtime will AUTOMATICALLY ensure the buffer is copied across to the actual device you enqueue a kenrel on later
        //   if you enqueue the kernel on a different device within this context
        // 1st parameter context to use
        // 2nd parameter startIterator
        // 3rd parameter endIterator
        // 4th parameter readOnly boolen, specifies whether the memory is: CL_MEM_READ_ONLY (true) or CL_MEM_READ_WRITE (false)
        // 5th parameter useHostPtr, array defined by itrators is implicitly copied into device memory(default: false)
        d_a = cl::Buffer(context, begin(h_a), end(h_a), true);
        d_b = cl::Buffer(context, begin(h_b), end(h_b), true);

        d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

        // TODO: start time stopping


        vadd(
            cl::EnqueueArgs(
                            queue,
                            cl::NDRange(count)
                            ),
            d_a,
            d_b,
            d_c,
            count);

        queue.finish();

        // TODO: end time stopping

        // copy data back from device
        cl::copy(queue, d_c, begin(h_c), end(h_c));

        // test the results
        int correct = 0;
        float tmp;

        for (int i = 0; i < count; i++) {
            tmp = h_a[i] + h_b[i];    // expected value for d_c[i]
            tmp -= h_c[i];            // compute errors
            if (tmp * tmp < TOL * TOL) {    // correct if square deviation is less
                correct++;                  // than tollenace squared
            }
            else {
                printf("tmp %f h_a %f h_b %f h_c %f \n", tmp, h_a[i], h_b[i], h_c[i]);
            }

        }

        // summarize results
        std::cout << "vector add to find C = A+B: " << correct << " out of " << count << " results were correct" << std::endl;

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