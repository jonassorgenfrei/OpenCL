/**
 * Vadd_kernel
 *
 * Element wise addition of two vectors (c = a + b | d = c + e | f = d + g | return f)
 *
 * In plain C code style
 *
 */

// enable opencl exceptions
#define __CL_ENABLE_EXCEPTIONS

#include "CL/cl.hpp"    // Khronos C++ Wrapper API

#include "filesystem.h"
#include "util.hpp"

#include <chrono> 
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
    std::vector<float> h_d(LENGTH, 0xdeadbeef);     // d vector (c+e) returned from the compute device
    std::vector<float> h_e(LENGTH);                 // e vector
    std::vector<float> h_f(LENGTH, 0xdeadbeef);     // f vector (d+g) returned from the compute device
    std::vector<float> h_g(LENGTH);                 // g vector


    std::vector<float> h_a3(LENGTH);                 // a3 vector
    std::vector<float> h_b3(LENGTH);                 // b3 vector
    std::vector<float> h_c3(LENGTH);                 // c3 vector
    std::vector<float> h_d3(LENGTH, 0xdeadbeef);     // d3 vector (a+b+c) returned from the compute device

    cl::Buffer d_a;                 // device memory used for the input  a vector
    cl::Buffer d_b;                 // device memory used for the input  b vector
    cl::Buffer d_c;                 // device memory used for the output c vector
    cl::Buffer d_d;                 // device memory used for the output d vector
    cl::Buffer d_e;                 // device memory used for the output e vector
    cl::Buffer d_f;                 // device memory used for the output f vector
    cl::Buffer d_g;                 // device memory used for the output g vector

    cl::Buffer d_a3;                 // device memory used for the input  a vector
    cl::Buffer d_b3;                 // device memory used for the input  b vector
    cl::Buffer d_c3;                 // device memory used for the output c vector
    cl::Buffer d_d3;                 // device memory used for the output d vector

    // fill vectors a and b with random float values
    int count = LENGTH;

    // create input vectors and assign values on the host
    for (int i = 0; i < count; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;

        h_e[i] = rand() / (float)RAND_MAX;
        h_g[i] = rand() / (float)RAND_MAX;

        h_a3[i] = rand() / (float)RAND_MAX;
        h_b3[i] = rand() / (float)RAND_MAX;
        h_c3[i] = rand() / (float)RAND_MAX;
    }

    try 
    {
        // create a context
        cl::Context context(DEVICE);

        // Load in kernel source, creating a program object for the context

        // the 3rd parameter specifis that the programm should be build
        // if build flags needs to be specified use program.build()
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

        d_e = cl::Buffer(context, begin(h_e), end(h_e), true);
        d_g = cl::Buffer(context, begin(h_g), end(h_g), true);


        d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);
        d_d = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);
        d_f = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

        // start timepoint
        auto start = std::chrono::high_resolution_clock::now();

        // RUN c = a + b
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

        // RUN d = c + e
        vadd(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(count)
                ),
            d_c,
            d_e,
            d_d,
            count);

        queue.finish();

        // RUN f = d + g
        vadd(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(count)
                ),
            d_d,
            d_g,
            d_f,
            count);

        queue.finish();

        // end time stopping
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        std::cout << "Time taken by execution: "  << duration.count() << " microseconds" << std::endl;


        // copy data back from device
        cl::copy(queue, d_f, begin(h_f), end(h_f));

        // test the results
        int correct = 0;
        float tmp;

        for (int i = 0; i < count; i++) {
            tmp = h_a[i] + h_b[i] + h_e[i] + h_g[i];    // expected value for d_c[i]
            tmp -= h_f[i];            // compute errors
            if (tmp * tmp < TOL * TOL) {    // correct if square deviation is less
                correct++;                  // than tollenace squared
            }
            else {
                printf("tmp %f h_a %f + h_b %f + h_e %f + h_g %f != h_f %f \n", tmp, h_a[i], h_b[i], h_e[i], h_g[i], h_f[i]);
            }

        }

        // summarize results
        std::cout << "vector add to find C = A+B D=C+E F=D+G Checked F: " << correct << " out of " << count << " results were correct" << std::endl;

        // vadd 3
        // ------

        // Load in kernel source, creating a program object for the context

        // the 3rd parameter specifis that the programm should be build
        // if build flags needs to be specified use program.build()
        cl::Program program_3(context, util::loadProgram(FileSystem::getPath("kernel/vadd3.cl")), true);


        // Get the command queue
        cl::CommandQueue queue_3(context);

        // create the kernel functor
        auto vadd_3 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int>(program_3, "vadd3");


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
        d_a3 = cl::Buffer(context, begin(h_a3), end(h_a3), true);
        d_b3 = cl::Buffer(context, begin(h_b3), end(h_b3), true);
        d_c3 = cl::Buffer(context, begin(h_c3), end(h_c3), true);
       
        d_d3 = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);


        // start timepoint
        start = std::chrono::high_resolution_clock::now();

        // RUN d = a + b + c
        vadd_3(
            cl::EnqueueArgs(
                queue_3,
                cl::NDRange(count)
                ),
            d_a3,
            d_b3,
            d_c3,
            d_d3,
            count);

        queue.finish();


        // end time stopping
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        std::cout << "Time taken by execution: " << duration.count() << " microseconds" << std::endl;

        // copy data back from device
        cl::copy(queue_3, d_d3, begin(h_d3), end(h_d3));

        // test the results
        correct = 0;
        tmp;

        for (int i = 0; i < count; i++) {
            tmp = h_a3[i] + h_b3[i] + h_c3[i];    // expected value for d_d3[i]
            tmp -= h_d3[i];            // compute errors
            if (tmp * tmp < TOL * TOL) {    // correct if square deviation is less
                correct++;                  // than tollenace squared
            }
            else {
                printf("tmp %f h_a3 %f + h_b3 %f + h_c3 %f != h_d3 %f \n", tmp, h_a3[i], h_b3[i], h_c3[i], h_d3[i]);
            }

        }

        // summarize results
        std::cout << "vector add to find D3 = A3+B3+C3: " << correct << " out of " << count << " results were correct" << std::endl;

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