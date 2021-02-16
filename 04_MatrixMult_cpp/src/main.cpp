/**
 * Matrix Multiplication in OpenCL
 *
 *
 * Cpp code style
 */

// enable opencl exceptions
#define __CL_ENABLE_EXCEPTIONS

#include "CL/cl.hpp"    // Khronos C++ Wrapper API

#include <chrono> 
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "filesystem.h"
#include "util.hpp"
#include "matrix_lib.h"

#include <iostream>
#include <fstream>

#include "config.h"

// device index
#define DEVICE_INDEX 0

// flag if CPU Matrix multiplication shall be run
#define RUN_CPU 0

//------------------------------------------------------------------------------

#define TOL     (0.001) // tolerance used in floating point comparisons
#define ORDER   1024    // order of the square matrices A,B and C
#define COUNT   10       // number of times to do each multiplication

#define AVAL    3.0     // A elements are constant and equal to AVAL
#define BVAL    5.0     // B elements are constant and equal to BVAL

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
    std::cout << "OpenCL Expanded Vadd_Kernel CPP - Version " << 
       VERSION_MAJOR << "." << VERSION_MINOR << std::endl;

    // declare variables
    std::vector<float> h_A, h_B, h_C, h_Ctest;  // matrices

    int Ndim = ORDER;                           // init dimensions to a global order  A[N][N], B[N][N], C[N][N]

    int szA, szB, szC;                          // num elements in each matrix

    szA = Ndim * Ndim;                          // sizes of the matrices
    szB = Ndim * Ndim;                          // sizes of the matrices
    szC = Ndim * Ndim;                          // sizes of the matrices

    // allocate host memory for matrices
    h_A = std::vector<float>(szA);
    h_B = std::vector<float>(szB);
    h_C = std::vector<float>(szC);
    h_Ctest = std::vector<float>(szC);

    // intialize opencl buffers for matrices
    cl::Buffer d_a, d_b, d_c;                             // matrices in device memory

    // initialize matrices a and b with float values
    initmat(Ndim, Ndim, h_A, AVAL);
    initmat(Ndim, Ndim, h_B, BVAL);
    // zero mat C
    initmat(Ndim, Ndim, h_C, 0.0f);

    // Run sequential mat mult on CPU
#if RUN_CPU
    {

        std::cout << "\n==== = Sequential, matrix mult(dot prod), order " << Ndim << " on host CPU ======\n" << std::endl;
        // start timepoint
        auto start = std::chrono::high_resolution_clock::now();

        mat_mul(Ndim, h_A, h_B, h_C);

        // end time stopping
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        std::cout << "Time taken by execution: " << duration.count()/(1000*1000) << " seconds" << std::endl;
    }
#endif

    // initialize matrices a and b with float values
    initmat(Ndim, Ndim, h_A, AVAL);
    initmat(Ndim, Ndim, h_B, BVAL);
    // zero mat C
    initmat(Ndim, Ndim, h_C, 0.0f);

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

        //--------------------------------------------------------------------------------
        // OpenCL matrix multiplication ... Naive
        //--------------------------------------------------------------------------------


        // Load in kernel source, creating a program object for the context

        // the 3rd parameter specifis that the programm should be build
        // if build flags needs to be specified use program.build()
        cl::Program program(context, util::loadProgram(FileSystem::getPath("kernel/matMul.cl")), true);
       
        // create the kernel functor
        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer>naive_mmul(program, "mat_mul");
              

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
        d_a = cl::Buffer(context, h_A.begin(), h_A.end(), true);
        d_b = cl::Buffer(context, h_B.begin(), h_B.end(), true);

        d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * szC);
 
        std::cout << "\n===== OpenCL, matrix mult, C(i,j) per work item, order " << Ndim << " ======\n" << std::endl;

        // Do the multiplication COUNT times
        for (int i = 0; i < COUNT; i++)
        {
            // zero mat C
            initmat(Ndim, Ndim, h_C, 0.0f);

            // start timepoint
            auto start = std::chrono::high_resolution_clock::now();

            // entire range of C matrix elements
            cl::NDRange global(Ndim, Ndim);

            // RUN C = A*B
            naive_mmul(
                cl::EnqueueArgs(queue, global),
                Ndim,
                d_a,
                d_b,
                d_c);

            queue.finish();


            // end time stopping
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            
            float mflops = 2.0 * Ndim * Ndim * Ndim / (1000000.0f * duration.count() / 1000 / 1000);
            std::cout << "Time taken by execution " << duration.count() / 1000 << " milliseconds at " << mflops << " MFLOPS" << std::endl;

            // copy data back from device
            cl::copy(queue, d_c, h_C.begin(), h_C.end());

            // test the results
            int correct = 0;
            float errsq =0.0f;

            for (int i = 0; i < Ndim; i++) {
                for (int j = 0; j < Ndim; j++) {
                    // check if matrix mult was sucessfull
                    float err = h_C[i * Ndim + j] - (Ndim * AVAL * BVAL);
                    errsq += err * err;
                }
            }
            
            if (std::isnan(errsq) || errsq > TOL) {
                std::cout << "\nErrors in multiplication: " << errsq << std::endl;
            }

        }       

        //--------------------------------------------------------------------------------
        // OpenCL matrix multiplication ... C row per work item
        //--------------------------------------------------------------------------------

        std::cout << "\n===== OpenCL, matrix mult, C row per work item, order " << Ndim << " ======\n" << std::endl;

        // the 3rd parameter specifis that the programm should be build
        // if build flags needs to be specified use program.build()
        program = cl::Program(context, util::loadProgram(FileSystem::getPath("kernel/matMulRow.cl")), true);
       
        // create the kernel functor
        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer>crow_mmul(program, "mat_mul");

        // Do the multiplication COUNT times
        for (int i = 0; i < COUNT; i++)
        {
            // zero mat C
            initmat(Ndim, Ndim, h_C, 0.0f);
            // start timepoint
            auto start = std::chrono::high_resolution_clock::now();

            // entire range of C matrix elements
            cl::NDRange global(Ndim);

            // RUN C = A*B
            crow_mmul(
                cl::EnqueueArgs(queue, global),
                Ndim,
                d_a,
                d_b,
                d_c);

            queue.finish();


            // end time stopping
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

            float mflops = 2.0 * Ndim * Ndim * Ndim / (1000000.0f * duration.count() / 1000 / 1000);
            std::cout << "Time taken by execution " << duration.count() / 1000 << " milliseconds at " << mflops << " MFLOPS" << std::endl;

            // copy data back from device
            cl::copy(queue, d_c, h_C.begin(), h_C.end());

            // test the results
            int correct = 0;
            float errsq = 0.0f;

            for (int i = 0; i < Ndim; i++) {
                for (int j = 0; j < Ndim; j++) {
                    // check if matrix mult was sucessfull
                    float err = h_C[i * Ndim + j] - (Ndim * AVAL * BVAL);
                    errsq += err * err;
                }
            }

            if (std::isnan(errsq) || errsq > TOL) {
                std::cout << "\nErrors in multiplication: " << errsq << std::endl;
            }

        }

        //--------------------------------------------------------------------------------
        // OpenCL matrix multiplication ... C row per work item, A row in pivate memory
        //--------------------------------------------------------------------------------

        std::cout << "\n===== OpenCL, matrix mult, C row, A row in priv mem, order " << Ndim << " ======\n" << std::endl;

        // the 3rd parameter specifis that the programm should be build
        // if build flags needs to be specified use program.build()
        program = cl::Program(context, util::loadProgram(FileSystem::getPath("kernel/matMulRowPriv.cl")), true);

        // create the kernel functor
        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer>arowpriv_mmul(program, "mat_mul");

        // Do the multiplication COUNT times
        for (int i = 0; i < COUNT; i++)
        {
            // zero mat C
            initmat(Ndim, Ndim, h_C, 0.0f);
            // start timepoint
            auto start = std::chrono::high_resolution_clock::now();

            // entire range of C matrix elements
            cl::NDRange global(Ndim);
            cl::NDRange local(ORDER / 16);

            // RUN C = A*B
            arowpriv_mmul(
                cl::EnqueueArgs(queue, global, local),
                Ndim,
                d_a,
                d_b,
                d_c);

            queue.finish();


            // end time stopping
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

            float mflops = 2.0 * Ndim * Ndim * Ndim / (1000000.0f * duration.count() / 1000 / 1000);
            std::cout << "Time taken by execution " << duration.count() / 1000 << " milliseconds at " << mflops << " MFLOPS" << std::endl;

            // copy data back from device
            cl::copy(queue, d_c, h_C.begin(), h_C.end());
           
            // test the results
            int correct = 0;
            float errsq = 0.0f;

            for (int i = 0; i < Ndim; i++) {
                for (int j = 0; j < Ndim; j++) {
                    // check if matrix mult was sucessfull
                    float err = h_C[i * Ndim + j] - (Ndim * AVAL * BVAL);
                    errsq += err * err;
                }
            }

            if (std::isnan(errsq) || errsq > TOL) {
                std::cout << "\nErrors in multiplication: " << errsq << std::endl;
            }

        }

        //--------------------------------------------------------------------------------
        // OpenCL matrix multiplication ... C row per work item, A row pivate, B col local
        //--------------------------------------------------------------------------------

        std::cout << "\n===== OpenCL, mat mult, C row, priv A, B cols loc, order " << Ndim << " ======\n" << std::endl;

        // the 3rd parameter specifis that the programm should be build
        // if build flags needs to be specified use program.build()
        program = cl::Program(context, util::loadProgram(FileSystem::getPath("kernel/matMulRowPrivBloc.cl")), true);

        // create the kernel functor
        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg>browloc_mmul(program, "mat_mul");

        // Do the multiplication COUNT times
        for (int i = 0; i < COUNT; i++)
        {
            // zero mat C
            initmat(Ndim, Ndim, h_C, 0.0f);
            // start timepoint
            auto start = std::chrono::high_resolution_clock::now();

            // entire range of C matrix elements
            cl::NDRange global(Ndim);
            cl::NDRange local(ORDER / 16);
            // calc size of local memory in bytes
            cl::LocalSpaceArg localmem = cl::Local(sizeof(float) * Ndim);

            // RUN C = A*B
            browloc_mmul(
                cl::EnqueueArgs(queue, global, local),
                Ndim,
                d_a,
                d_b,
                d_c, localmem);

            queue.finish();


            // end time stopping
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

            float mflops = 2.0 * Ndim * Ndim * Ndim / (1000000.0f * duration.count() / 1000 / 1000);
            std::cout << "Time taken by execution " << duration.count() / 1000 << " milliseconds at " << mflops << " MFLOPS" << std::endl;

            // copy data back from device
            cl::copy(queue, d_c, h_C.begin(), h_C.end());

            // test the results
            int correct = 0;
            float errsq = 0.0f;

            for (int i = 0; i < Ndim; i++) {
                for (int j = 0; j < Ndim; j++) {
                    // check if matrix mult was sucessfull
                    float err = h_C[i * Ndim + j] - (Ndim * AVAL * BVAL);
                    errsq += err * err;
                }
            }

            if (std::isnan(errsq) || errsq > TOL) {
                std::cout << "\nErrors in multiplication: " << errsq << std::endl;
            }

        }
   
        //--------------------------------------------------------------------------------
        // OpenCL matrix multiplication ... blocked
        //--------------------------------------------------------------------------------

        std::cout << "\n===== Parallel matrix mult (blocked), order " << Ndim << " on device ======\n" << std::endl;

        // the 3rd parameter specifis that the programm should be build
        // if build flags needs to be specified use program.build()
        program = cl::Program(context, util::loadProgram(FileSystem::getPath("kernel/matMulBlocForm.cl")), true);
  
        // create the kernel functor
        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl::LocalSpaceArg>block_mmul(program, "mat_mul");

        // Do the multiplication COUNT times
        for (int i = 0; i < COUNT; i++)
        {
            // zero mat C
            initmat(Ndim, Ndim, h_C, 0.0f);
            // start timepoint
            auto start = std::chrono::high_resolution_clock::now();

            // Work-group computes a block of C.  This size is also set
            // in a #define inside the kernel function.  Note this blocksize
            // must evenly divide the matrix order
            int blocksize = 16;

            // calc size of local memory in bytes
            cl::LocalSpaceArg A_block = cl::Local(sizeof(float) * blocksize * blocksize);
            cl::LocalSpaceArg B_block = cl::Local(sizeof(float) * blocksize * blocksize);

            // entire range of C matrix elements
            cl::NDRange global(Ndim, Ndim);
            cl::NDRange local(blocksize, blocksize);

            // RUN C = A*B
            cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl::LocalSpaceArg>block_mmul(program, "mat_mul");
       
            block_mmul(
                cl::EnqueueArgs(queue, global, local),
                Ndim,
                d_a,
                d_b,
                d_c,
                A_block,
                B_block);

            queue.finish();


            // end time stopping
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

            float mflops = 2.0 * Ndim * Ndim * Ndim / (1000000.0f * duration.count() / 1000 / 1000);
            std::cout << "Time taken by execution " << duration.count() / 1000 << " milliseconds at " << mflops << " MFLOPS" << std::endl;

            // copy data back from device
            cl::copy(queue, d_c, h_C.begin(), h_C.end());

            // test the results
            int correct = 0;
            float errsq = 0.0f;

            for (int i = 0; i < Ndim; i++) {
                for (int j = 0; j < Ndim; j++) {
                    // check if matrix mult was sucessfull
                    float err = h_C[i * Ndim + j] - (Ndim * AVAL * BVAL);
                    errsq += err * err;
                }
            }

            if (std::isnan(errsq) || errsq > TOL) {
                std::cout << "\nErrors in multiplication: " << errsq << std::endl;
            }

        }
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