/**
 * Vadd_kernel
 *
 * Element wise addition of two vectors (c = a + b)
 *
 * In plain C code style
 *
 */

#include "CL/cl.hpp"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>

#include "config.h"
#include <time.h>

 //  define VERBOSE if you want to print info about work groups sizes
 //#define  VERBOSE 1

 //pick up device type from compiler command line or from
 //the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

// clock
clock_t t;

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c

// --------------------------------------------------------------------------------------
// kernel: vadd 
// Purpose: compute the elementwise sum c = a + b
// input: a and b float vectors of length count
// output: c float vector of length count holding the sum a + b

const char *KernelSource = "\n" \
"__kernel void vadd(                                                 \n" \
"   __global float* a,                                                  \n" \
"   __global float* b,                                                  \n" \
"   __global float* c,                                                  \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       c[i] = a[i] + b[i];                                             \n" \
"}                                                                      \n" \
"\n";

// --------------------------------------------------------------------------------------

void check_error(cl_int err, const char* operation, char* filename, int line)
{
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "Error during operation '%s', ", operation);
        fprintf(stderr, "in '%s' on line %d\n", filename, line);
        fprintf(stderr, "Error code was (%d)\n", err);
        exit(EXIT_FAILURE);
    }
}


int output_device_info(cl_device_id device_id)
{
    int err;                            // error code returned from OpenCL calls
    cl_device_type device_type;         // Parameter defining the type of the compute device
    cl_uint comp_units;                 // the max number of compute units on a device
    cl_char vendor_name[1024] = { 0 };    // string to hold vendor name for compute device
    cl_char device_name[1024] = { 0 };    // string to hold name of compute device
#ifdef VERBOSE
    cl_uint          max_work_itm_dims;
    size_t           max_wrkgrp_size;
    size_t* max_loc_size;
#endif


    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device name!\n");
        return EXIT_FAILURE;
    }
    printf(" \n Device is  %s ", device_name);

    err = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device type information!\n");
        return EXIT_FAILURE;
    }
    if (device_type == CL_DEVICE_TYPE_GPU)
        printf(" GPU from ");

    else if (device_type == CL_DEVICE_TYPE_CPU)
        printf("\n CPU from ");

    else
        printf("\n non  CPU or GPU processor from ");

    err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), &vendor_name, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device vendor name!\n");
        return EXIT_FAILURE;
    }
    printf(" %s ", vendor_name);

    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &comp_units, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device number of compute units !\n");
        return EXIT_FAILURE;
    }
    printf(" with a max of %d compute units \n", comp_units);

#ifdef VERBOSE
    //
    // Optionally print information about work group sizes
    //
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint),
        &max_work_itm_dims, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get device Info (CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)!\n",
            err_code(err));
        return EXIT_FAILURE;
    }

    max_loc_size = (size_t*)malloc(max_work_itm_dims * sizeof(size_t));
    if (max_loc_size == NULL) {
        printf(" malloc failed\n");
        return EXIT_FAILURE;
    }
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, max_work_itm_dims * sizeof(size_t),
        max_loc_size, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get device Info (CL_DEVICE_MAX_WORK_ITEM_SIZES)!\n", err_code(err));
        return EXIT_FAILURE;
    }
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
        &max_wrkgrp_size, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get device Info (CL_DEVICE_MAX_WORK_GROUP_SIZE)!\n", err_code(err));
        return EXIT_FAILURE;
    }
    printf("work group, work item information");
    printf("\n max loc dim ");
    for (int i = 0; i < max_work_itm_dims; i++)
        printf(" %d ", (int)(*(max_loc_size + i)));
    printf("\n");
    printf(" Max work group size = %d\n", (int)max_wrkgrp_size);
#endif

    return CL_SUCCESS;

}

#define checkError(E, S) check_error(E,S,__FILE__,__LINE__)

int main(void)
{
    // Print Programm Infos
    std::cout << "OpenCL Vadd_Kernel - Version " << 
        Vadd_Kernel_VERSION_MAJOR << "." << Vadd_Kernel_VERSION_MINOR << std::endl;

    // declare variables

    int err; // error code returned from OpenCL calls

    float* h_a = (float*)calloc(LENGTH, sizeof(float));       // a vector
    float* h_b = (float*)calloc(LENGTH, sizeof(float));       // b vector
    float* h_c = (float*)calloc(LENGTH, sizeof(float));       // c vector (a+b) returned from the compute device

    unsigned int correct;           // number of correct results

    size_t global;                  // global domain size

    cl_device_id     device_id;     // compute device id
    cl_context       context;       // compute context
    cl_command_queue commands;      // compute command queue
    cl_program       program;       // compute program
    cl_kernel        ko_vadd;       // compute kernel

    cl_mem d_a;                     // device memory used for the input  a vector
    cl_mem d_b;                     // device memory used for the input  b vector
    cl_mem d_c;                     // device memory used for the output c vector

    // fill vectors a and b with random float values
    int count = LENGTH;

    // create input vectors and assign values on the host
    for (int i = 0; i < count; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    try 
    {

        // 01. define platform and queues
        // ------------------------------

        // set up platform and gpu device
        cl_uint numPlatforms;

        // find number of platforms
        err = clGetPlatformIDs(0, NULL, &numPlatforms);
        checkError(err, "Finding platforms");
        if (numPlatforms == 0)
        {
            std::cout << "Found 0 platforms!\n" << std::endl;
            return EXIT_FAILURE;
        }

        // get all platforms
        // array with available platform ids
        
        cl_platform_id * Platform = (cl_platform_id*)malloc(sizeof(cl_platform_id*)*numPlatforms);

        err = clGetPlatformIDs(numPlatforms, Platform, NULL);
        checkError(err, "Getting platforms");

        // Secure a GPU
        for (int i = 0; i < numPlatforms; i++)
        {
            err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
            if (err == CL_SUCCESS)
                break;
        }

        if (device_id == NULL)
            checkError(err, "Finding a device");
       
        {        
            // 01.5. print device info
            // -----------------------
            err = output_device_info(device_id);       
            checkError(err, "Printing device output");
        }

        // create a compute context with a single device
        context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
        checkError(err, "Creating context");

        // create a command queue
        commands = clCreateCommandQueue(context, device_id, 0, &err);
        checkError(err, "Creating command queue");

        // 02. define OpenCL memory objects
        // --------------------------------

        // create the input(a, b) and output(c) arrays in device memory
        d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, &err);
        checkError(err, "Creating buffer d_a");

        d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, &err);
        checkError(err, "Creating buffer d_a");

        d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, &err);
        checkError(err, "Creating buffer d_a");

        // write a and b vectors from host into compute device memory
        err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) * count, h_a, 0, NULL, NULL);
        checkError(err, "Copying h_a to device at d_a");

        err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float) * count, h_b, 0, NULL, NULL);
        checkError(err, "Copying h_b to device at d_b");


        // 03. create programm
        // -------------------

        // Create the compute program from the source buffer
        program = clCreateProgramWithSource(context, 1, (const char**)&KernelSource, NULL, &err);
        checkError(err, "Creating program");

        // 04. build the program
        // ---------------------

        // compile the programm to create a "dynamic" library from which specific kernels can be pulled
        err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            size_t len;
            char buffer[2048];

            std::cout << "Error: Failed to build programm executable!\n" << err << std::endl;
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);

            std::cout << buffer << std::endl;
            return EXIT_FAILURE;
        }

        // 05. create and setup kernel
        // ---------------------------

        // create compute kernel from the programm
        ko_vadd = clCreateKernel(program, "vadd", &err);
        checkError(err, "Creating kernel");

        // Set the argument to our compute kernel
        err = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_a);
        err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_b);
        err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
        err |= clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &count);
        checkError(err, "Setting kernel arguments");

        // 06. execute the kernel
        // ----------------------

        // start time measure
        t = clock();

        // execute the kernel over the entire range of our 1d input data set
        // letting the OpenCL runtime choose the work-group sizes
        global = count; // set work item dimensions
        err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
        checkError(err, "Enqueueing kernel");

        // Wait for the commadns to complete before stopping the time
        err = clFinish(commands);
        checkError(err, "Waiting for kernel to finish");

        // end timing
        t = clock() - t;
        double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds 
        std::cout << "The kernel ran in " << time_taken << "seconds" << std::endl;


        // 07. read results on the host
        // ----------------------------

        // read back the results from the compute device
        // third parameter sets if it should Block or not block
        // in-order queue which assures the previous commands are completed before the read can begin
        err = clEnqueueReadBuffer(commands, d_c, CL_TRUE, 0, sizeof(float) * count, h_c, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            std::cout << "Error: Failed to read output array! " << err << std::endl;
        }

        // test the results 
        correct = 0;
        float tmp;

        for (int i = 0; i < count; i++) {
            tmp = h_a[i] + h_b[i];  // assign element i of a+b to tmp
            tmp -= h_c[i];          // compute deviation of expected and output reult
            if (tmp * tmp < TOL * TOL) {
                correct++;
            } else {
                std::cout << " tmp " << tmp << " h_a " << h_a[i] << " h_b " << h_b[i] << " h_c " << h_c[i] << std::endl;
            }
        }

        // manually print random sample
        //std::cout << h_a[0] << "+" << h_b[0] << "=" << h_c[0]<<std::endl;

        // summarise the results 
        std::cout << "C = A+B: " << correct << " out of " << count << " results were correct.\n" << std::endl;

        // cleanup then shutdown
        clReleaseMemObject(d_a);
        clReleaseMemObject(d_b);
        clReleaseMemObject(d_c);
        clReleaseProgram(program);
        clReleaseKernel(ko_vadd);
        clReleaseCommandQueue(commands);
        clReleaseContext(context);

        // free mem
        free(Platform);
        free(h_a);
        free(h_b);
        free(h_c);

    } catch (std::exception e) {
        // catch errors and print error data
        std::cout << "OpenCL Error:" << e.what() << " returned " << std::endl;
        std::cout << "Check cl.h for error codes." << std::endl;
        exit(-1);
    } 
    return 0;
}