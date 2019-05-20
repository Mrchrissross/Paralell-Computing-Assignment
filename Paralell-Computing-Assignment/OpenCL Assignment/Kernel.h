#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <CL/cl.hpp>
#include "Utils.h"

typedef float mytype;

class Kernel
{
public:

	/// Initialises the kernel
	Kernel::Kernel(int platform_id, int device_id, const char* kernel_location, const char* _command, bool _scratch)
	{
		//Part 2 - host operations
		context = GetContext(platform_id, device_id); // Select computing devices
		cl::CommandQueue _queue(context, CL_QUEUE_PROFILING_ENABLE); // Create a queue to which we will push commands for the device
		queue = _queue;

		//2.2 Load & build the device code
		cl::Program::Sources sources;
		AddSources(sources, kernel_location);
		cl::Program _program(context, sources);
		program = _program;
		program.build(); // Build and debug the kernel code

		command = _command;
		scratch = _scratch;
	}

	///Runs Kernel Command
	vector<float> RunKernel(vector<float> A, float extra = 0)
	{
		//Part 3 - memory allocation
		//host - input
		try
		{
			// The following part adjusts the length of the input vector so it can be run for a specific workgroup size, 
			// if the total input length is divisible by the workgroup size this makes the code more efficient.
			size_t local_size = 128;
			size_t padding_size = A.size() % local_size;

			// If the input vector is not a multiple of the local_size, insert additional neutral elements (0 for addition) so that the total will not be affected
			if (padding_size)
			{
				//create an extra vector with neutral values
				std::vector<float> A_ext(local_size - padding_size, 0);
				//append that extra vector to our input
				A.insert(A.end(), A_ext.begin(), A_ext.end());
			}

			size_t input_elements = A.size(); //number of input elements
			size_t input_size = A.size() * sizeof(mytype);

			//host - output
			std::vector<float> B(input_elements);

			//device - buffers
			cl::Buffer buffer_1(context, CL_MEM_READ_ONLY, input_size);
			cl::Buffer buffer_2(context, CL_MEM_READ_WRITE, input_size);

			//Part 4 - device operations

			cl::Event prof_event;

			//4.1 copy array A to and initialise other arrays on device memory
			queue.enqueueWriteBuffer(buffer_1, CL_TRUE, 0, input_size, &A[0], NULL, &prof_event);
			queue.enqueueReadBuffer(buffer_2, CL_TRUE, 0, input_size, &B[0]); //zero B buffer on device memory

			//4.2 Setup and execute all kernels (i.e. device code)
			cl::Kernel kernel = cl::Kernel(program, command);
			kernel.setArg(0, buffer_1);
			kernel.setArg(1, buffer_2);

			//Add scratch based on whether the scratch bool is true of false.
			if(scratch)
				kernel.setArg(2, cl::Local(local_size * sizeof(mytype)));

			if(extra != 0)
				kernel.setArg(2, extra);

			//call all kernels in a sequence
			queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);

			//4.3 Copy the result from device to host
			queue.enqueueReadBuffer(buffer_2, CL_TRUE, 0, input_size, &B[0]);

			////Display the kernel execution time at the end of the program
			//std::cout <<
			//"Kernel execution time [ns]:"	<<  prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
			//<< std::endl;

			return B;
		}
		catch (cl::Error err)
		{
			std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
		}
	}
private:
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program program;

	const char* command;
	bool scratch;
};