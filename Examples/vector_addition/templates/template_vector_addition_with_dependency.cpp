#include <CL/sycl.hpp>

#define VECTOR_SIZE 1000

int main(int argc, char* argv[]) {

	std::cout << "Performing vector addition...\n"
		<< "Vector size: " << VECTOR_SIZE << std::endl;

	// define input and output vectors
	std::vector<int> in1(VECTOR_SIZE);
	std::vector<int> in2(VECTOR_SIZE);
	std::vector<int> out(VECTOR_SIZE);
	
	// instead of loading vectors on the host, use SYCL to load input vectors on the device
	/*
	// load vectors
	for (size_t i = 0; i < VECTOR_SIZE; i++) {
		in1.at(i) = i;
		in2.at(i) = i;
		out.at(i) = 0;
	}
	*/

	// braces ensure that buffer retains control of data until SYCL operations are complete
	{
		// create the queue using default device

		// create a range object for the buffers / kernels

		// create buffers

		// submit work to the queue - create input

			// create accessors to the buffers

			// load vectors using parallel_for, passing in range and operation
		
		// submit work to the queue - compute output

			// create accessors to the buffers
			// dependency is determined at runtime by read-after-write on the buffers

			// perform computation using parallel_for, passing in range and operation

	} // buffer data released back to the vectors

	std::cout << "Operation complete:\n"
		<< "[" << in1.at(0) << "] + [" << in2.at(0) << "] = [" << out.at(0) << "]\n"
		<< "[" << in1.at(1) << "] + [" << in2.at(1) << "] = [" << out.at(1) << "]\n"
		<< "...\n"
		<< "[" << in1.at(VECTOR_SIZE - 1) << "] + [" << in2.at(VECTOR_SIZE - 1) << "] = [" << out.at(VECTOR_SIZE - 1) << "]\n";

	return 0;
}