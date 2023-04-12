#include <CL/sycl.hpp>

#define VECTOR_SIZE 1000

int main(int argc, char* argv[]) {

	std::cout << "Performing vector addition...\n"
		<< "Vector size: " << VECTOR_SIZE << std::endl;

	// define input and output vectors

	// load vectors

	// create the queue using default device

	// allocate memory on the device via the queue
	// malloc_device is used for explicit data movement

	// submit work to the queue

		// copy in1 data to the device
		
	// submit work to the queue

		// copy in2 data to the device

	// tell the queue to wait for the previous tasks to complete before continuing

	// alternatively, we can capture queue information in an event when submitting work to the queue
	
		// perform computation using parallel_for, passing in range and operation

	// submit work to the queue

		// state the dependency in the kernel using event

		// copy data back to host

	// tell queue to wait until copy is done
	
	// free allocated memory

	std::cout << "Operation complete:\n"
		<< "[" << in1Host.at(0) << "] + [" << in2Host.at(0) << "] = [" << outHost.at(0) << "]\n"
		<< "[" << in1Host.at(1) << "] + [" << in2Host.at(1) << "] = [" << outHost.at(1) << "]\n"
		<< "...\n"
		<< "[" << in1Host.at(VECTOR_SIZE - 1) << "] + [" << in2Host.at(VECTOR_SIZE - 1) << "] = [" << outHost.at(VECTOR_SIZE - 1) << "]\n";

	return 0;
}