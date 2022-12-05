#include <CL/sycl.hpp>

#define VECTOR_SIZE 1024

int main(int argc, char* argv[]) {

	std::cout << "Performing vector addition...\n"
		<< "Vector size: " << VECTOR_SIZE << std::endl;

	// define input and output vectors
	std::vector<int> in1(VECTOR_SIZE);
	std::vector<int> in2(VECTOR_SIZE);
	std::vector<int> out(VECTOR_SIZE);

	// load vectors
	for(size_t i=0; i<VECTOR_SIZE; i++){
		in1.at(i) = i;
		in2.at(i) = i;
		out.at(i) = 0;
	}

	// create the queue using default device

	// create buffers
	
	// submit work to the queue...
		
		// create accessors

		// perform computation using parallel_for, passing in range and operation

	std::cout << "Operation complete:\n"
		<< "[" << in1.at(0) << "] + [" << in2.at(0) << "] = [" << out.at(0) << "]\n"
		<< "[" << in1.at(1) << "] + [" << in2.at(1) << "] = [" << out.at(1) << "]\n"
		<< "...\n"
		<< "[" << in1.at(VECTOR_SIZE - 1) << "] + [" << in2.at(VECTOR_SIZE - 1) << "] = [" << out.at(VECTOR_SIZE - 1) << "]\n";

	return 0;
}