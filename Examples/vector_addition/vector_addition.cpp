#include <CL/sycl.hpp>

#define VECTOR_SIZE 1000

int main(int argc, char* argv[]) {

	std::cout << "Performing vector addition...\n"
		<< "Vector size: " << VECTOR_SIZE << std::endl;

	// define input and output vectors
	std::vector<int> in1(VECTOR_SIZE);
	std::vector<int> in2(VECTOR_SIZE);
	std::vector<int> out(VECTOR_SIZE);

	// load vectors
	for (size_t i=0; i<VECTOR_SIZE; i++) {
		in1.at(i) = i;
		in2.at(i) = i;
		out.at(i) = 0;
	}

	// create the queue using default device
	cl::sycl::queue deviceQueue(cl::sycl::default_selector{});

	// create buffers
	cl::sycl::buffer in1Buffer(in1);
	cl::sycl::buffer in2Buffer(in2);
	cl::sycl::buffer outBuffer(out);

	deviceQueue.submit([&](cl::sycl::handler& queueHandler) {

		// create accessors to the buffers
		cl::sycl::accessor in1Accessor(in1Buffer, queueHandler, cl::sycl::read_only);
		cl::sycl::accessor in2Accessor(in2Buffer, queueHandler, cl::sycl::read_only);
		cl::sycl::accessor outAccessor(outBuffer, queueHandler, cl::sycl::write_only);

		// perform computation using parallel_for, passing in range and operation
		queueHandler.parallel_for(cl::sycl::range<1> { in1.size() }, [=](cl::sycl::id<1> i) {
			outAccessor[i] = in1Accessor[i] + in2Accessor[i];
		});

	});
	
	// tell queue to wait until operations are finished before we read the results
	deviceQueue.wait();

	std::cout << "Operation complete:\n"
		<< "[" << in1.at(0) << "] + [" << in2.at(0) << "] = [" << out.at(0) << "]\n"
		<< "[" << in1.at(1) << "] + [" << in2.at(1) << "] = [" << out.at(1) << "]\n"
		<< "...\n"
		<< "[" << in1.at(VECTOR_SIZE - 1) << "] + [" << in2.at(VECTOR_SIZE - 1) << "] = [" << out.at(VECTOR_SIZE - 1) << "]\n";

	return 0;
}