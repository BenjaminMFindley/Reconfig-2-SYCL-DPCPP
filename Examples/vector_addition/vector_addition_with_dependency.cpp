#include <CL/sycl.hpp>

#define VECTOR_SIZE 1000

int main(int argc, char* argv[]) {

	std::cout << "Performing vector addition...\n"
		<< "Vector size: " << VECTOR_SIZE << std::endl;

	// define input and output vectors
	std::vector<int> in1(VECTOR_SIZE);
	std::vector<int> in2(VECTOR_SIZE);
	std::vector<int> out(VECTOR_SIZE);
	
	// instead of loading vectors here, we will now load them in a kernel
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
		cl::sycl::queue deviceQueue(cl::sycl::default_selector{});

		// create a range object for the buffers
		cl::sycl::range<1> itemRange{ in1.size() };

		// create buffers
		cl::sycl::buffer<int, 1> in1Buffer(in1.data(), itemRange);
		cl::sycl::buffer<int, 1> in2Buffer(in2.data(), itemRange);
		cl::sycl::buffer<int, 1> outBuffer(out.data(), itemRange);

		deviceQueue.submit([&](cl::sycl::handler& queueHandler) {

			// create accessors to the buffers
			// The property no_init lets the runtime know that the
			// previous contents of the buffer can be discarded.
			cl::sycl::accessor in1Accessor(in1Buffer, queueHandler, cl::sycl::write_only, cl::sycl::no_init);
			cl::sycl::accessor in2Accessor(in2Buffer, queueHandler, cl::sycl::write_only, cl::sycl::no_init);

			// load vectors using parallel_for, passing in range and operation
			queueHandler.parallel_for(itemRange, [=](cl::sycl::id<1> i) {
				in1Accessor[i] = i;
				in2Accessor[i] = i;
			});

		});
		
		deviceQueue.submit([&](cl::sycl::handler& queueHandler) {

			// create accessors to the buffers
			// dependency is determined at runtime by read-after-write on the buffers
			cl::sycl::accessor in1Accessor(in1Buffer, queueHandler, cl::sycl::read_only);
			cl::sycl::accessor in2Accessor(in2Buffer, queueHandler, cl::sycl::read_only);
			cl::sycl::accessor outAccessor(outBuffer, queueHandler, cl::sycl::write_only, cl::sycl::no_init);

			// perform computation using parallel_for, passing in range and operation
			queueHandler.parallel_for(itemRange, [=](cl::sycl::id<1> i) {
				outAccessor[i] = in1Accessor[i] + in2Accessor[i];
			});

		});
	} // buffer data released back to the vectors

	std::cout << "Operation complete:\n"
		<< "[" << in1.at(0) << "] + [" << in2.at(0) << "] = [" << out.at(0) << "]\n"
		<< "[" << in1.at(1) << "] + [" << in2.at(1) << "] = [" << out.at(1) << "]\n"
		<< "...\n"
		<< "[" << in1.at(VECTOR_SIZE - 1) << "] + [" << in2.at(VECTOR_SIZE - 1) << "] = [" << out.at(VECTOR_SIZE - 1) << "]\n";

	return 0;
}