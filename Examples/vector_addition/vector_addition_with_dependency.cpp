#include <CL/sycl.hpp>

#define VECTOR_SIZE 1024

int main(int argc, char* argv[]) {

	std::cout << "Performing vector addition...\n"
		<< "Vector size: " << VECTOR_SIZE << std::endl;

	// define input and output vectors
	std::vector<int> in1(VECTOR_SIZE);
	std::vector<int> in2(VECTOR_SIZE);
	std::vector<int> out(VECTOR_SIZE);

	// create the queue by passing in a device selector 
	// programs submit tasks to a device via the queue, and may also monitor the queue for device completion/errors
	// here, we are using the default device selector, which selects the most capable device at runtime
	// other generic selectors exist for CPU, GPU, FPGA, FPGA Emulator
	// custom selectors are also supported to choose devices based on brand/performance/etc.
	// queues cannot be shared between devices, but you can submit one or more queues to a device multiple times
	cl::sycl::queue deviceQueue(cl::sycl::default_selector{});

	// create buffers by passing in the vectors
	// buffers are not copies of the data, but references to memory locations
	// memory locations may not be sequential and thus cannot be accessed like a typical array
	// instead, we will later create accessors, which are the only way to read/write to buffers
	cl::sycl::buffer in1Buffer(in1);
	cl::sycl::buffer in2Buffer(in2);
	cl::sycl::buffer outBuffer(out);

	// buffers are blocking
	// putting SYCL work in braces is one way to ensure the buffers retain control of data until SYCL operations are complete
	{

		// submit work to the queue by passing in a handler
		// the handler defines the interface to invoke kernels by submitting commands to a queue
		deviceQueue.submit([&](cl::sycl::handler& queueHandler) {

			// create accessors by passing in the buffer, handler, and access mode
			// using an accurate access mode gives the runtime more freedom in parallel operations
			cl::sycl::accessor in1Accessor(in1Buffer, queueHandler, cl::sycl::write_only);
			cl::sycl::accessor in2Accessor(in2Buffer, queueHandler, cl::sycl::write_only);

			// using the handler, call the parallel_for kernel, passing in range and operation(s)
			// the first parameter, range, defines the number of work items
			// the second parameter is a lambda function that defines what we will be doing on each work item
			// here we use "id", which will give us the work item's global location
			queueHandler.parallel_for(cl::sycl::range<1> { in1.size() }, [=](cl::sycl::id<1> i) {
				in1Accessor[i] = i;
				in2Accessor[i] = i;
			});

		});

		// submit work to the queue
		deviceQueue.submit([&](cl::sycl::handler& queueHandler) {

			// create accessors by passing in the buffer, handler, and access mode
			// dependency is determined implicitly at runtime by read-after-write access
			cl::sycl::accessor in1Accessor(in1Buffer, queueHandler, cl::sycl::read_only);
			cl::sycl::accessor in2Accessor(in2Buffer, queueHandler, cl::sycl::read_only);
			cl::sycl::accessor outAccessor(outBuffer, queueHandler, cl::sycl::write_only);

			// perform computation using parallel_for, passing in range and operation
			queueHandler.parallel_for(cl::sycl::range<1> { in1.size() }, [=](cl::sycl::id<1> i) {
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