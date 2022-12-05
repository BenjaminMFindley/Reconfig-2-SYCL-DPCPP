#include <CL/sycl.hpp>

#define VECTOR_SIZE 1024

int main(int argc, char* argv[]) {

	std::cout << "Performing vector addition...\n"
		<< "Vector size: " << VECTOR_SIZE << std::endl;

	// define input and output vectors
	std::vector<int> in1Host(VECTOR_SIZE);
	std::vector<int> in2Host(VECTOR_SIZE);
	std::vector<int> outHost(VECTOR_SIZE);

	// load vectors
	for (size_t i = 0; i < VECTOR_SIZE; i++) {
		in1Host.at(i) = i;
		in2Host.at(i) = i;
		outHost.at(i) = 0;
	}
	// create the queue using default device
	cl::sycl::queue deviceQueue(cl::sycl::default_selector{});

	// allocate memory on the device with malloc_device by specifying type, size, and queue
	// malloc_device is used for explicit data management
	// malloc_shared and malloc host can be used for implicit data management 
	auto in1Device = cl::sycl::malloc_device<int>(VECTOR_SIZE, deviceQueue);
	auto in2Device = cl::sycl::malloc_device<int>(VECTOR_SIZE, deviceQueue);
	auto outDevice = cl::sycl::malloc_device<int>(VECTOR_SIZE, deviceQueue);
	
	// submit work to the queue by passing in a handler
	// the handler defines the interface to invoke kernels by submitting commands to a queue
	deviceQueue.submit([&](cl::sycl::handler& queueHandler) {
		// using the handler, call memcpy, passing in memory destination, source, and size
		queueHandler.memcpy(in1Device, in1Host.data(), VECTOR_SIZE * sizeof(int));
	});

	// copy second input vector
	deviceQueue.submit([&](cl::sycl::handler& queueHandler) {
		queueHandler.memcpy(in2Device, in2Host.data(), VECTOR_SIZE * sizeof(int));
	});

	// calls to queues and handlers may execute asynchropnously
	// tell the queue to wait for the previous tasks to complete before continuing
	deviceQueue.wait();

	// alternatively, we can capture queue information in an event, and use that event
	// to state dependecency in the queue via the handler
	auto evaluationEvent = deviceQueue.submit([&](cl::sycl::handler& queueHandler) {
		
		// perform computation using parallel_for, passing in range and operation
		queueHandler.parallel_for(cl::sycl::range<1> { in1Host.size() }, [=](cl::sycl::id<1> i) {
			outDevice[i] = in1Device[i] + in2Device[i];
		});
	});
	
	deviceQueue.submit([&](cl::sycl::handler& queueHandler) {
		// state the dependency explicitly in the kernel using event information
		queueHandler.depends_on(evaluationEvent);

		// copy data back to host
		queueHandler.memcpy(outHost.data(), outDevice, VECTOR_SIZE * sizeof(int));
	});

	// tell the queue to wait until all previous task have completed
	deviceQueue.wait();
	
	// free the memory we allocated by passing in the device memory and the queue
	cl::sycl::free(in1Device, deviceQueue);
	cl::sycl::free(in2Device, deviceQueue);
	cl::sycl::free(outDevice, deviceQueue);	

	std::cout << "Operation complete:\n"
		<< "[" << in1Host.at(0) << "] + [" << in2Host.at(0) << "] = [" << outHost.at(0) << "]\n"
		<< "[" << in1Host.at(1) << "] + [" << in2Host.at(1) << "] = [" << outHost.at(1) << "]\n"
		<< "...\n"
		<< "[" << in1Host.at(VECTOR_SIZE - 1) << "] + [" << in2Host.at(VECTOR_SIZE - 1) << "] = [" << outHost.at(VECTOR_SIZE - 1) << "]\n";

	return 0;
}