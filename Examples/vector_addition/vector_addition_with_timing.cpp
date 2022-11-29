#include <CL/sycl.hpp>
#include <chrono>

#define VECTOR_SIZE 10000

int main(int argc, char* argv[]) {

	std::cout << "Performing vector addition...\n"
		<< "Vector size: " << VECTOR_SIZE << std::endl;

	// define input and output vectors
	std::vector<int> in1(VECTOR_SIZE);
	std::vector<int> in2(VECTOR_SIZE);
	std::vector<int> out(VECTOR_SIZE);
	std::vector<int> val(VECTOR_SIZE);

	// load input vectors
	for (size_t i=0; i<VECTOR_SIZE; i++) {
		in1.at(i) = i;
		in2.at(i) = i;
		out.at(i) = 0;
		val.at(i) = 0;
	}

	// begin host timing
	auto hostStart = std::chrono::high_resolution_clock::now();

	// host validation
	for (size_t i=0; i<val.size(); i++) {
		val.at(i) = in1.at(i) + in2.at(i);
	}

	// end host timing
	auto hostStop = std::chrono::high_resolution_clock::now();
	auto hostDuration = std::chrono::duration_cast<std::chrono::microseconds>(hostStop - hostStart);

	// report host timing
	std::cout << "Sequential compute time without SYCL: " << hostDuration.count() << " us\n";

	auto deviceStart = std::chrono::high_resolution_clock::now();

	// braces ensure all SYCL work completes before giving up access to buffer data
	{
		// select offload device
		cl::sycl::cpu_selector deviceSelector;
		//cl::sycl::gpu_selector deviceSelector;

		// create the queue
		cl::sycl::queue deviceQueue(deviceSelector);

		// display device info
		std::cout << "Running on "
			<< deviceQueue.get_device().get_info<cl::sycl::info::device::name>()
			<< "\n";

		// create a range object for the buffers
		cl::sycl::range<1> itemRange{ in1.size() };

		// create buffers
		cl::sycl::buffer<int, 1> in1Buffer(in1.data(), itemRange);
		cl::sycl::buffer<int, 1> in2Buffer(in2.data(), itemRange);
		cl::sycl::buffer<int, 1> outBuffer(out.data(), itemRange);

		deviceQueue.submit([&](cl::sycl::handler& queueHandler) {

			// create accessors for the input/output buffers
			cl::sycl::accessor in1Accessor(in1Buffer, queueHandler, cl::sycl::read_only);
			cl::sycl::accessor in2Accessor(in2Buffer, queueHandler, cl::sycl::read_only);
			cl::sycl::accessor outAccessor(outBuffer, queueHandler, cl::sycl::write_only);

			// perform operation using parallel_for
			// 1st param: number of work items
			// 2nd param: kernel to specify what to do per work item
			queueHandler.parallel_for(itemRange, [=](cl::sycl::id<1> i) {
				outAccessor[i] = in1Accessor[i] + in2Accessor[i];
			});

		});

		// wait for computations to complete
		deviceQueue.wait();
	}

	auto deviceStop = std::chrono::high_resolution_clock::now();
	auto deviceDuration = std::chrono::duration_cast<std::chrono::microseconds>(deviceStop - deviceStart);

	// report SYCL timing
	std::cout << "Total SYCL offload time: " << deviceDuration.count() << " us\n";

	// validate
	for (size_t i = 0; i < val.size(); i++){
		if (out.at(i) != val.at(i)) {
			std::cout << "Incorrect device values.\n"
				<< out.at(i) << " != " << val.at(i) << "\n";
			return -1;
		}
	}

	std::cout << "Host and device values match\n";

	return 0;
}