#include <CL/sycl.hpp>
#include <chrono>

#define VECTOR_SIZE 10000

int main(int argc, char* argv[]) {

	std::cout << "Performing vector addition...\n"
		<< "Vector size: " << VECTOR_SIZE << std::endl;

	// define input/output/validation vectors
	std::vector<int> in1(VECTOR_SIZE);
	std::vector<int> in2(VECTOR_SIZE);
	std::vector<int> out(VECTOR_SIZE);
	std::vector<int> val(VECTOR_SIZE);

	// load vectors
	for (size_t i=0; i<VECTOR_SIZE; i++) {
		in1.at(i) = i;
		in2.at(i) = i;
		out.at(i) = 0;
		val.at(i) = 0;
	}

	// begin host timing

	// host validation

	// end host timing

	// report host timing

	// begin device timing

	// braces ensure all SYCL work completes before giving up access to buffer data

		// select offload device using device selector

		// create the queue with device selector

		// display device info

		// create a range object for the buffers

		// create buffers

		// submit work to the queue

			// create accessors for the input/output buffers

			// perform operation using parallel_for
			// 1st param: number of work items
			// 2nd param: kernel to specify what to do per work item

		// wait for computations to complete

	// data control returned from buffer

	// end device timing

	// report device timing

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