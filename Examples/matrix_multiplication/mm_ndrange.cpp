#include <CL/sycl.hpp>
using namespace sycl;

void mm_ndrange_kernel(queue& deviceQueue, std::vector<double>& in1, std::vector<double>& in2, std::vector<double>& out, size_t N, size_t B) {
    std::cout << "Executing matrix multiplication ND-range kernel...\n\n";

    // create 2-D SYCL range item for the number of buffer items and work group items
    range<2> numItems{ N,N };
    range<2> workGroup{ B,B };

    // create buffers which are used to pass data between host and device
    // input data is 1-D, but here I cast to 2-D buffers for easier indexing
    buffer<double, 2> in1Buffer(in1.data(), numItems);
    buffer<double, 2> in2Buffer(in2.data(), numItems);
    buffer<double, 2> outBuffer(out.data(), numItems);

    // submit work to the queue and capture details in event
    auto queueEvent = deviceQueue.submit([&](handler& queueHandler) {

    // create accessors for device to read/write data in buffers
    auto in1Accessor = in1Buffer.get_access<access::mode::read>(queueHandler);
    auto in2Accessor = in2Buffer.get_access<access::mode::read>(queueHandler);
    auto outAccessor = outBuffer.get_access<access::mode::write>(queueHandler);

    // perform operation using parallel_for with basic_kernel
    // basic kernel is useful for "embarassing parallelism"
    // 1st param: num work items, here we are using the range item created above
    // 2nd param: kernel to specify what to do per work item, here we are 
    // addressing the work items with the basic 2-D index 
    queueHandler.parallel_for(nd_range{ numItems,workGroup }, [=](nd_item<2> item) {
        // first, we get the row and column index for the current work item
        auto rowIndex = item.get_global_id(0);
        auto colIndex = item.get_global_id(1);
        // calculate work item data by iterating through in1's rows and in2's columns
        for (int i = 0; i < N; i++) {
            // since index is 2-D, we can also pass it directly into the slices
            outAccessor[rowIndex][colIndex] += in1Accessor[rowIndex][i] * in2Accessor[i][colIndex];
        }
        });
    });

    // allow read access on output buffer
    outBuffer.get_access<access::mode::read>();

    // wait to get profile results until the queue is done executing on the kernel
    deviceQueue.wait();

    // get reported times from kernel event profile
    auto kernel_end = queueEvent.get_profiling_info<info::event_profiling::command_end>();
    auto kernel_start = queueEvent.get_profiling_info<info::event_profiling::command_start>();
    auto kernel_duration = round((kernel_end - kernel_start) / 1.0e6);

    std::cout << "Kernel execution time : " << kernel_duration << " milliseconds\n";
}