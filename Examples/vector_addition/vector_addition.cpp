// Cale Woodward
// Greg Stitt
// University of Florida

// This SYCL program will create a parallel (vectorized) version of the following
// sequential code:
//
// for (int i=0; i < VECTOR_SIZE; i++) {
//   out[i] = in1[i] + in2[i];


#include <CL/sycl.hpp>

#define VECTOR_SIZE 1000

class vector_add;

int main(int argc, char* argv[]) { 
  
  std::cout << "Performing vector addition...\n"
	    << "Vector size: " << VECTOR_SIZE << std::endl;

  // Declare the input and output vectors.
  // The _h suffix is used to signifiy that these variables are stored on the host.
  std::vector<int> in1_h(VECTOR_SIZE);
  std::vector<int> in2_h(VECTOR_SIZE);
  std::vector<int> out_h(VECTOR_SIZE);

  // Use another vector simply to verify functionality.
  std::vector<int> correct_out(VECTOR_SIZE);

  // Initialize vectors.
  for (size_t i=0; i<VECTOR_SIZE; i++) {
    in1_h[i] = i;
    in2_h[i] = i;
    out_h[i] = 0;
    correct_out[i] = i + i;
  }

  // Select a device to run the code and create a queue for sending commands.
  // Here we use the default_selector, which chooses a "default" device. The
  // default is usually a GPU if the host has access to one.
  cl::sycl::queue deviceQueue(cl::sycl::default_selector{});

  // NOTE: this could also be done in separate steps:
  // cl::sycl::default_selector selector;
  // cl::sycl::queue deviceQueue(selector);

  // NOTE: This syntax is deprecated in 2023 SYCL, so you may get warnings.
  // The new way is the following:
  // cl::sycl::queue deviceQueue(cl::sycl::default_selector_v);

  {
    // Declare bufferes that handle transfers to/from the device(s).
    // The buffer class has 2 template parameters: type (int) and number of dimenstions (1)
    // Each buffer's constructor (the part in braces), takes a pointer to the host data
    // to attach to the buffer, and a "range", which is similar to an NDRAnge in OpenCL.
    // The range specifies the number of dimensions (<1>) and the number of elements in each
    // dimension, which in this case is the size of each vector.
    cl::sycl::buffer<int, 1> in1Buffer {in1_h.data(), cl::sycl::range<1>(in1_h.size()) };
    cl::sycl::buffer<int, 1> in2Buffer {in2_h.data(), cl::sycl::range<1>(in2_h.size()) };
    cl::sycl::buffer<int, 1> outBuffer {out_h.data(), cl::sycl::range<1>(in2_h.size()) };

    // Next, we tell the device what to do by sending it a function via the queue's
    // submit function. The submit method takes a single parameter, which is specified here
    // using a lambda (which is the common convention). The lambda function is passed a
    // single parameter called a "handler." In later examples, we will show how the handler
    // can be made implicit, but here we use it explicitly.
    deviceQueue.submit([&](cl::sycl::handler& queueHandler) {

	// ALL CODE IN THIS SCOPE WILL EXECUTE ON THE SELECTED DEVICE
	
	// To allow the device to access the buffers, we need to create "accessors".
	// Accessors can be read_only, write_only, or read_write. Here, we only use
	// read_only for the inputs, and write_only for the outputs.
	cl::sycl::accessor in1Accessor(in1Buffer, queueHandler, cl::sycl::read_only);
	cl::sycl::accessor in2Accessor(in2Buffer, queueHandler, cl::sycl::read_only);
	cl::sycl::accessor outAccessor(outBuffer, queueHandler, cl::sycl::write_only);

	// The following code "vectorizes" the original sequential loop.
	// The parallel_for has a template parameter specifying a kernel name, and two
	// normal parameters that specify the range and function to perform in parallel.
	// The template parameter can be optional in some situations, which are not
	// documented here.
	//
	// Like the NDRange in OpenCL, the range specifies the number and dimensionality
	// of work-items (threads). For this example, we use a single dimension of threads
	// with a total number of threads equal to the size of the vectors.
	// The final parameter is the function to execute in parallel, which is again
	// usually a lambda. The function here takes an "id" object, which allows each
	// individual thread/work-item to identify itself. Without this id, each thread
	// would not know what memory to access.       
	queueHandler.parallel_for<class vector_add>(cl::sycl::range<1> { in1_h.size() }, [=](cl::sycl::id<1> i) {
	    outAccessor[i] = in1Accessor[i] + in2Accessor[i];
	  });

      });

    // Before continuing with the host code, we have to wait until device finishes.
    // Otherwise, the results might not be completed.
    deviceQueue.wait();
  }
  
  std::cout << "Operation complete:\n"
	    << "[" << in1_h[0] << "] + [" << in2_h[0] << "] = [" << out_h[0] << "]\n"
	    << "[" << in1_h[1] << "] + [" << in2_h[1] << "] = [" << out_h[1] << "]\n"
	    << "...\n"
	    << "[" << in1_h[VECTOR_SIZE - 1] << "] + [" << in2_h[VECTOR_SIZE - 1] << "] = [" << out_h[VECTOR_SIZE - 1] << "]\n";

  if (out_h == correct_out) {
    std::cout << "SUCCESS!" << std::endl;
  }
  else {
    std::cout << "ERROR: Execution failed." << std::endl;
  }
  
  return 0;
}
