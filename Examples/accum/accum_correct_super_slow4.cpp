// Greg Stitt
// University of Florida
//
// accum_correct_super_slow.cpp
//
// This SYCL program will create a parallel (vectorized) version of the following
// sequential code:
//
// int accum = 0;
// for (int i=0; i < VECTOR_SIZE; i++) {
//   accum += a[i];
//
// The previous example had a bug that was caused by work-items overwriting
// the inputs to other work-items due to work-items executing in an
// unexpected order. Unfortunately, there is no way to guarantee the
// order of execution of work-items, so instead we must transform the code
// so that work-items cannot overwrite inputs of other work-items.
//
// In this example, we accomplish this goal by adding an output array so
// that work-items read from an input array and write to the output array.
//
// The end result is a correct implementation. However, it is very slow,
// which we improve in the next examples.

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

#include <CL/sycl.hpp>

class accum;

void print_usage(const std::string& name) {
  std::cout << "Usage: " << name << " vector_size (must be positive)" << std::endl;      
}


int main(int argc, char* argv[]) { 

  // Check correct usage of command line.
  if (argc != 2) {
    print_usage(argv[0]);
    return 1;    
  }

  // Get vector size from command line.
  int vector_size;
  vector_size = atoi(argv[1]);

  if (vector_size <= 0) {
    print_usage(argv[0]);    
    return 1;    
  }

  std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
  unsigned num_work_items = ceil(vector_size / 2.0);
  std::vector<int> x_h(vector_size);
  std::vector<int> y_h(vector_size);
  int correct_out = 0;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(-10, 10);

  for (size_t i=0; i < vector_size; i++) {
    x_h[i] = dist(gen);
    correct_out += x_h[i];
  }
  
  try {
    cl::sycl::queue queue(cl::sycl::default_selector_v, [] (sycl::exception_list el) {
	for (auto ex : el) { std::rethrow_exception(ex); }
      } );

    start_time = std::chrono::system_clock::now();

    for (int size = vector_size; size > 1; size = ceil(size / 2.0)) {

      // CHANGES FROM PREVIOUS EXAMPLE
      // This syntax looks a little funny, but we use a separate scope to
      // ensure that the y_d[] outputs are copied back to y_h[] in between
      // iterations.
      {
	// CHANGES FROM PREVIOUS EXAMPLE
	// Here we create two buffers: one for the input and one for the output.
	cl::sycl::buffer<int, 1> x_buf {x_h.data(), cl::sycl::range<1>(size) };
	cl::sycl::buffer<int, 1> y_buf {y_h.data(), cl::sycl::range<1>(size) };
      
	queue.submit([&](cl::sycl::handler& handler) {

	    // CHANGES FROM PREVIOUS EXAMPLE
	    // The input buffer is now read_only instead of read_write.
	    // Similarly, the output buffer is write_only.
	    cl::sycl::accessor x_d(x_buf, handler, cl::sycl::read_only);
	    cl::sycl::accessor y_d(y_buf, handler, cl::sycl::write_only);
	  
	    handler.parallel_for<class accum>(cl::sycl::range<1> { num_work_items }, [=](cl::sycl::id<1> i) {

		// CHANGES FROM PREVIOUS EXAMPLE
		// In the previous example, we used a single shared vector
		// for both input and output. The risk in that startegy
		// is that without some guarantee about execution order
		// of work-items, there is no way to prevent race
		// conditions where work-time n overwrites the inputs of
		// any work-item n-1. Since there is no way to prevent
		// these race conditions, we must instead ensure that
		// no work-item can overwrite the inputs of another
		// work-item. To do that, we use a second array for outputs.
		// The only change from before is that this code now assigns
		// y_d[] instead of x_d[].
		if (2*i + 1 == size)
		  y_d[i] = x_d[2*i];
		else if (2*i + 1 < size)
		  y_d[i] = x_d[2*i] + x_d[2*i+1];
	      });
	  });
      
	queue.wait();
      }
      
      // CHANGES FROM PREVIOUS EXAMPLE
      // One disadvantage of this approach is that it requires us to
      // copy the output vector back to the input for every iteration.
      // We fortunately don't have to copy the entire array, just the
      // set of outputs from the previous iteration.
      for (int i=0; i < ceil(size/2.0); i++) {
	x_h[i] = y_h[i];
      }
    }
    
    end_time = std::chrono::system_clock::now();
  }
  catch (cl::sycl::exception& e) {
    std::cout << e.what() << std::endl;
    return 1;
  }

  if (correct_out != x_h[0]) {
    std::cout << "ERROR: output was " << x_h[0] << " instead of " << correct_out << std::endl;
    return 1;
  }

  std::chrono::duration<double> seconds = end_time - start_time;
  std::cout << "SUCCESS! Time: " << seconds.count() << "s" << std::endl;
  return 0;
}
