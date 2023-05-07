// Greg Stitt
// University of Florida
//
// accum_correct_super_slow1.cpp
//
// This SYCL program will create a parallel (vectorized) version of the following
// sequential code:
//
// int accum = 0;
// for (int i=0; i < VECTOR_SIZE; i++)
//   accum += x[i];
//
// The previous example had a bug that was caused by work-items overwriting
// the inputs to other work-items due to execution in an unexpected order.
// Unfortunately, there is no way to guarantee the order of execution of
// work-items, so instead we must transform the code so that work-items
// cannot overwrite inputs of other work-items.
//
// In this example, we accomplish this goal by including an output array so
// that work-items read from an input array and write to the output array.
//
// The end result is a correct implementation. However, it is very slow,
// which we improve in the next examples.
//
// When running the example on the DevCloud, the execution time of this example
// for 1000000000 (1 billion) inputs was 84.85s.

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

#include <CL/sycl.hpp>

class copy;

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
  size_t vector_size;
  vector_size = atoi(argv[1]);

  if (vector_size <= 0) {
    print_usage(argv[0]);    
    return 1;    
  }

  std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
  std::vector<int> x_h(vector_size);
  std::vector<int> y_h(vector_size);
  

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(-10, 10);

  for (size_t i=0; i < vector_size; i++) {
    x_h[i] = dist(gen);
    y_h[i] = 0;
  }
  
  try {

    cl::sycl::device device;
    cl::sycl::default_selector_v(device);
        
    cl::sycl::queue queue(cl::sycl::default_selector_v, [] (sycl::exception_list el) {
	for (auto ex : el) { std::rethrow_exception(ex); }
      } );

    // Create 
    int *x_shared = cl::sycl::malloc_shared<int>(vector_size, queue);
    for (size_t i=0; i < vector_size; i++)
      x_shared[i] = x_h[i];
    
    //int *x_usm_d = cl::sycl::malloc_device<int>(vector_size, queue);
    //    for (size_t i=0; i < vector_size; i++)
    //x_usm_d[i] = x_h[i];

    
    
    start_time = std::chrono::system_clock::now();

    cl::sycl::buffer<int, 1> x_buf {x_h.data(), cl::sycl::range<1>(vector_size) };
    cl::sycl::buffer<int, 1> y_buf {y_h.data(), cl::sycl::range<1>(vector_size) };

    
    queue.submit([&](cl::sycl::handler& handler) {
	  
	cl::sycl::accessor x_d(x_buf, handler, cl::sycl::read_only);
	cl::sycl::accessor y_d(y_buf, handler, cl::sycl::write_only);
	  
	handler.parallel_for<class copy>(cl::sycl::range<1> { vector_size }, [=](cl::sycl::id<1> i) {

	    y_d[i] = x_d[i];
	  });
      });
      
    queue.wait_and_throw();

    end_time = std::chrono::system_clock::now();
  }
  catch (cl::sycl::exception& e) {
    std::cout << e.what() << std::endl;
    return 1;
  }
  
  if (x_h != y_h) {
    std::cout << "ERROR: execution failed." << std::endl;
    return 1;
  }

  std::chrono::duration<double> seconds = end_time - start_time;
  std::cout << "SUCCESS! Time: " << seconds.count() << "s" << std::endl;
  return 0;
}
