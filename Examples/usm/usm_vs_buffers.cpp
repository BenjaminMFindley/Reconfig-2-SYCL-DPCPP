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

class copy_buffer;
class copy_usm_shared;
class copy_usm_device;
class copy_usm_host;

void print_usage(const std::string& name) {
  std::cout << "Usage: " << name << " vector_size (must be positive)" << std::endl;      
}


void copy_buffer(cl::sycl::queue &queue, const std::vector<int> &x_h, std::vector<int> &y_h) {

  if (x_h.size() != y_h.size()) {
    throw std::runtime_error("Vectors have different sizes");
  }
  
  cl::sycl::buffer<int, 1> x_buf {x_h.data(), cl::sycl::range<1>(x_h.size()) };
  cl::sycl::buffer<int, 1> y_buf {y_h.data(), cl::sycl::range<1>(y_h.size()) };

  queue.submit([&](cl::sycl::handler& handler) {

      cl::sycl::accessor x_d(x_buf, handler, cl::sycl::read_only);
      cl::sycl::accessor y_d(y_buf, handler, cl::sycl::write_only);

      handler.parallel_for<class copy_buffer>(cl::sycl::range<1> { x_h.size() }, [=](cl::sycl::id<1> i) {

	  y_d[i] = x_d[i];
	});
    });

  queue.wait_and_throw();  
}


void copy_usm_shared(cl::sycl::queue &queue, const int *x_d, int *y_d, size_t vector_size) {

  queue.submit([&](cl::sycl::handler& handler) {

      handler.parallel_for<class copy_usm_shared>(cl::sycl::range<1> { vector_size }, [=](cl::sycl::id<1> i) {

	  y_d[i] = x_d[i];
	});
    });

  queue.wait_and_throw();  
}


void copy_usm_device(cl::sycl::queue &queue, const std::vector<int> &x_h, std::vector<int> &y_h) {

  if (x_h.size() != y_h.size()) {
    throw std::runtime_error("Vectors have different sizes");
  }
  
  int *x_d = cl::sycl::malloc_device<int>(x_h.size(), queue);
  int *y_d = cl::sycl::malloc_device<int>(y_h.size(), queue);

  queue.memcpy(x_d, x_h.data(), sizeof(int) * x_h.size());
  
  queue.submit([&](cl::sycl::handler& handler) {

      handler.parallel_for<class copy_usm_device>(cl::sycl::range<1> { x_h.size() }, [=](cl::sycl::id<1> i) {

	  y_d[i] = x_d[i];
	});
    });

  queue.memcpy(y_h.data(), y_d, sizeof(int) * y_h.size());
  queue.wait_and_throw();


  cl::sycl::free(x_d, queue);
  cl::sycl::free(y_d, queue);
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

    // Create memory for USM shared allocation, where memory is accesible on host
    // and device, and transfers are implicit.
    int *x_shared = cl::sycl::malloc_shared<int>(vector_size, queue);
    int *y_shared = cl::sycl::malloc_shared<int>(vector_size, queue);
    for (size_t i=0; i < vector_size; i++)
      x_shared[i] = x_h[i];      
    
    start_time = std::chrono::system_clock::now();
    //copy_buffer(queue, x_h, y_h);   
    end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> buffer_time = end_time - start_time;
    if (x_h != y_h) {
      std::cout << "ERROR: buffer execution failed." << std::endl;
      //  return 1;
    }
    
    start_time = std::chrono::system_clock::now();
    copy_usm_shared(queue, x_shared, y_shared, vector_size);    
    end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> shared_time = end_time - start_time;
    if (memcmp(x_shared, y_shared, sizeof(int) * vector_size)) {
      std::cout << "ERROR: USM malloc_shared execution failed." << std::endl;
      return 1;
    }

   
    std::fill(y_h.begin(), y_h.end(), 0);
    
    start_time = std::chrono::system_clock::now();
    copy_usm_device(queue, x_h, y_h);
    end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> device_time = end_time - start_time;
    if (x_h != y_h) {
      std::cout << "ERROR: USM malloc_device execution failed." << std::endl;
      return 1;
    }
    
    std::cout << "SUCCESS!" << std::endl
	      << "Buffers: " << buffer_time.count() << "s" << std::endl
	      << "USM malloc_shared: " << shared_time.count() << "s" << std::endl
	      << "USM malloc_device: " << device_time.count() << "s" << std::endl;
  }
  catch (cl::sycl::exception& e) {
    std::cout << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}
