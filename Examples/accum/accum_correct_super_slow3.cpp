// Greg Stitt
// University of Florida
//
// accum_correct_super_slow2.cpp
//
// This SYCL program will create a parallel (vectorized) version of the following
// sequential code:
//
// int accum = 0;
// for (int i=0; i < VECTOR_SIZE; i++) {
//   accum += a[i];
//

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

      cl::sycl::buffer<int, 1> x_buf {x_h.data(), cl::sycl::range<1>(size) };
      cl::sycl::buffer<int, 1> y_buf {y_h.data(), cl::sycl::range<1>(ceil(size/2.0)) };
      
      queue.submit([&](cl::sycl::handler& handler) {
	  
	  cl::sycl::accessor x_d(x_buf, handler, cl::sycl::read_only);
	  cl::sycl::accessor y_d(y_buf, handler, cl::sycl::write_only);
	  
	  handler.parallel_for<class accum>(cl::sycl::range<1> { num_work_items }, [=](cl::sycl::id<1> i) {
	      
	      if (2*i + 1 == size)
		y_d[i] = x_d[2*i];
	      else if (2*i + 1 < size)
		y_d[i] = x_d[2*i] + x_d[2*i+1];
	    });
	});
      
      queue.wait();      
      y_buf.get_access<cl::sycl::access::mode::read>();

      // CHANGES FROM PREVIOUS EXAMPLE
      // Previously we copied the entire output vector
      // back to the input vector. This was very wasteful
      // because most iterations have few valid outputs. In
      // this version we optimize the code to only copy back
      // the actual outputs.
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
