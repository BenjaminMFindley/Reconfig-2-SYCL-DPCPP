// Cale Woodward
// Greg Stitt
// University of Florida
//
// vector_add.cpp
//
// This SYCL program will create a parallel (vectorized) version of the following
// sequential code:
//
// for (int i=0; i < VECTOR_SIZE; i++)
//   out[i] = in1[i] + in2[i];
//
// In this modified version of the code, we add exception handling to catch
// "synchronous" errors, which are errors that occur on the host. Catching errors
// that occur on the device will be explained in a later example.

#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

const int VECTOR_SIZE = 10;

class vector_add;

int main(int argc, char* argv[]) { 
  
  std::cout << "Performing vector addition...\n"
	    << "Vector size: " << VECTOR_SIZE << std::endl;

  /*std::array<std::array<int, VECTOR_SIZE>, VECTOR_SIZE> in1_h;
  std::array<std::array<int, VECTOR_SIZE>, VECTOR_SIZE> in2_h;
  std::array<std::array<int, VECTOR_SIZE>, VECTOR_SIZE> out_h;

  std::array<std::array<int, VECTOR_SIZE>, VECTOR_SIZE> correct_out;*/

  int in1_h[VECTOR_SIZE][VECTOR_SIZE];
  int in2_h[VECTOR_SIZE][VECTOR_SIZE];
  int out_h[VECTOR_SIZE][VECTOR_SIZE];
  int correct_out[VECTOR_SIZE][VECTOR_SIZE];
  
  for (size_t i=0; i < VECTOR_SIZE; i++) {
    for (size_t j=0; j < VECTOR_SIZE; j++) {
      in1_h[i][j] = i;
      in2_h[i][j] = j;
      out_h[i][j] = 0;
      correct_out[i][j] = i + j;
    }
  }

  try {
    cl::sycl::queue queue(cl::sycl::default_selector_v);

    cl::sycl::buffer<int, 2> in1_buf((int*) in1_h, cl::sycl::range<2> {VECTOR_SIZE, VECTOR_SIZE} );
    cl::sycl::buffer<int, 2> in2_buf((int*) in2_h, cl::sycl::range<2> {VECTOR_SIZE, VECTOR_SIZE} );
    cl::sycl::buffer<int, 2> out_buf((int*) out_h, cl::sycl::range<2> {VECTOR_SIZE, VECTOR_SIZE} );
    
    queue.submit([&](cl::sycl::handler& handler) {

	cl::sycl::accessor in1_d(in1_buf, handler, cl::sycl::read_only);
	cl::sycl::accessor in2_d(in2_buf, handler, cl::sycl::read_only);
	cl::sycl::accessor out_d(out_buf, handler, cl::sycl::write_only);

	handler.parallel_for<class vector_add>(cl::sycl::range<2> { VECTOR_SIZE, VECTOR_SIZE }, [=](cl::sycl::id<2> id) {
	    size_t x = id[0];
	    size_t y = id[1];
	    out_d[x][y] = in1_d[x][y] + in2_d[x][y];
	  });

      });

    queue.wait();
  }
  catch (cl::sycl::exception& e) {
    std::cout << e.what() << std::endl;
    return 1;
  }
  
  /*  std::cout << "Operation complete:\n"
	    << "[" << in1_h[0] << "] + [" << in2_h[0] << "] = [" << out_h[0] << "]\n"
	    << "[" << in1_h[1] << "] + [" << in2_h[1] << "] = [" << out_h[1] << "]\n"
	    << "...\n"
	    << "[" << in1_h[VECTOR_SIZE - 1] << "] + [" << in2_h[VECTOR_SIZE - 1] << "] = [" << out_h[VECTOR_SIZE - 1] << "]\n"
	    << std::endl;
  */

  for (size_t i=0; i < VECTOR_SIZE; i++) {
    for (size_t j=0; j < VECTOR_SIZE; j++) {
      std::cout << out_h[i][j] << " ";
    }
    std::cout << std::endl;
  }

  for (size_t i=0; i < VECTOR_SIZE; i++) {
    for (size_t j=0; j < VECTOR_SIZE; j++) {
      if (out_h[i][j] != correct_out[i][j]) {
	std::cout << "ERROR: Execution failed." << std::endl;
	return 1;
      }
    }
  }
  
  std::cout << "SUCCESS!" << std::endl;
    
  return 0;
}
