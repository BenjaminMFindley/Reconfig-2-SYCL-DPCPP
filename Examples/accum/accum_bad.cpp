// Greg Stitt
// University of Florida
//
// accum_bad.cpp
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

#include <CL/sycl.hpp>

const float ALLOWABLE_ERROR = 0.000001;
bool are_floats_equal(float a, float b, float abs_tol=ALLOWABLE_ERROR, float rel_tol=ALLOWABLE_ERROR) {

  float diff = fabs(a-b);
  return (diff <= abs_tol || diff <= rel_tol * fmax(fabs(a), fabs(b)));
}


class accum;

int main(int argc, char* argv[]) { 

  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " vector_size" << std::endl;
    return 1;
  }

  const unsigned VECTOR_SIZE = atoi(argv[1]);
  unsigned num_work_items = ceil(VECTOR_SIZE / 2.0);

  std::cout << "NUM_WORK_ITMES = " << num_work_items << std::endl;
  //for (int i=VECTOR_SIZE; i > 1; i = ceil(i/2.0))
  //  std::cout << i << std::endl;
  
  
  std::vector<float> x_h(VECTOR_SIZE);
  float correct_out = 0;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(0, 100);

  for (size_t i=0; i < VECTOR_SIZE; i++) {
    //    x_h[i] = dist(gen);
    x_h[i] = i;
    correct_out += x_h[i];
  }
  
  for (int size = VECTOR_SIZE; size > 1; size = ceil(size / 2.0)) {
    
    if (i*2+1 < size) 
      x_H[i] = x_d[2*i] + x_d[2*i + 1];
    else if (i*2+1 == size)
      x_d[i] = x_d[2*i];
  }

  
  try {
    cl::sycl::queue queue(cl::sycl::default_selector_v);
    
    cl::sycl::buffer<float, 1> x_buf {x_h.data(), cl::sycl::range<1>(VECTOR_SIZE) };
    
    queue.submit([&](cl::sycl::handler& handler) {

	cl::sycl::accessor x_d(x_buf, handler, cl::sycl::read_write);

	handler.parallel_for<class accum>(cl::sycl::range<1> { num_work_items }, [=](cl::sycl::id<1> i) {
	    
	    // In every iteration, the collection of work-items will reduce
	    // a "size"-element array to a "size/2"-element array by
	    // adding all the pairs in the original array. This process
	    // continues until there are only 2 (or 1) elements left.
	    for (int size = VECTOR_SIZE; size > 1; size = ceil(size / 2.0)) {

	      if (i*2+1 < size) 
		x_d[i] = x_d[2*i] + x_d[2*i + 1];
	      else if (i*2+1 == size)
		x_d[i] = x_d[2*i];
	    }
	  });
      });

    queue.wait();
  }
  catch (cl::sycl::exception& e) {
    std::cout << e.what() << std::endl;
    return 1;
  }

  if (!are_floats_equal(correct_out, x_h[0])) {
    std::cout << "ERROR: output was " << x_h[0] << " instead of " << correct_out << std::endl;
    return 1;
  }

  std::cout << "SUCCESS!" << std::endl;
  return 0;
}
