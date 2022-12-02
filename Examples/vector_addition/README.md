# Vector Addition

These programs provide a basic introduction to SYCL and DPC++. 

* [vector_addition.cpp](https://github.com/BenjaminMFindley/Reconfig-2-SYCL-DPCPP/blob/main/Examples/vector_addition/vector_addition.cpp)

  * Simple SYCL program to showcase data management using queues, buffers, accessors, and kernels
  * Use as a "hello world" to test DPC++ installation
  
* [vector_addition_with_dependency.cpp](https://github.com/BenjaminMFindley/Reconfig-2-SYCL-DPCPP/blob/main/Examples/vector_addition/vector_addition_with_dependency.cpp)
  * Introduces data dependency using implicit read-after-write buffer access
  * The buffer model is generally preferred for new SYCL programs

* [vector_addition_usm.cpp](https://github.com/BenjaminMFindley/Reconfig-2-SYCL-DPCPP/blob/main/Examples/vector_addition/vector_addition_usm.cpp)
  * Unified Shared Memory (USM) is an alternative to buffers/accessors which allows for explicit data movement between host and device
  * USM is useful for porting C++ code that was already written to use pointers (i.e. malloc/new)

* [vector_addition_with_timing.cpp](https://github.com/BenjaminMFindley/Reconfig-2-SYCL-DPCPP/blob/main/Examples/vector_addition/vector_addition_with_timing.cpp)
  * Adds device selector to choose offload device
  * Provides timing comparison between device (SYCL) and host (non-SYCL)