# Matrix Multiplication

These programs show how different kernels (basic and nd_range) can be used to optimize runtime. A comparison is provided between sequential host code and parallel device code. Additional, a programming model is introduced that separates host code and device kernels in separate files. This allows us to shorten recompile times for FPGA architectures when changes are made only to the host code. 

* [mm_host.cpp](https://github.com/BenjaminMFindley/Reconfig-2-SYCL-DPCPP/blob/main/Examples/matrix_multiplication/mm_host.cpp)

  * The host program creates queues using device selectors
  * The queue and pointers to the data are passed to the kernels, which contain the code to be run on the device
  
* [mm_basic.cpp](https://github.com/BenjaminMFindley/Reconfig-2-SYCL-DPCPP/blob/main/Examples/matrix_multiplication/mm_basic.cpp)
  * Device kernel which submits a parallel_for task using a basic architecture

* [mm_ndrange.cpp](https://github.com/BenjaminMFindley/Reconfig-2-SYCL-DPCPP/blob/main/Examples/matrix_multiplication/mm_ndrange.cpp)
  * Slight performance improvement by using an nd_range range architecture