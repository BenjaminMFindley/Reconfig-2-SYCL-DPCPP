# Vector Addition

These programs provide a basic introduction to SYCL and DPC++. 

* [vector_add_bad.cpp](vector_add_bad.cpp)
  * Simple SYCL program to showcase data management using queues, buffers, accessors, and kernels
  * Demonstrates a common bug.
  
* [vector_add1.cpp](vector_add1.cpp)
  * Corrects the bug from the previous example by demonstrating two different methods of ensureing output data is transferred back to the host.

* [vector_add2.cpp](vector_add2.cpp)
  * Adds exception handling to previous example
  
* [vector_add_terse.cpp](vector_add_terse.cpp)
  * Demonstrates a semantically equivalent version of the previous example with a more concise coding style.
  
## Devcloud instructions

Find suitable node (e.g. with gen9 gpu):  
`pbsnodes | grep -B 1 -A 8 "state = free" | grep -B 4 -A 4 gen9`

Login with interactive shell (replace the nodes section with a free node specified by the previous command):   
`qsub -I -l nodes=s001-n234:ppn=2`

Compile:   
`icpx -fsycl input_file -o output_file -Wall -O3`
   
Run:   
`./output_file`  
