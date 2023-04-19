# Accumulation

These programs demonstrate numerous common bugs with SYCL (and similar approaches) using an accumulation example that adds all the values in a given array.
In addition, it illustrates common optimization strategies that optimize execution time 1B inputs from 85 seconds to under 2 seconds/

* [accum_bad1.cpp](accum_bad1.cpp)
  * Demonstrates a common incorrect strategy for accumulation that "works" for small inputs (on the DevCloud nodes), but reports errors for larger inputs.

* [accum_bad2.cpp](accum_bad2.cpp)
  * Demonstrates a seemingly logical attempt to fix the bug in the previous example. While this fix does work for larger input sizes than the previous example, it still doesn't work for all inputs. 
  
* [accum_correct_super_slow1.cpp](accum_correct_super_slow1.cpp)
  * Presents a correct (albeit very slow) solution that uses a separate output array to avoid synchronization problems.

* [accum_correct_super_slow2.cpp](accum_correct_super_slow2.cpp)
  * Optimizes the previous example by minimizing the data transferred between the host and the compute device.

* [accum_correct_super_slow3.cpp](accum_correct_super_slow3.cpp)
  * Further optimizes the previous example by reducing the size of the output transfers.

* [accum_correct_super_slow3.cpp](accum_correct_super_slow4.cpp)
  * Further optimizes the previous example by reducing work-items.

* [accum_correct_slow1.cpp](accum_correct_slow1.cpp)
  * Presents an alternative correct solution that does not require a separate output array for synchronization.

* [accum_correct_slow2.cpp](accum_correct_slow2.cpp)
  * Optimizes the previous approach using local memory to minimize global memory accesses.
  * This solution reduces execution time to under 2 seconds for 1B inputs.
  *   
## Devcloud instructions

Find suitable node (e.g. with gen9 gpu):  
`pbsnodes | grep -B 1 -A 8 "state = free" | grep -B 4 -A 4 gen9`

Login with interactive shell (replace the nodes section with a free node specified by the previous command):   
`qsub -I -l nodes=s001-n234:ppn=2`

Compile:   
`icpx -fsycl input_file -o output_file -Wall -O3`
   
Run:   
`./output_file num_inputs`  
