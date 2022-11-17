#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <chrono>
using namespace sycl;

// define kernels for offloading computations
void mm_basic_kernel(queue& deviceQueue, std::vector<double>& in1, std::vector<double>& in2, std::vector<double>& out, size_t N);
void mm_ndrange_kernel(queue& deviceQueue, std::vector<double>& in1, std::vector<double>& in2, std::vector<double>& out, size_t N, size_t B);

#define MATRIX_SIZE 1024
#define WORKGROUP_SIZE 16

int main(int argc, char* argv[]) {

    size_t N = MATRIX_SIZE;
    size_t B = WORKGROUP_SIZE;
    bool printResult = false;
    bool compareResult = true;
    bool validateResult = false;

    std::cout << "\nRunning matrix multiplication\n" 
              << "Matrix size = [ " << N << " x " << N << " ]\n\n";

    // define 1-D vectors with size to hold NxN matrices
    std::vector<double> in1(N * N);
    std::vector<double> in2(N * N);
    std::vector<double> outCPU(N * N);
    std::vector<double> outGPU(N * N);
    std::vector<double> outFPGA(N * N);
    std::vector<double> outVal(N * N);

    // load vectors
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            in1[i * N + j]      = rand() % 100;
            in2[i * N + j]      = rand() % 100;
            outCPU[i * N + j]   = 0.0;
            outGPU[i * N + j]   = 0.0;
            outFPGA[i * N + j]  = 0.0;
            outVal[i * N + j]   = 0.0;
        }
    }
    
    // define property list for queues-- enables timing analysis
    auto propertyList = property::queue::enable_profiling();
    
    // use default selectors for the offload devices-- can make custom ones with ranked choices based on HW available
    cl::sycl::cpu_selector deviceSelectorCPU;
    cl::sycl::gpu_selector deviceSelectorGPU;
    //cl::sycl::ext::intel::fpga_selector deviceSelectorFPGA;
    //cl::sycl::ext::intel::fpga_emulator_selector deviceSelectorFPGAemu;

    // create queues for the offload devices
    queue deviceQueueCPU(deviceSelectorCPU, propertyList);
    queue deviceQueueGPU(deviceSelectorGPU, propertyList);

    //------------------------ CPU BASIC ----------------------------------------------

    // print CPU information
    std::cout << "Offload Device       : " << deviceQueueCPU.get_device().get_info<info::device::name>() << "\n";
    std::cout << "max_work_group_size  : " << deviceQueueCPU.get_device().get_info<info::device::max_work_group_size>() << "\n\n";

    // capture timing for start of offload
    auto deviceStartCPU = std::chrono::high_resolution_clock::now();

    // run matrix multiplication basic kernel on CPU
    mm_basic_kernel(deviceQueueCPU, in1, in2, outCPU, N);

    // capture timing for end of offload
    auto deviceStopCPU = std::chrono::high_resolution_clock::now();
    auto deviceDurationCPU = std::chrono::duration_cast<std::chrono::milliseconds>(deviceStopCPU - deviceStartCPU);

    std::cout << "Total offload time    : " << deviceDurationCPU.count() << " milliseconds\n\n";

    //------------------------ CPU ND-RANGE ----------------------------------------------
    
    // capture timing for start of offload
    deviceStartCPU = std::chrono::high_resolution_clock::now();

    // run matrix multiplication basic kernel on CPU
    mm_ndrange_kernel(deviceQueueCPU, in1, in2, outCPU, N, B);

    // capture timing for end of offload
    deviceStopCPU = std::chrono::high_resolution_clock::now();
    deviceDurationCPU = std::chrono::duration_cast<std::chrono::milliseconds>(deviceStopCPU - deviceStartCPU);

    std::cout << "Total offload time    : " << deviceDurationCPU.count() << " milliseconds\n\n";

    //------------------------ GPU BASIC ----------------------------------------------

    // print GPU information
    std::cout << "Offload Device      : " << deviceQueueGPU.get_device().get_info<info::device::name>() << "\n";
    std::cout << "max_work_group_size : " << deviceQueueGPU.get_device().get_info<info::device::max_work_group_size>() << "\n\n";

    // capture timing for start of GPU block
    auto deviceStartGPU = std::chrono::high_resolution_clock::now();

    // run matrix multiplication basic kernel on GPU
    mm_basic_kernel(deviceQueueGPU, in1, in2, outGPU, N);

    // capture timing for end of SYCL block
    auto deviceStopGPU = std::chrono::high_resolution_clock::now();
    auto deviceDurationGPU = std::chrono::duration_cast<std::chrono::milliseconds>(deviceStopGPU - deviceStartGPU);

    std::cout << "Total offload time    : " << deviceDurationGPU.count() << " milliseconds\n\n";


    //------------------------ GPU ND-RANGE ----------------------------------------------

    // capture timing for start of GPU block
    deviceStartGPU = std::chrono::high_resolution_clock::now();

    // run matrix multiplication basic kernel on GPU
    mm_ndrange_kernel(deviceQueueGPU, in1, in2, outGPU, N, B);

    // capture timing for end of SYCL block
    deviceStopGPU = std::chrono::high_resolution_clock::now();
    deviceDurationGPU = std::chrono::duration_cast<std::chrono::milliseconds>(deviceStopGPU - deviceStartGPU);

    std::cout << "Total offload time    : " << deviceDurationGPU.count() << " milliseconds\n\n";
    
    
    
    
    
    
    
    
    // can only validate if comparison exists
    if (validateResult) {
        compareResult = true;
    }

    // compare device time to host time
    if (compareResult) {
        // host computation for validation and timing comparision
        auto vanillaStart = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    outVal[i * N + j] += in1[i * N + k] * in2[k * N + j];
                }
            }
        }
        auto vanillaStop = std::chrono::high_resolution_clock::now();
        auto vanillaDuration = std::chrono::duration_cast<std::chrono::milliseconds>(vanillaStop - vanillaStart);

        std::cout << "Compare to non-SYCL sequential compute time:\n " << vanillaDuration.count() << " milliseconds\n";

        // validate device results with host results
        if (validateResult) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    if ((outCPU[i * N + j] - outVal[i * N + j]) > 1e-6) {
                        std::cout << "CPU validation failed\n";
                        return -1;
                    }
                    if ((outGPU[i * N + j] - outVal[i * N + j]) > 1e-6) {
                        std::cout << "GPU validation failed\n";
                        return -1;
                    }
                }
            }
            std::cout << "Validation passed\n";
        }
    }

    // print
    if (printResult&&(N<10)) {
        std::cout << "\n";
        for (int i = 0; i < N; i++) {
            std::cout << "[ ";
            for (int j = 0; j < N; j++) {
                std::cout << in1[i * N + j] << " ";
            }
            if (i == 0) {
                std::cout << "] * [ ";
            }
            else {
                std::cout << "]   [ ";
            }
            for (int j = 0; j < N; j++) {
                std::cout << in2[i * N + j] << " ";
            }
            if (i == 0) {
                std::cout << "] = [ ";
            }
            else {
                std::cout << "]   [ ";
            }
            for (int j = 0; j < N; j++) {
                std::cout << outCPU[i * N + j] << " ";
            }
            std::cout << "]\n";
        }
    }
    else if (printResult) {
        std::cout << "Too big to print\n";
    }

    return 0;
}