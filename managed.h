//create a managed class that will automatically allocate space in the unified memory, thus avoiding us having to deep copy anything from host -> device or using ugly double pointers to initialize everything within a kernel
//Based on Mark Harris NVIDIA blog post about unified memory in CUDA 6 (November, 2013).
#pragma once

class Managed {
public:
  void *operator new(size_t len) {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
  }

  void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};  
