#include <stdio.h>
#include <stdlib.h>
//#include "kernel.cu"
#include "support.cu"

#include "sha.h"

int main (int argc, char *argv[]) 
{
   Timer timer;
   cudaError_t cuda_ret;

   // -------------------------------------
   printf("Setting up.\n"); fflush(stdout);
   startTime(&timer);

   // Simple gpu hash table
   
      // might scrap
   stopTime(&timer); printf("%f s\n", elapsedTime(timer));

   // SHA-3 serial
   printf("Running sha-3 serial.\n"); fflush(stdout);
   startTime(&timer);
   sha3_benchmark(25); // in sha3_serial.cu
   
   cuda_ret = cudaDeviceSynchronize();
      if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
   stopTime(&timer); printf("%f s\n", elapsedTime(timer));

   // SHA-3 GPU parallel
}
