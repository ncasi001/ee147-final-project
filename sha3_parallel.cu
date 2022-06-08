#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

void gpu_init(); 
void run_benchmark();
char *read_message();
int gcd(int a, int b);

int clock_speed;
int number_multi_processors;
int number_blocks;
int number_threads;
int max_threads_per_mp;

cudaEvent_t start, stop;

int num_msgs;
const int dig_size = 256;
const int digsize_bytes = dig_size / 8;
const size_t str_length = 7;

#define ROTL(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

__device__ const char *chars = "123abcABC"; // interchangable

int gcd(int a, int b) {
    return (a == 0) ? b : gcd(b % a, a);
}

// rotation const
__device__ const uint64_t RC[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

// rotation offset
__device__ const int r[24] = {
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14, 
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

__device__ const int piln[24] = {
    10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4, 
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1 
};

__device__ void generate_message(char *message, uint64_t tid, int *str_len){
printf("in generate_message\n");
    int len = 0;
    const int num_chars = 94;
    char str[21];
    while (tid > 0){
  	str[len++] = chars[tid % num_chars];
	tid /= num_chars;
    }
	
    str[len] = '\0';
    memcpy(message, str, len + 1);
    *str_len = len;
}

// keccak function (24 rounds)
__device__ void keccak256(uint64_t state[25]){
    int i, j, rnd;
    uint64_t temp, C[5];

    // 24 rounds
    for(rnd = 0; rnd < 24; rnd++) {
       // theta
       for(i = 0; i < 5; i++) {
          C[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
       }

       for (i = 0; i < 5; i++) {
            temp = C[(i + 4) % 5] ^ ROTL(C[(i + 1) % 5], 1);
            for (j = 0; j < 25; j += 5) {
                state[j + i] ^= temp;
            }
        }

       // rho pi
       temp = state[1];
       for(i = 0; i < 24; i++) {
          j = piln[i];
          C[0] = state[j];
          state[j] = ROTL(temp, r[i]);
          temp = C[0];
       }

       // chi
       for(j = 0; i < 25; j += 5) {
          for(i = 0; i < 5; i++) {
             C[i] = state[j + i];
          }
          for(i = 0; i < 5; i++) {
             state[j + i] ^= (~C[(i + 1) % 5]) & C[(i + 2) % 5];
          }
       } 
     
       // iota
       state[0] ^= RC[rnd];
    }
}

// padding ????
__device__ void keccak(const char *msg, int msg_len, unsigned char *output, int output_len) {
    uint64_t state[25];
    uint8_t temp[144];
    int rsize = 136;
    int rsize_byte = 17;
    int i;

    memset(state, 0, sizeof(state));

    for( ; msg_len >= rsize; msg_len -= rsize, msg += rsize) {
       for(i = 0; i < rsize_byte; i++) {
          state[i] ^= ((uint64_t *) msg)[i];
       }
//printf("entering keccak256 1\n");
       keccak256(state);
    }

    // last block and padding
    memcpy(temp, msg, msg_len);
    temp[msg_len++] = 1;
    memset(temp + msg_len, 0, rsize - msg_len);
    temp[rsize - 1] |= 0x80;

    for(i = 0; i < rsize_byte; i++) {
       state[i] ^= ((uint64_t *) temp)[i];
    }
printf("entering keccak256 2\n");
    keccak256(state);
    memcpy(output, state, output_len);
}

__global__ void benchmark(const char *msgs, unsigned char *output, int num_msgs){
    const int str_len = 6;
    const int output_len = 32;
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    int num_threads = blockDim.x * gridDim.x;
	
    for (; tid < num_msgs; tid += num_threads) {
//printf("entering keccak()\n");
   	keccak(&msgs[tid * str_len], str_len, &output[tid * output_len], output_len);
    }
}

void gpu_init(){
    cudaDeviceProp device_prop;
    int device_count, block_size;
 
     cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        exit(EXIT_FAILURE);
    }

    if (cudaGetDeviceProperties(&device_prop, 0) != cudaSuccess) {
        exit(EXIT_FAILURE);
    }
    number_threads = device_prop.maxThreadsPerBlock;
    number_multi_processors = device_prop.multiProcessorCount;
    max_threads_per_mp = device_prop.maxThreadsPerMultiProcessor;
    block_size = (max_threads_per_mp / gcd(max_threads_per_mp, number_threads));
    number_threads = max_threads_per_mp / block_size;
    number_blocks = block_size * number_multi_processors;
    clock_speed = (int) (device_prop.memoryClockRate * 1000 * 1000);    
}

char *read_message(const char *file_name){
    FILE *f;
    if(!(f = fopen(file_name, "r"))) {
        printf("Error opening file %s", file_name);
        exit(1);
    }

    char *msgs = (char *) malloc(sizeof(char) * num_msgs * str_length);
    if (msgs == NULL) {
        perror("Error allocating memory for list of Strings.\n");
        exit(1);
    }
	
    int index = 0;
    char buf[10];
    while(1){
 	if (fgets(buf, str_length + 1, f) == NULL)
	    break;
	buf[strlen(buf) - 1] = '\0';
	memcpy(&msgs[index], buf, str_length);
	index += str_length - 1;
    }
printf("message:%s\n", msgs);	
    return msgs;
}

void run_benchmark(const char *file_name, int num_msgs){
   
    size_t arr_size = sizeof(char) *str_length * num_msgs;
    size_t output_size = digsize_bytes * num_msgs;

    // allocate host
    char* h_msg = read_message(file_name);
    unsigned char* h_output = (unsigned char*) malloc(output_size);
    char* d_msg;
    unsigned char* d_output;

    // allocate device
    cudaMalloc((void**) &d_msg, arr_size);
    cudaMalloc((void**) &d_output, output_size);
    int num_runs = 25;
    //test 

    // copy host to device
    int j;
    for(j = 0; j < num_runs; j++) {
//printf("entering benchmark<<<>>>\n");
       benchmark<<<number_blocks, number_threads>>>(d_msg, d_output, num_msgs);
    }

    // free memory
    free(h_msg);
    free(h_output);
    cudaFree(d_msg);
    cudaFree(d_output);
}

int main() {

    clock_t start, end;
    const char *file_name;
	
    file_name = "sha3.txt";
    num_msgs = 4000000;
    // test
//printf("entering gpu_init()\n");	
    gpu_init();
    
    start = clock();
// 
//printf("entering run_benchmark()\n");
    run_benchmark(file_name, num_msgs);

    end = clock();     
    printf("time used: %f\n",  (double)(end - start) / CLOCKS_PER_SEC); 
    printf("GPU encryption throughput: %f bytes/second\n",  (double)(num_msgs*10) / ((double)(end - start) / CLOCKS_PER_SEC)); 

    return EXIT_SUCCESS;
}
