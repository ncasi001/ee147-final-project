#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include <dirent.h>
#include <ctype.h>
#include <time.h>

#include "sha256.cuh"

char* trim(char *str){
    size_t len = 0;
    char *frontp = str;
    char *endp = NULL;

    if(str == NULL) { return NULL; }
    if(str[0] == '\0') { return str; }

    len = strlen(str);
    endp = str + len;

    /* Move the front and back pointers to address the first non-whitespace
     * characters from each end.
     */
    while(isspace((unsigned char) *frontp)) { ++frontp; }
    if(endp != frontp) {
        while(isspace((unsigned char) *(--endp)) && endp != frontp) {}
    }

    if(str + len - 1 != endp)
            *(endp + 1) = '\0';
    else if(frontp != str &&  endp == frontp)
            *str = '\0';

    /* Shift the string so that it starts at str so that if it's dynamically
     * allocated, we can still free it on the returned pointer.  Note the reuse
     * of endp to mean the front of the string buffer now.
     */
    endp = str;
    if(frontp != str) {
         while(*frontp) { *endp++ = *frontp++; }
            *endp = '\0';
    }


    return str;
}

__global__ void sha256_cuda(JOB ** jobs, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// perform sha256 calculation here
	if (i < n) {
		SHA256_CTX ctx;
		sha256_init(&ctx);
		sha256_update(&ctx, jobs[i]->data, jobs[i]->size);
		sha256_final(&ctx, jobs[i]->digest);
	}
}

void pre_sha256() {
	// copy symbols
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}


void runJobs(JOB** jobs, int n) {
	int blockSize = 4;
	int numBlocks = (n + blockSize - 1) / blockSize;
	sha256_cuda <<< numBlocks, blockSize >>> (jobs, n);
}


JOB* JOB_init(BYTE * data, long size, char * fname) {
	JOB* j;
	checkCudaErrors(cudaMallocManaged(&j, sizeof(JOB)));	//j = (JOB *)malloc(sizeof(JOB));
	checkCudaErrors(cudaMallocManaged(&(j->data), size));
	j->data = data;
	j->size = size;
	for (int i = 0; i < 64; i++) {
		j->digest[i] = 0xff;
	}
	strcpy(j->fname, fname);
	return j;
}


BYTE* get_file_data(char * fname, unsigned long * size) {
	FILE* f = 0;
	BYTE* buffer = 0;
	unsigned long fsize = 0;

	f = fopen(fname, "rb");
	if (!f) {
		fprintf(stderr, "get_file_data Unable to open '%s'\n", fname);
		return 0;
	}
	fflush(f);

	if (fseek(f, 0, SEEK_END)) {
		fprintf(stderr, "Unable to fseek %s\n", fname);
		return 0;
	}
	fflush(f);
	fsize = ftell(f);
	rewind(f);

	//buffer = (char *)malloc((fsize+1)*sizeof(char));
	checkCudaErrors(cudaMallocManaged(&buffer, (fsize+1)*sizeof(char)));
	fread(buffer, fsize, 1, f);
        
	unsigned long numbytes;
        fseek(f, 0L, SEEK_END);
        numbytes = ftell(f);
        printf("size of file: %lu bytes\n", numbytes);
	
	fclose(f);
	*size = fsize;
	return buffer;
}
/*
void print_usage(){
	printf("Usage: CudaSHA256 [OPTION] [FILE]...\n");
	printf("Calculate sha256 hash of given FILEs\n\n");
	printf("OPTIONS:\n");
	printf("\t-f FILE1 \tRead a list of files (separeted by \\n) from FILE1, output hash for each file\n");
	printf("\t-h       \tPrint this help\n");
	printf("\nIf no OPTIONS are supplied, then program reads the content of FILEs and outputs hash for each FILEs \n");
	printf("\nOutput format:\n");
	printf("Hash following by two spaces following by file name (same as sha256sum).\n");
	printf("\nNotes:\n");
	printf("Calculations are performed on GPU, each seperate file is hashed in its own thread\n");
}
*/
int main(int argc, char **argv) {
	int i = 0, n = 0;
	unsigned long temp;
	BYTE* buff;
	JOB** jobs;
        char index;

        clock_t start, end; 

        n = argc - optind;
	if (n > 0){
		checkCudaErrors(cudaMallocManaged(&jobs, n * sizeof(JOB *)));
		// iterate over file list - non optional arguments
		for (i = 0, index = optind; index < argc; index++, i++){
			buff = get_file_data(argv[index], &temp);
			jobs[i] = JOB_init(buff, temp, argv[index]);
		}
                start = clock();
		pre_sha256();
		runJobs(jobs, n);
	}
        

	cudaDeviceSynchronize();
        print_jobs(jobs, n);
	cudaDeviceReset();

        end = clock();
        printf("time used: %fs\n",  (double)(end - start) / CLOCKS_PER_SEC);               
	return 0;
}
