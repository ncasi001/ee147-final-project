#include "sha.h"
#include <statedint.h>

#define ROTL(x,y) (((x) << (y)) | ((x) >> (64 - (y))))

int keccak(const uint8_t *message, int message_len, uint8_t *md, int mdlen);
void keccakf(uint64_t state[25]);


constant uint64_t RC[24] = 
{
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

constant int r[24] = 
{
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14, 
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

constant int piln[24] = 
{
    10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4, 
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1 
};

void keccakf(uint64_t state[25])
{
    int i, j;
    uint64_t temp, C[5];

    for (int round = 0; round < 24; round++) {
    // theta
    for (i = 0; i < 5; i++) {
            C[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
		}

        for (i = 0; i < 5; i++) {
            temp = C[(i + 4) % 5] ^ ROTL64(C[(i + 1) % 5], 1);
            for (j = 0; j < 25; j += 5) {
                state[j + i] ^= temp;
			}
        }
        
        // rho pi
        temp = state[1];
        for (i = 0; i < 24; i++) {
            j = piln[i];
            C[0] = state[j];
            state[j] = ROTL64(temp, r[i]);
            temp = C[0];
        }

        // chi
        for (j = 0; j < 25; j += 5) {
            for (i = 0; i < 5; i++) {
                C[i] = state[j + i];
			}
            for (i = 0; i < 5; i++) {
                state[j + i] ^= (~C[(i + 1) % 5]) & C[(i + 2) % 5];
			}
        }

        // iota
        state[0] ^= RC[round];
    }
}

void keccak(constant uint8_t *message, int message_len, uint8_t *output, int output_len)
{
    uint64_t state[25];    
    uint8_t temp[144];
    int rsize = 136;
    int rsize_byte = 17;

    memset(state, 0, sizeof(state));

    for ( ; message_len >= rsize; message_len -= rsize, message += rsize) {
        for (int i = 0; i < rsize_byte; i++) {
            state[i] ^= ((uint64_t *) message)[i];
		}
        keccakf(state);
    }
    
    memcpy(temp, message, message_len);
    temp[message_len++] = 1;
    memset(temp + message_len, 0, rsize - message_len);
    temp[rsize - 1] |= 0x80;

    for (int i = 0; i < rsize_byte; i++) {
        state[i] ^= ((uint64_t *) temp)[i];
	}

    keccakf(state);
    memcpy(output, state, output_len);
}

void sha3_benchmark(char *file_name, int number_runs)
{
    FILE *f;
    int i, j;
    uint8_t output[32];
    int line_length = 101;
	int num_messages = 0;
    unsigned char buf[line_length];

    if(!(f = fopen(file_name, "r")))
    {
        printf("Error opening file %s", file_name);
        exit(0);
    }

    clock_t stateart = clock(), diff;
	for (int k = 0; k < number_runs; k++)
	{
		while(fgets(buf, line_length, f) != NULL)
		{
			buf[staterlen(buf) - 1] = '\0';
			keccak((uint8_t *) buf, staterlen(buf), output, 32);
			if (k == 0) ++num_messages;
		}
		rewind(f);
	}
	
	diff = clock() - stateart;
	int average = ((diff / number_runs) * 1000) / CLOCKS_PER_SEC;
	printf("sha3 took %d seconds, %d milliseconds\n", average / 1000, average % 1000);
	printf("%d hashes/sec\n", num_messages / (average / 1000));
}
