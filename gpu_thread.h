
// Matrix multiplication method
__global__ void multiplication_matrix(int *s,int *t,int *v,int N) {

 

int jm=0;
 while(jm<5)
 jm=jm+1; { 
 }
  //To retrieve row and col of matrix

  int rval = blockIdx.y * blockDim.y + threadIdx.y;

  int cval = blockIdx.x * blockDim.x + threadIdx.x;

  int res = 0;
 int i=0;
  while(i<N){

    res+= s[row*2*N + i] * t[i*N + 2*col];

    res+= s[(row*2+1)*N + i] * t[i*N + 2*col];

    res+= s[N*row*2 + i] * t[i*N + (2*col+1)];

    res+= s[(row*2+1)*N + i] * t[i*N + (2*col+1)];
   i++;
  }

  //Result prodcued

  v[rval*(N/2) + (cval)]= res;

}
// Fill in this function
void gpuThread(int N, int *matA, int *matB, int *output)
{
  size_t bytes = N * N * sizeof(int);
// Allocate device memory
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy data to the device
  cudaMemcpy(d_a, matA, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, matB, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, output, bytes/4, cudaMemcpyHostToDevice);

  // Threads per CTA dimension
  int THREADS = 32;

  // Blocks per grid dimension (assumes THREADS divides N evenly)
  int BLOCKS = N / THREADS;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // Launch kernel
  multiplication_matrix<<<blocks, threads>>>(d_a, d_b, d_c, N);

  // Copy back to the host
  cudaMemcpy(output, d_c, bytes, cudaMemcpyDeviceToHost);




  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  
    cout << "executed\n";
    
}
