# amm
The program allocates three n-by-n matrices: A, B, and C, where n is predefined as 1024. Matrices A and B are filled with random values, and matrix C is initialized to zero. 

The program then iteratively tests different block sizes for matrix multiplication, ranging from a minimum block size of 16 to a maximum of 128, increasing in steps of 16. For each block size, the find_optimal_threads function dynamically determines the optimal number of threads by actually performing matrix multiplication on smaller, 256x256 matrices for each potential thread count, from one up to the number of available CPU cores. 

This is done to measure and compare performance times, selecting the thread count that results in the fastest computation. The main computation leverages nested loops parallelized with OpenMP directives, breaking down the matrix multiplication into blocks to be computed concurrently, optimizing both memory access patterns and computation speed. The program outputs the block size and number of threads that achieved the best performance, characterized by the lowest computation time.cbrwx
