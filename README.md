# A test implementation of multiple ways of doing matrix multiplication
This is loosely based on the implementation in Ulrich Drepper's excellent "What Every Programmer Should Know About Memory"

There are 3 approaches implemented in this version
- Naive implementation
- An implementation that transposes the second matrix before multiplying
- An unrolled loop that processses the matrices in tiles of 64 by 64 bytes (cache line size on most modern architectures)
