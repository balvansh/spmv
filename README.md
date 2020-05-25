# SPMV

SPMV is a CUDA based benchmark for GPU which calculates the sparse vector matrix multiplication of compressed sparse row matrix.

## Make

```bash
make
```

## Usage

Available Kernels:

1. Monolithic Kernel with no optimizations
2. Grid+Stride implementation
3. Loop Unrolling enabled
4. Data Transfer Optimization
5. Loop Unrolling and Data Transfer Optimization
6. Loop Unrolling Data Transfer Optimization and Shared Memory

```bash
./output_spmv [#vector to be multiplied] [#number of iterations] [verbose=0/1] [input_file] [#select_kernel]
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

