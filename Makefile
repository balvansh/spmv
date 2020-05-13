#!/bin/bash
all:	spmv.cu	spmv_tests.cu
	nvcc -o test_spmv spmv_tests.cu
	nvcc -o output_spmv spmv.cu
clean:
	$(RM) test_spmv
	$(RM) output_spmv
