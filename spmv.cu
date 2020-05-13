#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>

inline cudaError_t _checkError(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

void read_matrix(int **r_ptr, int** c_ind,float** v, char*fname,int* r_count,int* v_count){	
	
	//*************************************************************
        //READ AND CONVERT COO MATRIX TO CSR MATRIX
        //*************************************************************

	FILE * file;
    	if ((file = fopen(fname, "r+")) == NULL)
	{
	    printf("ERROR: file open failed\n");
	    return;
	}
	
	int column_count,row_count,values_count;
	fscanf(file, "%d %d %d\n",&row_count,&column_count,&values_count);
	*r_count = row_count;
	*v_count = values_count;
	int i;
	int *row_ptr =(int*) malloc((row_count+1) * sizeof(int));
	int *col_ind =(int*) malloc(values_count * sizeof(int));
	for(i=0; i<values_count; i++){
		col_ind[i] = -1;
	}
	float *values =(float*) malloc(values_count * sizeof(float));
	int row,column;
	float value;

	//*************************************************************
        //GENERATING THE ROW VECTOR FOR CSR
       	//*************************************************************

	while (1) {
		int ret = fscanf(file, "%d %d %f\n",&row,&column,&value);
		column --;
		row --;
		if(ret == 3){
			row_ptr[row]++;
		} else if(ret == EOF) {
		   	break;
		} else {
		    	printf("No match.\n");
		}
	}
    	rewind(file);
	//printf("The row count: %d\n",row_count);
    	int index = 0;
    	int val = 0;
	for(i = 0; i<row_count;i++){
		val = row_ptr[i];
		//printf("The value is : %d\n",val);
		row_ptr[i] = index;
		index += val;
		//printf("row_ptr[%d] = %d\n",i, row_ptr[i]);
	}
	row_ptr[row_count] = values_count;
	fscanf(file, "%d %d %d\n",&row_count,&column_count,&values_count);
	i = 0;
	while (1) {
		int ret = fscanf(file, "%d %d %f\n",&row,&column,&value);
		column --;
		row --;
		if(ret == 3){
			while(col_ind[i+row_ptr[row]] != -1){ i++;}
			col_ind[i+row_ptr[row]] = column;
			values[i+row_ptr[row]] = value;
			i=0;
		} else if(ret == EOF) {
		   	break;
		} else {
		    	printf("No match.\n");
		}
	}
    	fclose(file);
    	*r_ptr = row_ptr;
    	*c_ind = col_ind;
    	*v = values;
}

void cpu_multiply(const int num_rows,const int *ptr, const int *indices, const float *data, const float *x, float* y){
	float dot;
	int row_start, row_end;
	for (int i=0; i<num_rows; i++){
		dot = 0;
                /*printf("CPU Matrix\n");
                for(int l = 0; l<num_rows;l++){
                        if(l+1 <= num_rows){
                                for(int k = ptr[l]; k < ptr[l+1];k++){
                                        printf("%d %d %.10f\n",l+1,indices[k]+1,data[k]);
                                }
                        }
                }*/
                row_start = ptr[i];
                row_end = ptr[i + 1];
                //printf("row_start: %d\t row end: %d\n",row_start, row_end);
                for(int j = row_start; j < row_end; j++){
                        //printf("data[%d] = %10f * indices[%d] = %d x[indices[%d]]= %10f\n", i, data[i], i, indices[i]+1, i,  x[indices[i]]);
                        dot+= data[j] * x[indices[j]];
                }
		y[i] += dot;
	}
}

__global__ void vector_multiply_unopti(const int num_rows,const int *ptr,const int *indices,const float *data,const float *x, float* y){
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	/*printf("Kernel Matrix\n");
        for(int i = 0; i<num_rows;i++){
        	if(i+1 <= num_rows){
        		for(int k = ptr[i]; k < ptr[i+1];k++){
                		printf("%d %d %.10f\n",i+1,indices[k]+1,data[k]);
                	}
        	}	
        }*/
	//printf("In Kernel:\n",row);
	int i;
	int row_start, row_end;
	float dot;
	if(row < num_rows){
		//printf("row=%d\n",row);
		dot = 0;
		/*printf("Kernel Matrix\n");
        	for(int l = 0; l<num_rows;l++){
                	if(l+1 <= num_rows){
                        	for(int k = ptr[l]; k < ptr[l+1];k++){
                                	printf("%d %d %.10f\n",l+1,indices[k]+1,data[k]);
                        	}
                	}
        	}*/
		row_start = ptr[row];
		row_end = ptr[row + 1];
		//printf("row_start: %d\t row end: %d\n",row_start, row_end);
		for(i = row_start; i < row_end; i++){
			//printf("data[%d] = %10f * indices[%d] = %d x[indices[%d]]= %10f\n", i, data[i], i, indices[i]+1, i,  x[indices[i]]);
			dot+= data[i] * x[indices[i]];
		}
	}
	y[row] += dot;
}

__global__ void vector_multiply_unroll(const int num_rows,const int *ptr,const int *indices,const float *data,const float *x, float* y){
        int row;// = blockIdx.x * blockDim.x + threadIdx.x;
        //printf("In Kernel:\n",row);
        int i;
        int row_start, row_end;
        float dot;
	#pragma unroll
        for (row = blockIdx.x * blockDim.x + threadIdx.x; row < num_rows; row += blockDim.x * gridDim.x)
        {
                dot = 0;
                /*printf("Kernel Matrix\n");
                for(int l = 0; l<num_rows;l++){
                        if(l+1 <= num_rows){
                                for(int k = ptr[l]; k < ptr[l+1];k++){
                                        printf("%d %d %.10f\n",l+1,indices[k]+1,data[k]);
                                }
                        }
                }*/
                row_start = ptr[row];
                row_end = ptr[row + 1];
                #pragma unroll
                for(i = row_start; i < row_end; i++){
                        dot+= data[i] * x[indices[i]];
                }
        }

}

//TODO: Fix this implementation
__global__ void vector_multiply_shared(const int num_rows,const int *ptr,const int *indices,const float *data,const float *x, float* y){
        __shared__ float s[256];
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("In Kernel:\n",row);

	s[row]=0.0;
	/*for(int l=0; l<num_rows; l++){
		s[l] = 0.0;
	}*/
        //printf("No error:\n",row);
        int i;
        int row_start, row_end;
        //float dot;
        if(row < num_rows){
                //dot = 0;
		s[row] = 0.0;
                /*printf("Kernel Matrix\n");
                for(int l = 0; l<num_rows;l++){
                        if(l+1 <= num_rows){
                                for(int k = ptr[l]; k < ptr[l+1];k++){
                                        printf("%d %d %.10f\n",l+1,indices[k]+1,data[k]);
                                }
                        }
                }*/
                row_start = ptr[row];
                row_end = ptr[row + 1];
                #pragma unroll
                for(i = row_start; i < row_end; i++){
                        s[row]+= data[i] * x[indices[i]];
                }
        }
        //s[row] += dot;
	__syncthreads();
	y[row] = s[row];
}

__global__ void vector_multiply_stride(const int num_rows,const int *ptr,const int *indices,const float *data,const float *x, float* y){
        int row;// = blockIdx.x * blockDim.x + threadIdx.x;
        //printf("In Kernel:\n",row);
        int i;
        int row_start, row_end;
        float dot;
	for (row = blockIdx.x * blockDim.x + threadIdx.x; row < num_rows; row += blockDim.x * gridDim.x) 
      	{
        	dot = 0;
                /*printf("Kernel Matrix\n");
                for(int l = 0; l<num_rows;l++){
                        if(l+1 <= num_rows){
                                for(int k = ptr[l]; k < ptr[l+1];k++){
                                        printf("%d %d %.10f\n",l+1,indices[k]+1,data[k]);
                                }
                        }
                }*/
                row_start = ptr[row];
                row_end = ptr[row + 1];
                for(i = row_start; i < row_end; i++){
                        dot+= data[i] * x[indices[i]];
                }
      	}
        y[row] += dot;
}

//TODO: Fix the prefetch method

__global__ void vector_multiply_stride_prefetch(const int num_rows,const int *ptr,const int *indices,const float *data,const float *x, float* y){
        int row;// = blockIdx.x * blockDim.x + threadIdx.x;
        //printf("In Kernel:\n",row);
        int i;
        int row_start, row_end;
        float dot;
	#pragma unroll
        for (row = blockIdx.x * blockDim.x + threadIdx.x; row < num_rows; row += blockDim.x * gridDim.x)
        {
                dot = 0;
		//__prefetch_global_l1(row_start);
                /*printf("Kernel Matrix\n");
                for(int l = 0; l<num_rows;l++){
                        if(l+1 <= num_rows){
                                for(int k = ptr[l]; k < ptr[l+1];k++){
                                        printf("%d %d %.10f\n",l+1,indices[k]+1,data[k]);
                                }
                        }
                }*/
                row_start = ptr[row];
                row_end = ptr[row + 1];
                #pragma unroll
                for(i = row_start; i < row_end; i++){
                        dot+= data[i] * x[indices[i]];
                }
        }
        y[row] += dot;
}

int main (int argc, char* argv[]){

	//*************************************************************
	//Argument List: vector_to_multiply, repetitions_for_the code, 
	//		 verbose, filename_matrixi, 
	//		 optimizations_to_apply
	//*************************************************************

	//*************************************************************
	//HOST CODE
	//*************************************************************
	
	if ( argc != 6){
		printf( "Incorrect usage");
	}
	else{
		int* row_ptr;
		int* col_ind;
		float* values;
		int r_count, v_count, i, k, j;
		int num = atoi(argv[1]);
		int repetitions = atoi(argv[2]);
		int mode = atoi(argv[3]);
		char* fname = argv[4];
		int opti = atoi(argv[5]);
	
		//*************************************************************
		//READING THE GIVEN COORDINATE FORM MATRIX AND CONVERTING IT TO 
		//COMPRESSED SPARSE ROW MATRIX
		//*************************************************************
	
		read_matrix(&row_ptr, &col_ind, &values, fname, &r_count, &v_count);
		float* x =(float*) malloc(r_count* sizeof(float));
		float* y =(float*) calloc(r_count, sizeof(float));
		float* cy =(float*) calloc(r_count, sizeof(float));
		for(i = 0; i<r_count;i++){
		    	x[i]= (float)num;
		}

		if(mode == 1){

			//*************************************************************
			//PRINT OUT THE GIVEN COORDINATE MATRIX STORED IN THE FORM OF
			//CSR MATRIX
			//*************************************************************

			fprintf(stdout,"PERFORMING SPMV ON MATRIX: \n");
			for(i = 0; i<r_count;i++){
		    		if(i+1 <= r_count){
		    			for(k = row_ptr[i]; k < row_ptr[i+1];k++){
		    				fprintf(stdout,"%d %d %.10f\n",i+1,col_ind[k]+1,values[k]);
		    			}
		    		}	
		    	}

	  	}
		//TODO: Timer for CPU run
		//*************************************************************
                //RUNNING ON CPU
                //*************************************************************
		/*struct timeval cstart, cend;
    		long mtime, secs, usecs;*/ 
		float cdot;	
		//gettimeofday(&cstart, NULL);
		for(k = 0; k<repetitions; k++){
			//cpu_multiply(r_count, row_ptr, col_ind, values, x, y);
        		for (i=0; i<r_count; i++){
                		cdot = 0;
                		for(j = row_ptr[i]; j < row_ptr[i+1]; j++){
                        		cdot+= values[j] * x[col_ind[j]];
                		}
                	cy[i] += cdot;
        		}
		}
		/*gettimeofday(&cend, NULL);
		secs  = cend.tv_sec  - cstart.tv_sec;
    		usecs = cend.tv_usec - cstart.tv_usec;
    		mtime = ((secs) * 1000 + usecs/1000.0) + 0.5;
    		printf("Elapsed time: %ld millisecs\n", mtime);*/

		//*************************************************************
                //COMPLETED RUNNING ON CPU
                //*************************************************************

		//*************************************************************
		//SETTING UP GPU RUN
		//*************************************************************

		int *d_row_ptr, *d_col_ind;
		float *d_values, *d_x, *d_y;

		cudaEvent_t start,stop,st,sp;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
		cudaEventCreate(&st);
                cudaEventCreate(&sp);
		float milliseconds = 0;
		//float timing = 0;
		//int bytes = r_count * sizeof(float);

		switch(opti){
			case 1:

				//*************************************************************
                		//RUNNING KERNEL WITH NO OPTIMIZATIONS MONOLITHIC
                		//*************************************************************

				cudaMalloc(&d_row_ptr, r_count*sizeof(int));
                		cudaMalloc(&d_col_ind, v_count*sizeof(int));
                		cudaMalloc(&d_values, v_count*sizeof(int));
                		cudaMalloc(&d_x, r_count*sizeof(float));
                		cudaMalloc(&d_y, r_count*sizeof(float));

				cudaMemcpy(d_row_ptr, row_ptr, r_count*sizeof(int), cudaMemcpyHostToDevice);
                		cudaMemcpy(d_col_ind, col_ind, v_count*sizeof(int), cudaMemcpyHostToDevice);
                		cudaMemcpy(d_values, values, v_count*sizeof(int), cudaMemcpyHostToDevice);

				cudaEventRecord(start);
               	 		for(k = 0; k<repetitions; k++){
                        		//printf("Calling Kernel\n");
                        		cudaMemcpy(d_x, x, r_count*sizeof(float), cudaMemcpyHostToDevice);
                        		cudaMemcpy(d_y, y, r_count*sizeof(float), cudaMemcpyHostToDevice);

                        
                        		float blocksize = 64;
                        		float blocknum = ceil(r_count/blocksize); 
                        		//printf("blocknum: %f\n",blocknum);
                        		//blocknum = 1.0;
                        		//printf("blocknum: %f\n",blocknum);
                        		vector_multiply_unopti<<<blocknum, blocksize>>>(r_count, d_row_ptr, d_col_ind, d_values, d_x, d_y);
                        		cudaMemcpy(y, d_y, r_count*sizeof(float), cudaMemcpyDeviceToHost);
                        		for(i = 0; i<r_count;i++){
                                		x[i] = y[i];
                                		y[i]= 0.0;
                        		}
                		}

                		cudaEventRecord(stop);
                		cudaEventSynchronize(stop);

                		milliseconds = 0;
                		cudaEventElapsedTime(&milliseconds, start, stop);

				break;
			case 2:

                                //*************************************************************
                                //RUNNING KERNEL WITH NO OPTIMIZATIONS STRIDE
                                //*************************************************************

                                cudaMalloc(&d_row_ptr, r_count*sizeof(int));
                                cudaMalloc(&d_col_ind, v_count*sizeof(int));
                                cudaMalloc(&d_values, v_count*sizeof(int));
                                cudaMalloc(&d_x, r_count*sizeof(float));
                                cudaMalloc(&d_y, r_count*sizeof(float));

                                cudaMemcpy(d_row_ptr, row_ptr, r_count*sizeof(int), cudaMemcpyHostToDevice);
                                cudaMemcpy(d_col_ind, col_ind, v_count*sizeof(int), cudaMemcpyHostToDevice);
                                cudaMemcpy(d_values, values, v_count*sizeof(int), cudaMemcpyHostToDevice);

                                cudaEventRecord(start);
                                for(k = 0; k<repetitions; k++){
                                        //printf("Calling Kernel\n");
                                        cudaMemcpy(d_x, x, r_count*sizeof(float), cudaMemcpyHostToDevice);
                                        cudaMemcpy(d_y, y, r_count*sizeof(float), cudaMemcpyHostToDevice);


                                        float blocksize = 64;
                                        float blocknum = ceil(r_count/blocksize);
                                        //printf("blocknum: %f\n",blocknum);
                                        //blocknum = 1.0;
                                        //printf("blocknum: %f\n",blocknum);
                                        vector_multiply_stride<<<blocknum, blocksize>>>(r_count, d_row_ptr, d_col_ind, d_values, d_x, d_y);
                                        cudaMemcpy(y, d_y, r_count*sizeof(float), cudaMemcpyDeviceToHost);
                                        for(i = 0; i<r_count;i++){
                                                x[i] = y[i];
                                                y[i]= 0.0;
                                        }
                                }

                                cudaEventRecord(stop);
                                cudaEventSynchronize(stop);

                                milliseconds = 0;
                                cudaEventElapsedTime(&milliseconds, start, stop);

                                break;
			case 3:
				//*************************************************************
                                //RUNNING KERNEL WITH LOOP UNROLLING
                                //*************************************************************

                                cudaMalloc(&d_row_ptr, r_count*sizeof(int));
                                cudaMalloc(&d_col_ind, v_count*sizeof(int));
                                cudaMalloc(&d_values, v_count*sizeof(int));
                                cudaMalloc(&d_x, r_count*sizeof(float));
                                cudaMalloc(&d_y, r_count*sizeof(float));

                                cudaMemcpy(d_row_ptr, row_ptr, r_count*sizeof(int), cudaMemcpyHostToDevice);
                                cudaMemcpy(d_col_ind, col_ind, v_count*sizeof(int), cudaMemcpyHostToDevice);
                                cudaMemcpy(d_values, values, v_count*sizeof(int), cudaMemcpyHostToDevice);

                                cudaEventRecord(start);
                                for(k = 0; k<repetitions; k++){
                                        //printf("Calling Kernel\n");
                                        cudaMemcpy(d_x, x, r_count*sizeof(float), cudaMemcpyHostToDevice);
                                        cudaMemcpy(d_y, y, r_count*sizeof(float), cudaMemcpyHostToDevice);


                                        float blocksize = 64;
                                        float blocknum = ceil(r_count/blocksize);
                                        //printf("blocknum: %f\n",blocknum);
                                        //blocknum = 1.0;
                                        //printf("blocknum: %f\n",blocknum);
                                        vector_multiply_unroll<<<blocknum, blocksize>>>(r_count, d_row_ptr, d_col_ind, d_values, d_x, d_y);
                                        cudaMemcpy(y, d_y, r_count*sizeof(float), cudaMemcpyDeviceToHost);
                                        for(i = 0; i<r_count;i++){
                                                x[i] = y[i];
                                                y[i]= 0.0;
                                        }
                                }

                                cudaEventRecord(stop);
                                cudaEventSynchronize(stop);

                                milliseconds = 0;
                                cudaEventElapsedTime(&milliseconds, start, stop);
				break;
			case 4:
				//*************************************************************
                                //RUNNING KERNEL WITH DATA TRANSFER OPTIMIZATIONS
                                //*************************************************************
				
				//int bytes = r_count * sizeof(float);

                          	cudaMallocHost(&d_row_ptr, r_count*sizeof(int));
                		cudaMallocHost(&d_col_ind, v_count*sizeof(int));
                		cudaMallocHost(&d_values, v_count*sizeof(int));
                		cudaMallocHost(&d_x, r_count*sizeof(float));
                		cudaMallocHost(&d_y, r_count*sizeof(float));   
                            
                                cudaMemcpy(d_row_ptr, row_ptr, r_count*sizeof(int), cudaMemcpyHostToDevice);

                                cudaMemcpy(d_col_ind, col_ind, v_count*sizeof(int), cudaMemcpyHostToDevice);
                                cudaMemcpy(d_values, values, v_count*sizeof(int), cudaMemcpyHostToDevice);

                                cudaEventRecord(start);
                                for(k = 0; k<repetitions; k++){
                                        //printf("Calling Kernel\n");
					//cudaEventRecord(st);
                                        cudaMemcpy(d_x, x, r_count*sizeof(float), cudaMemcpyHostToDevice);
					//cudaEventRecord(sp);
					//cudaEventSynchronize(sp);
					//cudaEventElapsedTime(&timing, st, sp);
					
					//printf("Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / timing);

                                        cudaMemcpy(d_y, y, r_count*sizeof(float), cudaMemcpyHostToDevice);


                                        float blocksize = 64;
                                        float blocknum = ceil(r_count/blocksize);
                                        //printf("blocknum: %f\n",blocknum);
                                        //blocknum = 1.0;
                                        //printf("blocknum: %f\n",blocknum);
                                        vector_multiply_unopti<<<blocknum, blocksize>>>(r_count, d_row_ptr, d_col_ind, d_values, d_x, d_y);
					//cudaEventRecord(st);
                                        cudaMemcpy(y, d_y, r_count*sizeof(float), cudaMemcpyDeviceToHost);
					//cudaEventRecord(sp);
					//cudaEventSynchronize(sp);
					//cudaEventElapsedTime(&timing, st, sp);

                                        //printf("Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / timing);
                                        for(i = 0; i<r_count;i++){
                                                x[i] = y[i];
                                                y[i]= 0.0;
                                        }
                                }

                                cudaEventRecord(stop);
                                cudaEventSynchronize(stop);

                                milliseconds = 0;
                                cudaEventElapsedTime(&milliseconds, start, stop);
				
				cudaEventDestroy(st);
                		cudaEventDestroy(sp);

				break;
			case 5:
				//*************************************************************
                                //RUNNING KERNEL WITH DATA TRANSFER OPTIMIZATIONS AND 
				//LOOP UNROLLING
                                //*************************************************************

                                cudaMallocHost(&d_row_ptr, r_count*sizeof(int));
                                cudaMallocHost(&d_col_ind, v_count*sizeof(int));
                                cudaMallocHost(&d_values, v_count*sizeof(int));
                                cudaMallocHost(&d_x, r_count*sizeof(float));
                                cudaMallocHost(&d_y, r_count*sizeof(float));

                                cudaMemcpy(d_row_ptr, row_ptr, r_count*sizeof(int), cudaMemcpyHostToDevice);
                                cudaMemcpy(d_col_ind, col_ind, v_count*sizeof(int), cudaMemcpyHostToDevice);
                                cudaMemcpy(d_values, values, v_count*sizeof(int), cudaMemcpyHostToDevice);

                                cudaEventRecord(start);
                                for(k = 0; k<repetitions; k++){
                                        //printf("Calling Kernel\n");
                                        cudaMemcpy(d_x, x, r_count*sizeof(float), cudaMemcpyHostToDevice);
                                        cudaMemcpy(d_y, y, r_count*sizeof(float), cudaMemcpyHostToDevice);


                                        float blocksize = 64;
                                        float blocknum = ceil(r_count/blocksize);
                                        //printf("blocknum: %f\n",blocknum);
                                        //blocknum = 1.0;
                                        //printf("blocknum: %f\n",blocknum);
                                        vector_multiply_unroll<<<blocknum, blocksize>>>(r_count, d_row_ptr, d_col_ind, d_values, d_x, d_y);
                                        cudaMemcpy(y, d_y, r_count*sizeof(float), cudaMemcpyDeviceToHost);
                                        for(i = 0; i<r_count;i++){
                                                x[i] = y[i];
                                                y[i]= 0.0;
                                        }
                                }

                                cudaEventRecord(stop);
                                cudaEventSynchronize(stop);

                                milliseconds = 0;
                                cudaEventElapsedTime(&milliseconds, start, stop);

				break;
			case 6:
                                //*************************************************************
                                //RUNNING KERNEL WITH DATA TRANSFER OPTIMIZATIONS,
                                //LOOP UNROLLING AND USING SHARED MEMORY
                                //*************************************************************

                                cudaMallocHost(&d_row_ptr, r_count*sizeof(int));
                                cudaMallocHost(&d_col_ind, v_count*sizeof(int));
                                cudaMallocHost(&d_values, v_count*sizeof(int));
                                cudaMallocHost(&d_x, r_count*sizeof(float));
                                cudaMallocHost(&d_y, r_count*sizeof(float));

                                cudaMemcpy(d_row_ptr, row_ptr, r_count*sizeof(int), cudaMemcpyHostToDevice);
                                cudaMemcpy(d_col_ind, col_ind, v_count*sizeof(int), cudaMemcpyHostToDevice);
                                cudaMemcpy(d_values, values, v_count*sizeof(int), cudaMemcpyHostToDevice);

                                cudaEventRecord(start);
                                for(k = 0; k<repetitions; k++){
                                        //printf("Calling Kernel\n");
                                        cudaMemcpy(d_x, x, r_count*sizeof(float), cudaMemcpyHostToDevice);
                                        cudaMemcpy(d_y, y, r_count*sizeof(float), cudaMemcpyHostToDevice);


                                        float blocksize = 64;
                                        float blocknum = ceil(r_count/blocksize);
                                        //printf("blocknum: %f\n",blocknum);
                                        //blocknum = 1.0;
                                        //printf("blocknum: %f\n",blocknum);
                                        vector_multiply_shared<<<blocknum, blocksize>>>(r_count, d_row_ptr, d_col_ind, d_values, d_x, d_y);
                                        cudaMemcpy(y, d_y, r_count*sizeof(float), cudaMemcpyDeviceToHost);
                                        for(i = 0; i<r_count;i++){
                                                x[i] = y[i];
                                                y[i]= 0.0;
                                        }
                                }

                                cudaEventRecord(stop);
                                cudaEventSynchronize(stop);

                                milliseconds = 0;
                                cudaEventElapsedTime(&milliseconds, start, stop);

                                break;
			default:
				//*************************************************************
                                //RUNNING KERNEL WITH NO OPTIMIZATIONS IN DEFAULT CASE
                                //*************************************************************

                                cudaMalloc(&d_row_ptr, r_count*sizeof(int));
                                cudaMalloc(&d_col_ind, v_count*sizeof(int));
                                cudaMalloc(&d_values, v_count*sizeof(int));
                                cudaMalloc(&d_x, r_count*sizeof(float));
                                cudaMalloc(&d_y, r_count*sizeof(float));

                                cudaMemcpy(d_row_ptr, row_ptr, r_count*sizeof(int), cudaMemcpyHostToDevice);
                                cudaMemcpy(d_col_ind, col_ind, v_count*sizeof(int), cudaMemcpyHostToDevice);
                                cudaMemcpy(d_values, values, v_count*sizeof(int), cudaMemcpyHostToDevice);

                                cudaEventRecord(start);
                                for(k = 0; k<repetitions; k++){
                                        //printf("Calling Kernel\n");
                                        cudaMemcpy(d_x, x, r_count*sizeof(float), cudaMemcpyHostToDevice);
                                        cudaMemcpy(d_y, y, r_count*sizeof(float), cudaMemcpyHostToDevice);


                                        float blocksize = 64;
                                        float blocknum = ceil(r_count/blocksize);
                                        //printf("blocknum: %f\n",blocknum);
                                        //blocknum = 1.0;
                                        //printf("blocknum: %f\n",blocknum);
                                        vector_multiply_unopti<<<blocknum, blocksize>>>(r_count, d_row_ptr, d_col_ind, d_values, d_x, d_y);
                                        cudaMemcpy(y, d_y, r_count*sizeof(float), cudaMemcpyDeviceToHost);
                                        for(i = 0; i<r_count;i++){
                                                x[i] = y[i];
                                                y[i]= 0.0;
                                        }
                                }

                                cudaEventRecord(stop);
                                cudaEventSynchronize(stop);

                                milliseconds = 0;
                                cudaEventElapsedTime(&milliseconds, start, stop);
				
		}

		/*cudaMalloc(&d_row_ptr, r_count*sizeof(int));
		cudaMalloc(&d_col_ind, v_count*sizeof(int));
		cudaMalloc(&d_values, v_count*sizeof(int));
		cudaMalloc(&d_x, r_count*sizeof(float));
		cudaMalloc(&d_y, r_count*sizeof(float));*/
		
		/*cudaMallocHost(&d_row_ptr, r_count*sizeof(int));
                cudaMallocHost(&d_col_ind, v_count*sizeof(int));
                cudaMallocHost(&d_values, v_count*sizeof(int));
                cudaMallocHost(&d_x, r_count*sizeof(float));
                cudaMallocHost(&d_y, r_count*sizeof(float));

		cudaMemcpy(d_row_ptr, row_ptr, r_count*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_col_ind, col_ind, v_count*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_values, values, v_count*sizeof(int), cudaMemcpyHostToDevice);
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);	  	
	  	
		cudaEventRecord(start);
	  	for(k = 0; k<repetitions; k++){
			printf("Calling Kernel\n");
			cudaMemcpy(d_x, x, r_count*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_y, y, r_count*sizeof(float), cudaMemcpyHostToDevice);
		
			// kernel call
			float blocksize = 64;
			float blocknum = ceil(r_count/blocksize); //number of threads fixed and equal to row count
			//printf("blocknum: %f\n",blocknum);
			//blocknum = 1.0;
			//printf("blocknum: %f\n",blocknum);
			//vector_multiply_unopti<<<blocknum, blocksize>>>(r_count, d_row_ptr, d_col_ind, d_values, d_x, d_y);
			vector_multiply_unroll<<<blocknum, blocksize>>>(r_count, d_row_ptr, d_col_ind, d_values, d_x, d_y);
			cudaMemcpy(y, d_y, r_count*sizeof(float), cudaMemcpyDeviceToHost);
			for(i = 0; i<r_count;i++){
				x[i] = y[i];
			    	y[i]= 0.0;
			}
		}
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);*/

		//*************************************************************
                //GPU RUN COMPLETED
                //*************************************************************

		//*************************************************************
                //PRINTING RESULTANT MATRIX.
		//ONLY NON ZERO ELEMENTS ARE DISPLAYED.
                //*************************************************************

		int count = 0;
		if(mode == 1){
			
			fprintf(stdout,"Calculated Vector Matrix\n");
		    	for(i = 0; i<r_count;i++){
		    		if(x[i] != 0){
		    		fprintf(stdout,"%.10f\n",x[i]);
		    		count++;
		    		}
		    	}
		    	fprintf(stdout,"count = %d\n", count);

		}
		fprintf(stdout,"time = %f\n", milliseconds);
		//fprintf(stdout,"CPU time = %f\n",cpu_time_used);
		
		//*************************************************************
                //FREE MEMORY
                //*************************************************************
		cudaFree(d_row_ptr);
		cudaFree(d_col_ind);
		cudaFree(d_values);
		cudaFree(d_x);
		cudaFree(d_y);
		cudaEventDestroy(start);
        	cudaEventDestroy(stop);
	}
}
