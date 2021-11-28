#include <stdio.h>
#include <cstdlib>
#include <mpi.h>

#define TAG 0

int size;

int rank;


// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from stdin
//
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr)
{
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(0);

    if(rank == 0)
    {

        std::cin>> *n_ptr >> *m_ptr >> *l_ptr;

        int n = *n_ptr, m = *m_ptr, l = *l_ptr;

        *a_mat_ptr = (int *)malloc(sizeof(int) * n * m);
        *b_mat_ptr = (int *)malloc(sizeof(int) * m * l);


        int *a_mat = *a_mat_ptr;
        int *b_mat = *b_mat_ptr;

        for(int i=0; i< n * m ; i++)
            std::cin>> a_mat[i];

        for(int i=0; i< m * l ; i++)
            std::cin>> b_mat[i];

        MPI_Bcast(n_ptr,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(m_ptr,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(l_ptr,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(*a_mat_ptr, n * m,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(*b_mat_ptr, m * l,MPI_INT,0,MPI_COMM_WORLD);

    } else
    {
        MPI_Bcast(n_ptr,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(m_ptr,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(l_ptr,1,MPI_INT,0,MPI_COMM_WORLD);
        int n = *n_ptr, m = *m_ptr, l = *l_ptr;
        *a_mat_ptr = (int *)malloc(sizeof(int) * n * m);
        *b_mat_ptr = (int *)malloc(sizeof(int) * m * l);
        MPI_Bcast(*a_mat_ptr, n * m,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(*b_mat_ptr, m * l,MPI_INT,0,MPI_COMM_WORLD);
    }

}

// Just matrix multiplication (your should output the result in this function)
// 
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat)
{
    int *b_mat_trans = (int *)malloc(sizeof(int) * l * m);
    for(int i = 0 ; i < l ; i++)
        for(int j = 0  ; j < m ; j++)
            b_mat_trans[i * m + j] = b_mat[j * l + i];
    if(rank == 0)
    {
        int mat_size = n * l;

        int *recv = (int *)malloc(sizeof(int) * (size-1) * (mat_size / size + 1));
        MPI_Request *requests = (MPI_Request *)malloc(sizeof(MPI_Request) * (size-1));
        MPI_Status *status = (MPI_Status *)malloc(sizeof(MPI_Status) * (size-1));
        
        int *start = (int *)malloc(sizeof(int) * (size-1));
        int *len = (int *)malloc(sizeof(int) * (size-1));
        int node_rank;

        for(int i=0; i<size-1 ; i++)
        {
            node_rank = i+1;
            start[i] = mat_size / size * node_rank;
            len[i] = mat_size / size;
            if(i < mat_size % size)
            {
                len[i] += node_rank;
                start[i] += node_rank;
            }
            else
                start[i] += mat_size % size;
            
            MPI_Status status;
            MPI_Irecv(recv + i*(mat_size / size + 1), len[i], MPI_INTEGER, node_rank,TAG, MPI_COMM_WORLD, &requests[i]);
            
        }


        int *ans_mat = (int *)malloc(sizeof(int *) * mat_size);
        
        int root_len = mat_size / size;
        if(mat_size % size)
            root_len += 1;
        
        memset(ans_mat, 0, root_len*sizeof(int));

        for(int i=0 ; i<root_len ; i++)
        {
            for(int j=0 ; j<m ; j++)
                ans_mat[i] += a_mat[i / l * m + j] * b_mat_trans[(i % l)*m + j]; // b_mat[i % l + l*j]
        }


        MPI_Waitall(size-1, requests, status);

        for(int i=0; i<size-1 ; i++)
            memcpy(ans_mat+start[i], recv + i*(n * l / size + 1), sizeof(int)*len[i]);
        
        free(recv);
        free(requests);
        free(start);
        free(len);

        
        
        /*int mat_size = n * l;
        int *ans_mat = (int *)malloc(sizeof(int *) * mat_size);
        
        int root_len = mat_size / size;
        if(mat_size % size)
            root_len += 1;
        
        memset(ans_mat, 0, root_len*sizeof(int));

        for(int i=0 ; i<root_len ; i++)
        {
            for(int j=0 ; j<m ; j++)
                ans_mat[i] += a_mat[i / l * m + j] * b_mat_trans[(i % l)*m + j]; // b_mat[i % l + l*j]
        }

        int *recv = (int *)malloc(sizeof(int) * (mat_size / size + 1));
        int len, start, node_rank;
        for(int i=0; i<size-1 ; i++)
        {
            node_rank = i+1;
            start = mat_size / size * node_rank;
            len = mat_size / size;
            if(i < mat_size % size)
            {
                len += node_rank;
                start += node_rank;
            }
            else
                start += mat_size % size;
            
            MPI_Status status;
            MPI_Recv(recv, len, MPI_INTEGER, node_rank, TAG, MPI_COMM_WORLD, &status);
            memcpy(ans_mat+start, recv, sizeof(int)*len);
            
        }
        
        free(recv);*/


        for(int i=0 ; i<mat_size ; i++)
        {
            printf("%d ",ans_mat[i]);
            if((i+1)%l!=0)
                continue;
            printf("\n");

        }
            
        free(ans_mat);
    } else
    {
        int *ans_local;
        int mat_size = n * l;
        int len = mat_size / size;
        int start = mat_size / size * rank;
        if(rank < mat_size % size)
        {
            len += 1;
            start += rank;
        }
        else
            start += mat_size % size;
        
        ans_local = (int *)malloc(sizeof(int) * len);
        
        memset(ans_local, 0, len*sizeof(int));

        for(int i=0 ; i<len ; i++)
        {
            for(int j=0 ; j<m ; j++)
                ans_local[i] += a_mat[(start+i) / l * m + j] * b_mat_trans[((start+i) % l)*m + j]; // b_mat[(start+i) % l + l*j]
            
        }


        // MPI_Request request;
        // MPI_Status status;

        // MPI_Isend(ans_local, len, MPI_INTEGER, 0, TAG, MPI_COMM_WORLD, &request);
        // MPI_Waitall(1, &request, &status);

        MPI_Send(ans_local, len, MPI_INTEGER, 0, TAG, MPI_COMM_WORLD);
        free(ans_local);
    }

    free(b_mat_trans);

}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat)
{
    free(a_mat);
    free(b_mat);
}