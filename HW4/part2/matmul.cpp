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
    for(int i = 0 ; i < m ; i++)
        for(int j = 0  ; j < l ; j++)
            b_mat_trans[j * m + i] = b_mat[i * l + j];
    
    int *ans_local;
    int mat_size = n * l;
    int slice = mat_size / size;
    int array_size = mat_size / size + 1;
    int len = mat_size / size;
    int start = mat_size / size * rank;
    if(rank < mat_size % size)
    {
        len += 1;
        start += rank;
    }
    else
        start += mat_size % size;
    
    ans_local = (int *)malloc(sizeof(int) * array_size);
    
    memset(ans_local, 0, len*sizeof(int));

    for(int i=0 ; i<len ; i++)
    {
        int row = (start+i) / l * m;
        int col = ((start+i) % l) * m;
        for(int j=0 ; j<m ; j++)
            ans_local[i] += a_mat[row + j] * b_mat_trans[col + j];
    }
        

    if(rank == 0)
    {

        int *ans_mat = (int *)malloc(sizeof(int *) * size * array_size);

        MPI_Gather(ans_local, array_size, MPI_INTEGER, ans_mat, array_size, MPI_INTEGER, 0, MPI_COMM_WORLD);

        for(int i=0 ; i<size ; i++)
        {
            int index = array_size * i;
            for(int j=0 ; j<slice ; j++)
                printf("%d ",ans_mat[index + j]);
            if(i < mat_size % size)
                printf("%d ",ans_mat[array_size*i + slice]);
            printf("\n");
        }


        free(ans_mat);
    } else
        MPI_Gather(ans_local, array_size, MPI_INTEGER, NULL, 0, NULL, 0, MPI_COMM_WORLD);

    free(ans_local);
    free(b_mat_trans);

}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat)
{
    free(a_mat);
    free(b_mat);
}