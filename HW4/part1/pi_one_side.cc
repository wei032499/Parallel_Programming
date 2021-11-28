#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

#define TAG 0

static double UINT_MAX_HALF = 2147483647.5;
static unsigned int seed = 0;

inline unsigned int xorshift32(unsigned int x)
{
    /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

inline long long int toss(long long int number_of_tosses)
{

    long long int number_in_circle = 0;

    double x, y, distance_squared;
    unsigned int r1 = rand_r(&seed), r2 = rand_r(&seed);
    for (long long int i = 0; i < number_of_tosses; ++i)
    {

        x = ((double)r1 - UINT_MAX_HALF) / UINT_MAX_HALF;
        y = ((double)r2 - UINT_MAX_HALF) / UINT_MAX_HALF;

        r1 = xorshift32(r1);
        r2 = xorshift32(r2);

        distance_squared = x * x + y * y;
        if (distance_squared <= 1)
            number_in_circle++;
    }

    return number_in_circle;
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    seed = world_rank*time(NULL);

    int count = toss(tosses/world_size);

    if (world_rank == 0)
    {
       int *count_recv;
       MPI_Alloc_mem(world_size * sizeof(int), MPI_INFO_NULL, &count_recv);

       for (int i = 0; i < world_size; i++)
          count_recv[i] = 0;

       // Create a window. Set the displacement unit to sizeof(int) to simplify
       // the addressing at the originator processes
       MPI_Win_create(count_recv, world_size * sizeof(int), sizeof(int), MPI_INFO_NULL,
          MPI_COMM_WORLD, &win);

       int ready = 0;
       while (!ready)
       {
          MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
          ready = 1;
          for(int i=1; i<world_size; i++)
            if(count_recv[i] == 0)
            {
                ready = 0;
                break;
            }
          MPI_Win_unlock(0, win);
       }

       for(int i=1; i<world_size; i++)
            count += count_recv[i];

       // Free the allocated memory
       MPI_Free_mem(count_recv);
    }
    else
    {
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

       // Register with the master
       MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
       MPI_Put(&count, 1, MPI_INT, 0, world_rank, 1, MPI_INT, win);
       MPI_Win_unlock(0, win);
    }

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        pi_result = (double)4 * count / tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}