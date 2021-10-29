#include <iostream>
#include <random>
#include <cmath>
#include <pthread.h>
#include <sstream>
// #include <bitset>
// #include <immintrin.h>

using namespace std;

void *toss(void *data)
{
    long long int number_of_tosses = (long long int)data;
    long long int number_in_circle = 0;

    //random generator
    random_device rd;
    default_random_engine generator(rd());
    // uniform_real_distribution<double> unif(-1.0, 1.0);
    uniform_int_distribution<unsigned int> unif;
    unsigned int seed = unif(generator);

    double x, y, distance_squared;
    for (long long int i = 0; i < number_of_tosses; ++i)
    {
        x = ((double)rand_r(&seed) / RAND_MAX) * 2 - 1;
        y = ((double)rand_r(&seed) / RAND_MAX) * 2 - 1;
        distance_squared = x * x + y * y;
        if (distance_squared <= 1)
            number_in_circle++;
    }

    // for (int i = 0; i < ceil(number_of_tosses / 4); i++)
    // {
    //     alignas(32) double x_arr[] = {((double)rand_r(&seed) / RAND_MAX) * 2 - 1, ((double)rand_r(&seed) / RAND_MAX) * 2 - 1, ((double)rand_r(&seed) / RAND_MAX) * 2 - 1, ((double)rand_r(&seed) / RAND_MAX) * 2 - 1},
    //                        y_arr[] = {((double)rand_r(&seed) / RAND_MAX) * 2 - 1, ((double)rand_r(&seed) / RAND_MAX) * 2 - 1, ((double)rand_r(&seed) / RAND_MAX) * 2 - 1, ((double)rand_r(&seed) / RAND_MAX) * 2 - 1};
    //     // __m256d x = _mm256_set_pd(((double)rand_r(&seed) / RAND_MAX) * 2 - 1, ((double)rand_r(&seed) / RAND_MAX) * 2 - 1, ((double)rand_r(&seed) / RAND_MAX) * 2 - 1, ((double)rand_r(&seed) / RAND_MAX) * 2 - 1);
    //     // __m256d y = _mm256_set_pd(((double)rand_r(&seed) / RAND_MAX) * 2 - 1, ((double)rand_r(&seed) / RAND_MAX) * 2 - 1, ((double)rand_r(&seed) / RAND_MAX) * 2 - 1, ((double)rand_r(&seed) / RAND_MAX) * 2 - 1);
    //     __m256d x = _mm256_load_pd(x_arr);
    //     __m256d y = _mm256_load_pd(y_arr);
    //     __m256d x_square = _mm256_mul_pd(x, x);
    //     __m256d y_square = _mm256_mul_pd(y, y);
    //     __m256d distance_squared = _mm256_add_pd(x_square, y_square);
    //     __m256d compare = _mm256_cmp_pd(distance_squared, _mm256_set_pd(1, 1, 1, 1), _CMP_LE_OS);
    //     // cout << &x << " " << &y << endl;

    //     bitset<4> bs(_mm256_movemask_pd(compare));
    //     number_in_circle += bs.count();
    // }

    pthread_exit((void *)number_in_circle);
}

int main(int argc, char *argv[])
{

    int number_of_threads;
    long long int number_of_tosses, tosses_left, number_in_circle = 0;
    double pi_estimate;
    pthread_t *threads;

    stringstream ss;
    ss.str(argv[1]);
    ss >> number_of_threads;
    ss.clear();
    ss.str(argv[2]);
    ss >> number_of_tosses;

    threads = new pthread_t[number_of_threads];
    tosses_left = number_of_tosses;

    for (int i = 0; i < number_of_threads; i++)
    {
        long long int number_of_tosses_thread = ceil(tosses_left / (number_of_threads - i));
        tosses_left -= number_of_tosses_thread;
        pthread_create(&(threads[i]), NULL, toss, (void *)number_of_tosses_thread);
    }
    for (int i = 0; i < number_of_threads; i++)
    {
        void *number_in_circle_thread;
        pthread_join(threads[i], &number_in_circle_thread);
        number_in_circle += (long long int)number_in_circle_thread;
    }
    delete[] threads;
    pi_estimate = 4 * number_in_circle / ((double)number_of_tosses);

    cout << "PI estimate: " << pi_estimate << endl;
}