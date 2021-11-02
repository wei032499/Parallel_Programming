#include <iostream>
#include <random>
#include <pthread.h>
#include <sstream>

static double UINT_MAX_HALF = 2147483647.5; // UINT_MAX >> 1

using namespace std;

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

    //random generator
    random_device rd;
    default_random_engine generator(rd());
    uniform_int_distribution<unsigned int> unif;

    long long int number_in_circle = 0;

    double x, y, distance_squared;
    unsigned int r1 = xorshift32(unif(generator)), r2 = xorshift32(unif(generator));
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

void *toss_thread(void *data)
{

    long long int number_of_tosses = (long long int)data;

    long long int number_in_circle = toss(number_of_tosses);

    pthread_exit((void *)number_in_circle);
}

int main(int argc, char *argv[])
{

    int number_of_threads;
    long long int number_of_tosses, number_in_circle = 0;
    double pi_estimate;
    pthread_t *threads;

    stringstream ss1(argv[1]), ss2(argv[2]);
    ss1 >> number_of_threads;
    ss2 >> number_of_tosses;

    threads = new pthread_t[number_of_threads];

    long long int number_of_tosses_thread = number_of_tosses / number_of_threads;

    for (int i = 1; i < number_of_threads; i++)
        pthread_create(&(threads[i]), NULL, toss_thread, (void *)(number_of_tosses_thread));

    number_in_circle += toss(number_of_tosses_thread);

    for (int i = 1; i < number_of_threads; i++)
    {
        void *number_in_circle_thread;
        pthread_join(threads[i], &number_in_circle_thread);
        number_in_circle += (long long int)number_in_circle_thread;
    }
    pi_estimate = 4 * number_in_circle / ((double)number_of_tosses);

    delete[] threads;

    cout << pi_estimate << endl;
}