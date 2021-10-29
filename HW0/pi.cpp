#include <iostream>
#include <random>
using namespace std;
int main()
{
    int number_of_tosses = 9999999, number_in_circle = 0, toss;
    double x, y, pi_estimate, distance_squared;

    cout << "Number of tosses: " << number_of_tosses << endl;
    cout << "tossing..." << endl;

    random_device rd;
    default_random_engine generator(rd());
    uniform_real_distribution<double> unif(-1.0, 1.0);

    for (toss = 0; toss < number_of_tosses; toss++)
    {
        x = unif(generator);
        y = unif(generator);
        distance_squared = x * x + y * y;
        if (distance_squared <= 1)
            number_in_circle++;
    }
    pi_estimate = 4 * number_in_circle / ((double)number_of_tosses);

    cout << "PI estimate: " << pi_estimate << endl;
}