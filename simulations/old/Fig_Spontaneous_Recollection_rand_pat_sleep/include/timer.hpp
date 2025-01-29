#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <string>

class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
    bool running;

public:
    // Constructor
    Timer();

    // Core timer operations
    void start();
    void stop();

    // Get elapsed time in different units
    double elapsed_nanoseconds();
    double elapsed_microseconds();
    double elapsed_milliseconds();
    double elapsed_seconds();

    // Utility functions
    void print_elapsed();
};

#endif // TIMER_HPP