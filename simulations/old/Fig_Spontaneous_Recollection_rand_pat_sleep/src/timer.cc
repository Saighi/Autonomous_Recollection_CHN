#include "timer.hpp"
#include <iostream>
#include <iomanip>

Timer::Timer() : running(false) {}

void Timer::start() {
    start_time = std::chrono::high_resolution_clock::now();
    running = true;
}

void Timer::stop() {
    end_time = std::chrono::high_resolution_clock::now();
    running = false;
}

double Timer::elapsed_nanoseconds() {
    auto end = running ? std::chrono::high_resolution_clock::now() : end_time;
    return std::chrono::duration<double, std::nano>(end - start_time).count();
}

double Timer::elapsed_microseconds() {
    auto end = running ? std::chrono::high_resolution_clock::now() : end_time;
    return std::chrono::duration<double, std::micro>(end - start_time).count();
}

double Timer::elapsed_milliseconds() {
    auto end = running ? std::chrono::high_resolution_clock::now() : end_time;
    return std::chrono::duration<double, std::milli>(end - start_time).count();
}

double Timer::elapsed_seconds() {
    auto end = running ? std::chrono::high_resolution_clock::now() : end_time;
    return std::chrono::duration<double>(end - start_time).count();
}

void Timer::print_elapsed() {
    double elapsed = elapsed_nanoseconds();
    std::string unit = "ns";

    if (elapsed >= 1e9) {
        elapsed = elapsed_seconds();
        unit = "s";
    } else if (elapsed >= 1e6) {
        elapsed = elapsed_milliseconds();
        unit = "ms";
    } else if (elapsed >= 1e3) {
        elapsed = elapsed_microseconds();
        unit = "µs";
    }

    std::cout << std::fixed << std::setprecision(3) 
              << elapsed << " " << unit << std::endl;
}