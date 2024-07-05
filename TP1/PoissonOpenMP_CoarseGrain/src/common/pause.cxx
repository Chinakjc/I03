#include "pause.hxx"

#include <thread>

void pause(int d) {

   std::this_thread::sleep_for (std::chrono::microseconds(d));

}

