cmake_minimum_required(VERSION 2.8)

project(cutils C)

set(CMAKE_C_STANDARD 11)
find_package(Threads REQUIRED)
if (NOT MSVC)
  set(CMAKE_C_FLAGS "-Wall -Wextra -pedantic")
endif()

set(PROGRAMS
  linpack
)

foreach(program ${PROGRAMS})
  add_executable(${program} ${program}.c)
endforeach(program)

target_link_libraries(linpack ${CMAKE_THREAD_LIBS_INIT} m)
