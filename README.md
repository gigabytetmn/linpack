# linpack

This repository contains an implementation of the Linpack benchmark obtained from [netlib](http://www.netlib.org/benchmark/linpackc.new). It has been lightly modified to not expect any keyboard input.

To run the benchmark:

- `sudo docker run -it --rm elreyes/linpack`

The benchmark results will be printed on the console.

By default the benchmark starts one worker per CPU available to the process
(using its CPU affinity where supported). Each worker runs an independent
calculation with its own matrix and workspace. `KFLOPS` is aggregate throughput
across all workers; `Reps` is the number of repetitions performed by each
worker. The reported memory requirement is the total for all worker workspaces.

It is possible to customize the array size used by the benchmark code (size
is 200 by default) and the number of workers. For example, to use four workers:

`sudo docker run -it --rm -e LINPACK_ARRAY_SIZE=600 -e LINPACK_THREADS=4 elreyes/linpack`

`LINPACK_THREADS` defaults to the number of available CPUs. Setting it lets you
limit or explicitly choose the worker count. Each extra worker requires an
additional private workspace, so memory use grows linearly with this value.

### Building the Image Locally

Alterntively, you can clone this repository and build your own Docker image locally from scratch. To do so, follow these steps:

- Clone this repository (requires git to be installed in your system... Google for your particular OS/distro)
  - `git clone https://github.com/ereyes01/linpack.git`

- Build the Docker image
  - `cd linpack`
  - `sudo docker build -t linpack ./`
  
- Run the benchmark
  - `sudo docker run -it --rm linpack`
  
