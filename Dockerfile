FROM gcc:4.9.4
COPY . /usr/src/linpack
WORKDIR /usr/src/linpack
RUN gcc -O3 -march=native -pthread -o linpack linpack.c -lm
ENTRYPOINT ["./linpack"]
