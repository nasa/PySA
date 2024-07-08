FROM debian:stable-slim

# Install baseline
RUN apt update && apt install -y git \
                                 cmake \
                                 g++ \
                                 gfortran \
                                 libboost-all-dev

# Install pysa-dpll
COPY . /tmp/pysa-stern
WORKDIR /tmp/pysa-stern

RUN mkdir build && cmake -B ./build -S . -DCMAKE_Fortran_COMPILER=gfortran \
    -DCMAKE_BUILD_TYPE=Release -DMPI=OFF -DPYMODULES=OFF \
    && cmake --build ./build --target sternx \
    && cmake --build ./build --target mldpt

CMD [ "./build/src/sternx", "./tests/mnci_n64_t3_0_mld.txt" ,"--sterncpp", "--max_iters", "100000"]