FROM debian:stable-slim

# Install baseline
RUN apt update && \
    apt install -y git \
                   cmake \
                   g++ \
                   gfortran \
                   libboost-all-dev && \
    apt clean && apt autoclean

COPY . /tmp/pysa-stern

# Install stern
RUN cmake -B /tmp/pysa-stern/build \
          -S /tmp/pysa-stern \
          -DCMAKE_Fortran_COMPILER=gfortran \
          -DCMAKE_BUILD_TYPE=Release \
          -DMPI=OFF \
          -DPYMODULES=OFF && \
    cmake --build /tmp/pysa-stern/build --target install -j && \
    rm -fr /tmp/pysa-stern

ENTRYPOINT [ "/usr/local/bin/sternx" ]
