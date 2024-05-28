FROM debian:stable-slim

# Install baseline
RUN apt update && apt install -y git \
                                 cmake \
                                 g++ \
                                 python3-dev \
                                 python3-venv 

# Set virtual environment
RUN python3 -m venv /app/venv
ENV VIRTUAL_ENV=/app/venv/
ENV PATH="/app/venv/bin/:$PATH"

# Install pysa-dpll
COPY . /tmp/pysa-dpll
WORKDIR /tmp/pysa-dpll
RUN pip install --no-cache-dir -U .

# Remove temp folder
RUN rm -fr /tmp/pysa-dpll

# Set entrypoint
ENTRYPOINT ["pysa-dpll"]
