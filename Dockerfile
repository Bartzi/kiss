FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
	git \
	software-properties-common \
	pkg-config \
	unzip \
    zsh \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libgl1 \
    wget

RUN pip3 install cython

ARG UNAME=user
ARG UID=1000
ARG GID=100

RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/zsh $UNAME

RUN mkdir /data
ARG BASE=/app
RUN mkdir -p ${BASE}

ARG MPIDIR=/opt/openmpi
RUN mkdir -p ${MPIDIR}
WORKDIR ${MPIDIR}
RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.5.tar.gz && tar xf openmpi-4.0.5.tar.gz
WORKDIR ${MPIDIR}/openmpi-4.0.5
RUN ./configure --with-cuda
RUN make -j ${nproc}
RUN make install
RUN ldconfig

RUN pip3 install cupy-cuda110 mpi4py

COPY requirements_docker.txt ${BASE}/requirements.txt

WORKDIR ${BASE}
RUN pip3 install -r requirements.txt

USER $UNAME
RUN git clone https://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh
RUN cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc

CMD ["/bin/zsh"]
