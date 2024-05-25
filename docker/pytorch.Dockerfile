FROM nvcr.io/nvidia/pytorch:23.05-py3


# env config
ENV LANG=C.UTF-8
ENV PYTHON_VERSION=3.8

USER root

# base config
SHELL ["/bin/bash", "-c"]
COPY slogan /etc
RUN cat /etc/slogan >> /etc/bash.bashrc
RUN source /etc/bash.bashrc
#RUN chmod a+rwx /etc/bash.bashrc


# change timezone
ARG TZ="Asia/Shanghai"
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

# copy config to docker
COPY set_root_pw.sh run_ssh.sh init_ssh.sh supervisord.conf /
COPY .bashrc $HOME/

# Update CUDA signing key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# base dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
            supervisor \
            tree \
            lrzsz \
            default-libmysqlclient-dev  \
            libpq-dev \
            sasl2-bin \
            libsasl2-2 \
            libsasl2-dev \
            libsasl2-modules \
            openssh-server \
            pwgen \
            make \
            net-tools \
            npm && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# update pip
RUN pip install --upgrade pip

# extra python dependencies
RUN pip install --no-cache-dir \
        pybind11 \
        thrift \
        sasl \
        thrift_sasl \
        smart_open==2.0.0 \
        SQLAlchemy \
        mysqlclient \
        psycopg2 \
        pyhive \
        elasticsearch \
        redis \
        redis-py-cluster \
        shapely \
        openpyxl \
        seaborn \
        pandarallel \
        minio \
        ujson \
        pyarrow\
        fastparquet \
        clickhouse_driver \
        dask[dataframe] \
        tqdm \
        concurrent_log_handler \
        aredis \
        oss2 \
        loguru \
        transformers==4.25.1 \
        onnx==1.14.0 \
        onnxruntime-gpu==1.14.0 \
        pycuda==2022.2.2 \
        lmdb==1.4.1 \
        torch_scatter==2.1.1 \
        torch_sparse==0.6.17 \
        torch_geometric==2.3.1 \
        tensorboardX==2.5.1 \
        ogb==1.3.6 \
        rdkit_pypi==2022.9.5 \
        dgl==1.1.2


# deal with vim and matplotlib Mojibake
COPY simhei.ttf /usr/local/lib/python${PYTHON_VERSION}/dist-packages/matplotlib/mpl-data/fonts/ttf/
RUN echo "set encoding=utf-8 nobomb" >> /etc/vim/vimrc && \
    echo "set termencoding=utf-8" >> /etc/vim/vimrc && \
    echo "set fileencodings=utf-8,gbk,utf-16le,cp1252,iso-8859-15,ucs-bom" >> /etc/vim/vimrc && \
    echo "set fileformats=unix,dos,mac" >> /etc/vim/vimrc && \
    rm -rf /root/.cache/matplotlib


# SSH config
RUN mkdir /var/run/sshd && \
    sed -i "s/.*UsePrivilegeSeparation.*/UsePrivilegeSeparation no/g" /etc/ssh/sshd_config && \
    sed -i "s/.*UsePAM.*/UsePAM no/g" /etc/ssh/sshd_config && \
    sed -i "s/.*PermitRootLogin.*/PermitRootLogin yes/g" /etc/ssh/sshd_config && \
    sed -i "s/.*PasswordAuthentication.*/PasswordAuthentication yes/g" /etc/ssh/sshd_config && \
    chmod +x /*.sh && sed -i -e 's/\r$//' /*.sh
ENV AUTHORIZED_KEYS **None**
EXPOSE 22

# deal with vim and matplotlib Mojibake
COPY simhei.ttf /opt/conda/lib/python${PYTHON_VERSION}/site-packages/matplotlib/mpl-data/fonts/ttf/
RUN echo -e "set encoding=utf-8 nobomb\nset termencoding=utf-8\nset fileencodings=utf-8,gbk,utf-16le,cp1252,iso-8859-15,ucs-bom\nset fileformats=unix,dos,mac" >> /etc/vim/vimrc && \
    rm -rf /root/.cache/matplotlib

# supervisor start
CMD /usr/bin/supervisord -c /supervisord.conf
