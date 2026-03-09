FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel
ENV TORCH_CUDA_ARCH_LIST "8.6"
RUN apt update && apt upgrade -y && apt -y autoremove
RUN DEBIAN_FRONTEND=noninteractive apt install -y git libaio-dev libmpich-dev build-essential python3-venv

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install \
	openai \
	numpy \
	janome \
	python-Levenshtein \
	matplotlib \
	compel \
	diffusers \
	accelerate \
	transformers \
	tqdm

WORKDIR /mnt
CMD python main.py
