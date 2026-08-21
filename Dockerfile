FROM python:3.12-slim

# 推論はすべて別コンテナ (llama-server / comfyui) に HTTP で投げるので、
# このコンテナに GPU も PyTorch も要らない。ここはオーケストレーション専用。
RUN pip install --no-cache-dir \
	jupyter \
	numpy \
	matplotlib \
	tqdm \
	Pillow \
	janome \
	python-Levenshtein \
	openai

RUN mkdir -p /root/.jupyter && touch /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
 echo c.NotebookApp.open_browser = False >> /root/.jupyter/jupyter_notebook_config.py

WORKDIR /mnt
CMD jupyter notebook --allow-root --NotebookApp.token='' --ServerApp.iopub_msg_rate_limit=2000 --ServerApp.rate_limit_window=10
