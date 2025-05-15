FROM kangweiliao/kubruntu:latest
COPY . /workspace
WORKDIR /workspace
RUN apt-get update && apt-get install -y ffmpeg
RUN python3 -m pip install -r requirements.txt
