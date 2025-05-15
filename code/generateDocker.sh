#!/bin/bash -x

worker_file=${1}
shift # shifts all arguments to the left thus removing ${1}
run_mode=${1}
shift # shifts all arguments to the left thus removing ${1}
NR_VIDEOS=${1}
shift
NR_WORKERS=${1}
shift


# --- Specify the hypertune configuration
# hypertune is not used locally
cat > /tmp/hypertune.yml << EOF
  trainingInput:
    hyperparameters:
      goal: MAXIMIZE
      hyperparameterMetricTag: "answer"
      maxTrials: $NR_VIDEOS
      maxParallelTrials: $NR_WORKERS
      maxFailedTrials: 100
      enableTrialEarlyStopping: False
      # --- each of these become an argparse argument
      params:
      - parameterName: seed
        type: INTEGER
        minValue: 0
        maxValue: 52423
EOF


# --- The container configuration
# ENTRYPOINT ["python3", "/workspace/${worker_file}", "--camera=linear_movement"]
cat > Dockerfile <<EOF
EOF
cat > Dockerfile <<EOL
FROM kangweiliao/kubruntu:latest
COPY . /workspace
WORKDIR /workspace
RUN apt-get update && apt-get install -y ffmpeg
RUN python3 -m pip install -r requirements.txt
EOL
if [[ "${run_mode}" == "direct" ]]
then 
cat > Dockerfile <<EOL
ENTRYPOINT ["python3", "/workspace/${worker_file}"]
EOL
fi


if [[ "${run_mode}" == "direct" ]]
then 
  # --- Launches the job locally
  TAG="local/direct"
  VOL="--volume $(pwd):/workspace"
  docker build -f Dockerfile -t "$TAG" "$PWD"
  docker run $VOL $TAG "$@"
else
  TAG="local/hydra_kubric:latest"
  VOL="--volume $(pwd):/workspace"
  docker build -f Dockerfile -t "$TAG" "$PWD"
fi