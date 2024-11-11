REDIS_PORT=6379
CONTAINER_NAME=redis-llm
MAIN_PORT=5000
# REMOVE_PREVIOUS=1

# if [ $REMOVE_PREVIOUS -eq 1 ]; then
#     docker stop $CONTAINER_NAME
#     docker rm $CONTAINER_NAME
# fi
sudo sysctl -w vm.overcommit_memory=1

# first let the redis down

CONTAINER_NAME=$CONTAINER_NAME REDIS_PORT=$REDIS_PORT docker-compose -f config/docker-compose.yml up -d

# stop:
# docker-compose -f config/docker-compose.yml down

# stop one by one:
# docker stop redis-llm # the redis

python3 backend.py --port $MAIN_PORT --redis_port $REDIS_PORT --n_workers 4

CONTAINER_NAME=$CONTAINER_NAME REDIS_PORT=$REDIS_PORT docker-compose -f config/docker-compose.yml down