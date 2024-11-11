# pdf2lec_llm
Project for CS194(294)-196, UC Berkeley, FA24. A slideshow to lecture generating system utilizing LLM agents

# Dependencies
Please install the following dependencies first.
```bash
pip3 install -r requirements.txt
```
Also, ensure that you have installed **redis**, **docker**, and **docker-compose**.

The machine for writing the code is `Ubuntu 24.04.1 LTS`.

# Run
Please follow these steps.
1. Set the configuration in `config/redis.conf` and `config/docker-compose.yml`.
   Now we only run the **Redis database** which stores the task status by docker-compose.

2. Run the following command to start the Redis database (just for rn, maybe adding the backend server later).
   ```bash
   docker-compose -f config/docker-compose.yml up -d
   ```

3. Run the following command to start the backend server.
   ```bash
   python3 backend.py --port 5000 --redis_port 6379
   ```

4. To shut down the backend server, we can just press `Ctrl+C`.
5. To shut down the Redis database, we can run the following command.
   ```bash
   docker-compose -f config/docker-compose.yml down
   ```
These can also be directly run in one script.
```bash
. run.sh
```

# Test
We can test the backend server APIs by **changing and running** the following command.
```bash
. test_api.sh
```
