# Splink Docker Development Environment

This directory contains a Docker Compose configuration for setting up a complete development environment for Splink with PostgreSQL, pgAdmin, and Apache Spark.

## Services

The docker-compose setup includes the following services:

### PostgreSQL Database
- **Container Name**: `splink-db-postgres`
- **Image**: `postgres:18.0`
- **Port**: `5432`
- **Database**: `splink_db`
- **Username**: `splinkognito`
- **Password**: `splink123!`

### pgAdmin
- **Container Name**: `splink-pgadmin`
- **Image**: `dpage/pgadmin4:latest`
- **Port**: `80`
- **Email**: `a@b.com`
- **Password**: `b`
- **Purpose**: Web-based PostgreSQL administration tool

### Apache Spark Master
- **Container Name**: `splink-spark-master`
- **Image**: `apache/spark:3.5.3`
- **Ports**:
  - `8080`: Spark Master Web UI
  - `7077`: Spark Master Port (for worker connections)
- **Purpose**: Spark cluster master node for distributed data processing

### Apache Spark Worker
- **Container Name**: `splink-spark-worker`
- **Image**: `apache/spark:3.5.3`
- **Port**: `8081` (Spark Worker Web UI)
- **Configuration**:
  - Memory: 2GB
  - Cores: 2
- **Purpose**: Spark cluster worker node that executes Spark jobs

## Quick Start

### Starting the Services

From the repository root, run:
```bash
./scripts/postgres_docker/setup.sh
```

Or manually:
```bash
docker compose -f scripts/postgres_docker/docker-compose.yaml up -d
```

### Stopping the Services

From the repository root, run:
```bash
./scripts/postgres_docker/teardown.sh
```

Or manually:
```bash
docker compose -f scripts/postgres_docker/docker-compose.yaml down -v
```

**Note**: The `-v` flag removes volumes. Omit it if you want to preserve data.

## Accessing the Services

### PostgreSQL
Connect to PostgreSQL using any PostgreSQL client:
- **Host**: `localhost`
- **Port**: `5432`
- **Database**: `splink_db`
- **Username**: `splinkognito`
- **Password**: `splink123!`

### pgAdmin
Access the web interface at: `http://localhost:80`
- **Email**: `a@b.com`
- **Password**: `b`

The PostgreSQL server is pre-configured in pgAdmin (see `servers.json`).

### Spark Master Web UI
Access the Spark Master UI at: `http://localhost:8080`

Here you can monitor:
- Connected workers
- Running applications
- Completed jobs
- Resource usage

### Spark Worker Web UI
Access the Spark Worker UI at: `http://localhost:8081`

Here you can monitor:
- Worker status
- Executor information
- Resource usage

## Connecting to Spark from Splink

To use the Spark cluster from Splink, connect to the master at `spark://localhost:7077`:

```python
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from splink.internals.spark.database_api import SparkAPI

conf = SparkConf()
conf.set("spark.master", "spark://localhost:7077")
# Add other configuration as needed

spark = SparkSession.builder.config(conf=conf).getOrCreate()
db_api = SparkAPI(spark_session=spark)
```

## Troubleshooting

### Services Not Starting
Check the logs for any specific service:
```bash
docker logs splink-db-postgres
docker logs splink-pgadmin
docker logs splink-spark-master
docker logs splink-spark-worker
```

### Port Conflicts
If ports are already in use, you can modify the port mappings in `docker-compose.yaml`. Change the left side of the port mapping (host port) while keeping the right side (container port) unchanged.

For example, to use port 8082 instead of 8080 for Spark Master:
```yaml
ports:
  - 8082:8080  # Changed from 8080:8080
```

### Spark Worker Not Connecting
Ensure the Spark master is running before the worker starts. The `depends_on` directive should handle this, but you can verify by checking the worker logs for connection messages.

## Development Notes

- The PostgreSQL database uses persistent volumes, so data is preserved between container restarts (unless you use `down -v`).
- Spark is configured with minimal security for local development. **Do not use this configuration in production**.
- The Spark worker is configured with 2GB memory and 2 cores, which is suitable for development. Adjust these values in `docker-compose.yaml` if needed for your use case.
