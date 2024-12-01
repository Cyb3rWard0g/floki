# Installation

## Install Floki

!!! info
    make sure you have Python already installed. `Python >=3.9`

### As a Python package using Pip

```bash
pip install floki-ai
```

### Remotely from GitHub

```bash
pip install git+https://github.com/Cyb3rWard0g/floki.git
```

### From source with `poetry`:

```bash
git clone https://github.com/Cyb3rWard0g/floki

cd floki

poetry install
```

## Install Dapr CLI

Install the Dapr CLI to manage Dapr-related tasks like running applications with sidecars, viewing logs, and launching the Dapr dashboard. It works seamlessly with both self-hosted and Kubernetes environments. For a complete step-by-step guide, visit the official [Dapr CLI installation page](https://docs.dapr.io/getting-started/install-dapr-cli/).

Verify the CLI is installed by restarting your terminal/command prompt and running the following:

```bash
dapr -h
```

## Initialize Dapr in Local Mode

!!! info
    Make sure you have [Docker](https://docs.docker.com/get-started/get-docker/) already installed. I use [Docker Desktop](https://www.docker.com/products/docker-desktop/).

Initialize Dapr locally to set up a self-hosted environment for development. This process installs Dapr sidecar binaries, runs essential services like Redis (state store and message broker) and Zipkin (observability), and prepares a default components folder. For detailed steps, see the official [guide on initializing Dapr locally](https://docs.dapr.io/getting-started/install-dapr-selfhost/).

To initialize the Dapr control plane containers and create a default configuration file, run:

```bash
dapr init
```

Verify you have container instances with `daprio/dapr`, `openzipkin/zipkin`, and `redis` images running:

```bash
docker ps
```
