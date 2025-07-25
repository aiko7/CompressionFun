FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04:20211001.v1

# Remove old NVIDIA GPG keys
RUN apt-key del 7fa2af80 || true
RUN apt-key del A4B469963BF863CC || true

# Remove old NVIDIA repository sources to prevent conflicts
RUN rm -f /etc/apt/sources.list.d/*nvidia* /etc/apt/sources.list.d/*cuda*

# Install prerequisites first (without updating from potentially broken repos)
RUN apt-get update --allow-releaseinfo-change && apt-get install -y wget gnupg ca-certificates software-properties-common

# Download and install the new CUDA keyring package
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm cuda-keyring_1.0-1_all.deb

# Alternative: Add the keys manually if keyring doesn't work
RUN wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | apt-key add - && \
    wget -qO - https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add -

# Add deadsnakes PPA for Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa -y

# Update package lists with the new keys and PPA
RUN apt-get update

# Install Python 3.11 and other dependencies
RUN apt-get install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils curl git

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# CRITICAL: Remove or rename the old python binaries first AND miniconda
RUN rm -f /usr/bin/python /usr/bin/python3 || true
RUN rm -f /opt/miniconda/bin/python /opt/miniconda/bin/python3 || true

# Create symbolic links directly (more reliable than update-alternatives)
RUN ln -s /usr/bin/python3.11 /usr/bin/python
RUN ln -s /usr/bin/python3.11 /usr/bin/python3

# CRITICAL: Update PATH to prioritize our Python 3.11 over miniconda
ENV PATH="/usr/bin:/usr/local/bin:$PATH"

# Verify Python version immediately
RUN python --version && python3 --version

# Install Poetry using the correct Python
RUN python -m pip install poetry

# Add Poetry to PATH (but after our python path)
ENV PATH="/usr/bin:/usr/local/bin:/root/.local/bin:$PATH"

# Configure Poetry to not create virtual environments
ENV POETRY_VIRTUALENVS_CREATE=false
RUN poetry config virtualenvs.create false

# CRITICAL: Force Poetry to use Python 3.11 explicitly
ENV POETRY_PYTHON_PATH=/usr/bin/python3.11

WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Clear any Poetry cache that might have old Python version info
RUN poetry cache clear --all pypi -n || true

# Regenerate lock file with Python 3.11 to ensure compatibility
RUN poetry lock 

# Install project dependencies (Poetry will now use Python 3.11)
RUN poetry install --no-interaction --no-ansi --no-root

# Verify installations
RUN python -c "import sys; print(f'Python version: {sys.version}')"
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
RUN python -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"

# Set final environment variables
ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages:$PYTHONPATH

#COPY . .
#CMD ["poetry", "run", "python", "main.py"]