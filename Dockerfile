FROM python:3.10.6-slim-bullseye

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set environment variables for the user name, group name and user home
# directory to be used further down
ARG USER="bioc"
ARG GROUP="bioc"
ARG WORKDIR="/home/${USER}"

# Create user home, user and group & make user and group the owner of the user
# home
RUN mkdir -p $WORKDIR \
&& groupadd -r $GROUP \
&& useradd --no-log-init -r -g $GROUP $USER \
&& chown -R ${USER}:${GROUP} $WORKDIR \
&& chmod 700 $WORKDIR

# Set the container user to the new user
USER $USER

# Make sure the location where Pip installs console scripts/executables is
# available in the $PATH variable so that the container's operating sytem is
# able to locate them
ENV PATH="${WORKDIR}/.local/bin:${PATH}"

# Set the working directory to the user's home
WORKDIR $WORKDIR

# Copy the entire content of the current directory to the working directory and
# make sure the copied files are owned by the container user and corresponding
# group
COPY --chown=${USER}:${GROUP} . $WORKDIR

# Install the required dependencies
#COPY requirements.txt .
#RUN python -m pip install -r requirements.txt

# install app
RUN pip install -e .

# Set default entry point for containers created from the image
ENTRYPOINT ["NAME_OF_YOUR_TOOL_EXECUTABLE"]

# Set default command-line arguments
CMD ["--help"]