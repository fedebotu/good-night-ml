# good-night-ml
This project is about sending good night messages just before your loved one(s) goes to sleep, by predicting the time based on the Telegram accesses timestamps and/or last seen time.

## Introduction
### Why this project?
I started this for fun some time ago. After moving to Korea, I and a girl I had a (sort of) relationship with decided to just be friends due to the distance and not to send each other sweet things like the good night message. Since I didn't want to break the deal, I thought: _"Why not let an AI send the good night message? She will receive it and I technically wasn't the one sending it!"_
So, here is how the idea was born. Nothing too complex, just an algorithm collecting the access timestamps, detecting the time when she goes to sleep and wake up, make some inferences with machine learning about tomorrow's time to go to bed, send a message just before that and... voilà! As you can imagine, it took me some time to figure out all the details of this project, but at the end it is kind of working. Was it something really useful? Probably not. But who cares, it was fun!

## Usage
The following `.py` executables can be either used as daemons or executed normally. I recommend trying to first executing them with your account or someone you know (you don't want to send random messages around, do you?) and see how they work out; and finally converting them to system daemons so that they can be completely automatic.

### `data_collector.py`

### `predictor.py`

### `sender.py`


## Adding the services to systemd
In order to add `data_collector.py` and `sender.py` as system daemons, first create a file under `/etc/systemd/system` (the following example is for the `data_collector.py`, then change the names for `sender.py`):
`sudo touch data-collector@goodnightml.service`

Then copy and paste the following inside changing the parameters:
`sudo nano data-collector@goodnightml.service`

```
[Unit]
Description=Data collector for good-night-ml

[Service]
Type=simple
User=[YOUR_USER]
WorkingDirectory=[INSTALLATION_PATH]/good-night-ml
Restart=always
ExecStart=/usr/bin/python3 data_collector.py

[Install]
WantedBy=multi-user.target
```
Try first if the script is working by starting it (if you are testing, remember to change the `good_nighter`!):
`sudo systemctl start data-collector@goodnightml.service`
`sudo systemctl status data-collector@goodnightml.service`
If the status is `active`, it is working then stop the service
`sudo systemctl stop data-collector@goodnightml.service`

Enable the service to start automatically at reboot via the following command:
`sudo systemctl enable data-collector@goodnightml.service`

Now you can easily check the service status by doing:
`sudo service data-collector@goodnightml.service status/start/stop`

## Adding the cron job
The `predictor.py` does not need to always run in the background, so we can add it to crontab and run it as a periodic task: we just need one prediction (or set of predictions, if we are tracking multiple users) about the next time to go to sleep for each day:
`sudo crontab -e`
Add the following at the end of the file with your time (you may follow [this guide](https://ostechnix.com/a-beginners-guide-to-cron-jobs/) as an easy reference)
```0 18 * * * /usr/bin/python3 [YOUR_PATH]/good-night-ml/predictor.py```

## Installing PyTorch on ARM devices (i.e. Raspberry Pi)
Installing PyTorch on my Raspberry Pi 4B was not that easy for me, so I decided to include the instructions for installing it below with some comments. You can find the [original Stackoverflow post](https://stackoverflow.com/questions/62755739/libtorch-on-raspberry-cant-load-pt-file-but-working-on-ubuntu) and credits go to that writer.

### Increase RBPi SWAP

First of all, if you have a Raspberry PI 3 or lower, you need to increase the SWAP, since the build is a RAM eater.

> If you have a RBPi 4 or higher with more than 3GB of RAM, skip this step.

Modify the file `/etc/dphys-swapfile` :

```
CONF_SWAPFILE=2048M
```

Then call the following command to update changes.

```
sudo dphys-swapfile setup
```

### Install base packages

Install the following packages:

```
sudo apt install build-essential make cmake git python3-pip libatlas-base-dev
```

Libtorch needs CMake>=`3.15` to be built properly, check cmake version with  ``cmake --version``

If it's lower than 3.15, follow the following commands to build a newer version and remove the previous one:

```
wget https://github.com/Kitware/CMake/releases/download/v3.18.0-rc1/cmake-3.18.0-rc1.tar.gz
tar -xzf cmake-3.18.0-rc1.tar.gz
cd cmake<version>
mkdir build
cd build
cmake ..
make
sudo make install

sudo apt remove cmake
sudo ln -s /usr/local/bin/cmake /usr/bin/cmake
sudo ldconfig
```

## Building PyTorch from source to get libtorch backend for ARM

> Don't forget to increase the SWAP to 2048M if you don't have 3GB or RAM.

Getting all needed libraries:

```
sudo apt-get update
sudo apt-get install build-essential tk-dev libncurses5-dev libncursesw5-dev libreadline6-dev libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev libexpat1-dev liblzma-dev zlib1g-dev
```

Getting PyTorch sources (*change --branch=release/x.y according to the version you want to install*):

```
git clone --recursive https://github.com/pytorch/pytorch --branch=release/1.7
cd pytorch
```

Init all the submodules :

```
git submodule update --init --recursive
git submodule update --remote third_party/protobuf # To prevent a bug I had
```

Getting all needed libraries:

```
sudo apt-get update
sudo apt-get install build-essential tk-dev libncurses5-dev libncursesw5-dev libreadline6-dev libdb5.3-dev libgdbm-dev libsqlite3-dev libssl-dev libbz2-dev libexpat1-dev liblzma-dev zlib1g-dev
```

Setting up environment variables for the build.

Add the following lines to the `~/.bashrc` file.

```
export NO_CUDA=1
export NO_DISTRIBUTED=1
export NO_MKLDNN=1 
export NO_NNPACK=1
export NO_QNNPACK=1
```

Log in as root, and use the .bashrc file to setup the environment variables

```
sudo su
source /home/<user>/.bashrc
```

Install python dependencies

```
pip3 install setuptools pyyaml numpy
```

Build and install PyTorch, time to grab a :coffee:, it make take a while (took ~ 2 hours on my RPi 4B 4GB ram!)

> Don't forget the -E that forces the environment variables to be used.

```
sudo -E python3 setup.py install
```

Check the installation worked:

```
cd 
python3
import torch
torch.__version__
```

## Building your program with Torch

In your `CMakeLists.txt` :

```
cmake_minimum_required(VERSION 2.6)
project(projectName)

set(CMAKE_PREFIX_PATH "/home/pi/pytorch/torch") # Adding the directory where torch as been installed
set(CMAKE_CXX_STANDARD 14) # C14 required to compile Torch
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0) # Torch is compiled with CXX11_ABI, so your program needs to be also, or you may have conflicts in some libraries (such as GTest for example)

# Specifying we are using pthread for UNIX systems.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread -Wall")

find_package(Torch REQUIRED)

if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

message(STATUS "Pytorch status :")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
message(STATUS "    Torch Flags: ${TORCH_CXX_FLAGS}")

# Program executable
add_executable(projectName <sources>)

target_link_libraries(projectName PRIVATE pthread dl util ${TORCH_LIBRARIES})         
```
