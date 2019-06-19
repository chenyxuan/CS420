# Traffic Signal Control

## Introduction

Traffic signal control problem, one of the biggest urban problems, is drawing increasing attention in recent years. Recent advances are enabled by large-scale real-time traffic data collected from various sources such as vehicle tracking device, location-based mobile services, and road surveillance cameras through advanced sensing technology and web infrastructure. Traffic signal control is interesting but complex because of the dynamics of traffic flow and the difficulties to coordinate thousands of traffic signals. Reinforcement learning becomes one of the promising approaches to optimize traffic signal plans, as shown in several recent studies. At the same time, traffic signal control is also one of the major real-world application scenarios for reinforcement learning.

There is a online game which may help you to get better understanding of this urban problems: http://www.tscc2050.com/

## Problem Definition

This competition provides the traffic flow data, traffic scenario data (single intersection), the simulator and a basic signal plan to the participants. During the first stage of the competition, the participants are required to provide the traffic signal plan for that traffic scenario. During the second stage, the participants are required to submit the model that generates the signal plan. The final result is the travel time of vehicles based on the traffic signal plans.

## Evaluation

### Evaluation metric: average travel time

For those vehicles which complete the trip from the starting road to the ending road, we define the timestamp it reaches the destination (enter the last road segment) as t_e. Then the travel time is defined as the time difference between its starting time t_s and ending time t_e and it is calculated by: t = t_e - t_s.

As for the rest of the vehicles which have not reached the destination when the system time is over at timestamp T_max, their travel time is calculated as: t = T_max - t_s.

The metric Travel Time is calculated as the average travel time of all vehicles: T = \Sum t / n. where n is the number of vehicles.

Obviously, the objective of this problem is to seek a better signal timing plan to minimize the average travel time T.

### IMPORTANT: In this project we have only one scenario

## CityFlow

In this project, we use [CityFlow](https://cityflow-project.github.io) as the traffic simulator, which is finished majorly by APEX Lab.

And you can turn to [official documents](https://cityflow.readthedocs.io/en/latest/index.html) for more specific details. However, I will provide some important information below.

### Installation

There are two ways to use CityFlow.

#### Docker
The easiest way to use CityFlow is via docker.

```bash
docker build -t cityflow -f cityflow.dockerfile --rm .
```

This will create a docker image with tag ```cityflow```
This docker contains ```C++ related dependencies```, ```python3.6``` and ```flask```

```bash
docker run -it --rm -v .:/mnt cityflow /bin/bash
```

Create and start a container, CityFlow is out-of-the-box along with miniconda with python3.6.

```python
import cityflow
eng = cityflow.Engine
```

#### Build From Source

If you want to get nightly version of CityFlow or running on native system, you can build CityFlow from source. **Currently, we only support building on Unix systems.*- This guide is based on Ubuntu 16.04.

1. Check that you have python 3.6 installed. Other version of python might work, however, we only tested on python with version >= 3.6.

2. Install cpp dependencies

    ```bash
    apt update && apt-get install -y build-essential libboost-all-dev cmake
    ```

3. Clone CityFlow project from github.

    ```bash
    git clone --recursive git@github.com:cityflow-project/CityFlow.git
    ```

    Notice that CityFlow uses pybind11 to integrate C++ code with python, the repo has pybind11 as a submodule, please use ```--recursive``` to clone all codes.

4. Go to CityFlow project’s root directory and run

    ```bash
    pip install .
    ```

5. Wait for installation to complete and CityFlow should be successfully installed.

    ```python
    import cityflow
    eng = cityflow.Engine
    ```

*CityFlow also works well on Mac OS if you have installed boost(version >= 1.50), cmake and C++ compiler. We provide two binary ```.so``` file: ```cityflow.cpython-36m-darwin.so``` is the cityflow library for Mac OS under Python3.6, while ```cityflow.cpython-36m-x86_64-linux-gnu.so``` for Linux. Unfortunately, Windows user may only use WSL, Docker or Virtual Machine to use it. Honestly, our code is compatible with Windows because of our minimal dependence. But currently we have no time to test the correction on Windows. We welcome anyone who can help us to test and fix potential bugs.*

### Create Engine

``` bash
import cityflow
eng = cityflow.Engine(config_path, thread_num=1)
```

- ```config_path```: path for config file.
- ```thread_num```: number of working threads.

Here is a sample config file, and the meanings of parameters are shown on the [docs](https://cityflow.readthedocs.io/en/latest/start.html).

```json
{
    "interval": 1.0,
    "warning": true,
    "seed": 0,
    "dir": "data/",
    "roadnetFile": "roadnet/testcase_roadnet_3x3.json",
    "flowFile": "flow/testcase_flow_3x3.json",
    "rlTrafficLight": false,
    "laneChange": false,
    "saveReplay": true,
    "roadnetLogFile": "frontend/web/testcase_roadnet_3x3.json",
    "replayLogFile": "frontend/web/testcase_replay_3x3.txt"
}
```

### Simulation
To simulate one step, simply call ```eng.next_step()```

```python
eng.next_step()
```

### Data Access API

```get_vehicle_count()```:

- Get number of total running vehicles.

```get_lane_vehicle_count()```:

- Get number of running vehicles on each lane.

```get_lane_waiting_vehicle_count()```:

- Get number of waiting vehicles on each lane. Currently, vehicles with speed less than 0.1m/s is considered as waiting.

```get_lane_waiting_vehicle_count()```:

- Get vehicle ids on each lane.

```get_vehicle_speed()```:

- Get speed of each vehicle.

```get_vehicle_distance()```:

- Get distance travelled on current lane of each vehicle.

```get_current_time()```:

- Get simulation time (in seconds)

```getAverageTravelTime()```:

- Get average travel time (in seconds), which represent the performance of your algorithm.

### Control API

```set_tl_phase(intersection_id, phase_id)```:

- Set the phase of traffic light of ```intersection_id``` to ```phase_id```. Only works when ```rlTrafficLight``` is set to ```true```.
- The intersection_id should be defined in ```roadnetFile```
- ```phase_id``` is the no. of phase of the traffic light, defined in ```roadnetFile```

## Test Case

For now we have 5 different scenario, but finally we will only choose one scenario to grade (we will public the final testcase in this week)

## An Example

After installed CityFlow, you can run ```run_default_plan``` as an example. In this file, we use default plan, that means that we do not use any control API in the code. You must change the value of ```rlTrafficLight``` in ```config.json``` from ```false``` to ```true```. After the program done, you can use web front in the folder to see how well the plan is.

```bash
cd fontend/pixi
python3 app.py
```

And open ```http://localhost:8080/?roadnetFile=roadnet.json&logFile=replay.txt``` in browser

However, we suggest to turn off the replay saving when training, for saving the replay log may cost much duplicate time.

## Submission

Here is a list of files which you must provided fully, or you will got a low score or even zero.

1. Source Code
    - Make sure do this project independently.

2. Trained model file
    - Make sure that your code can load the model file.

3. Final Score
    - Submit the final score judged by given judger.
    - TA will randomly sample some submits to test weather the submit  model performs as well as the submit score.

4. A Simple Report
    - Explain the algorithm used in the project briefly.

## Contact information

Siyuan Feng is responsible for this project. Feel free to contact with TA if you have any problem.

Email: [hzfengsy@sjtu.edu.cn](mailto:hzfengsy@sjtu.edu.cn)