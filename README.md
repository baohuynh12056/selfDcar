
# SelfDCar 

**SelfDCar** is a reinforcement learning project that uses **YOLOv8** for real-time object detection in the game **Traffic Racer**. The agent receives visual input through **scrcpy**, processes it with computer vision, and learns to control the car to avoid obstacles and drive autonomously.

**Note:** This project is developed and tested on **Linux (Ubuntu 22.04)** and is designed to work only with **Android devices** (via scrcpy) 

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/baohuynh12056/selfDcar.git
cd selfDcar
```
### 2. Creat a virtual environment.
If you have not created one yet, run:
```bash
python3 -m venv .venv
```
### 3. Activate the virtual environment
```bash
source .venv/bin/activate
```
### 4.Install Python dependencies
Make sure you have Python 3.10+ installed, then run:
```bash
pip install -r requirements.txt
```
### 5. Install external dependencies
- **scrcpy**: stream and control the Android game *Traffic Racer*
```bash
 sudo apt update 
 sudo apt install scrcpy
```
- **v4l2loopback**: create a virtual webcam device (required for video streaming)
```bash
 sudo apt install v4l2loopback-dkms v4l2loopback-utils

```
### 6. Verify installation
Check that scrcpy works:
```bash
scrcpy --version
```
On Linux, also check that v4l2loopback is available:
```bash
modprobe v4l2loopback
```
## Usage/How to Run

### 1. Connect Android device
- Enable **USB debugging** on your Android phone (Developer Options).  
- Connect the phone to your computer via USB cable.  
- Verify that the device is detected:
  ```bash
  adb devices
### 2. Setup virtual camera 
Use **v4l2loopback** to create a virtual camera device that scrcpy can stream into:
```bash
sudo modprobe v4l2loopback video_nr=10 card_label="scrcpy" exclusive_caps=1
```
This will create a virtual camera at /dev/video10.
You can check with:
```bash
ls /dev/video*
```
### 3. Launch the game
- Open **Traffic Racer** on your Android device.  
- Select the **Endless - One Way** mode (this mode is supported by the agent).  

### 4. Prepare training data (optional)
This project already includes pre-trained weights:  
- `car_detector.pt` (YOLOv8 model for car detection)  
- `ppo_car_agent.pt` (baseline PPO reinforcement learning agent)  
You can use these models directly to run the project without additional data preparation.  

Alternatively, if you want to create your own dataset and train from scratch:  

    1. Use the tools provided in the `tools/` folder:  
        - `data_collector.py` → collect images and gameplay data from scrcpy  
        - `simple_labeler` → label collected images for YOLO training  
        - `train_yolo/` → train a YOLOv8 model with your labeled dataset  

    2. Once training is complete, replace the default weights (`car_detector.pt` and `ppo_car_agent.pt`) with your newly trained models.  

If you prefer, you can also download prepared datasets from the provided Google Drive link (https://drive.google.com/drive/folders/1jq3_rW2RZr084RNqc6LZY47G1P7don1-?usp=sharing)
### 5. Run the agent  
You can specify different modes via command-line arguments.
```bash
python -m game_car_ai.src.ai.train --mode <option> --timesteps <option> --model-path <option> --device <option>
```
#### Options:
- ```--mode```: choose the execution mode (default = ```train```)
    - ```train``` → train a new agent.
    - ```tune``` → run hyperparameter tuning.
    - ```run``` → run an already trained agent
- ```--timesteps```: : number of RL training steps (default = 100000). Only for ```train``` mode.
- ```--model-path``` : path to save/load the model (default = ```game_car_ai/assets/weights/ppo_car_agent/ppo_car_agent```)
- ```--device``` : computation device (default = ```auto```)
    - ```auto``` → automatically choose GPU if available, otherwise fallback to CPU
    - ```cuda``` → force GPU
    - ```cpu``` → force CPU
#### Example commands:
- Train agent with default 100k steps:
```bash
python -m game_car_ai.src.ai.train --mode train --device cuda
```
- Run a pre-trained agent:
```bash
python -m game_car_ai.src.ai.train --mode run --model-path game_car_ai/assets/weights/ppo_car_agent/ppo_car_agent --device cuda
```
⚠️ **Important note**:  
This project uses `pyautogui` to send keyboard commands.  
`pyautogui` always controls the **currently active window** on your computer.  

👉 Therefore, you must **click and focus on the scrcpy window** (the last one that opened).  
If scrcpy is not the active window, the agent’s key presses will be sent to whatever application is currently in focus — not the game.
#### 🎥 Demo
Watch the video tutorial on how to set up and run the project:  

[![Watch the video](https://i.ytimg.com/an_webp/ycZmLaZIPuQ/mqdefault_6s.webp?du=3000&sqp=CLyEnsYG&rs=AOn4CLDciK6S7K6Z4GSrrCjpvURttwrcsw)](https://youtu.be/ycZmLaZIPuQ?si=84KgSZUygPrk8HGk)

## Acknowledgements

This project is developed purely for **educational purposes** by a first-year student,  
with the goal of gaining a deeper understanding of **Computer Vision** and **Reinforcement Learning** at a basic level.  

The main objective is to practice and explore fundamental libraries for building AI systems.  
I hope to further improve this project in the near future so that the self-driving car can operate  
more accurately and potentially have real-world applications.

Special thanks to the open-source community and the authors of the resources below,  
whose work and tutorials greatly inspired and guided this project.
### References & Resources
- “Reinforcement Learning là gì? Khám phá các thuật toán trong học tăng cường” – VNPT AI. [Link](https://vnptai.io/vi/blog/detail/reinforcement-learning-la-gi)  
- “Reinforcement Learning in Games: A Complete Guide” – Amit Yadav, Medium. [Link](https://medium.com/@amit25173/reinforcement-learning-in-games-a-complete-guide-24d1cab79317)
- [Reinforcement Learning: Essential Concepts – Joshua Starmer, StatQuest](https://www.youtube.com/watch?v=Z-T0iJEXiwM)  
- [I Coded One Project EVERY WEEK for a YEAR – Carter Semrad](https://www.youtube.com/watch?v=nr8biZfSZ3Y)