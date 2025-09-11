# tools/create_data_yaml.py
import yaml
import os

def create_data_yaml():
    """Tạo file data.yaml cho YOLO"""
    data = {
        'path': '../datasets/car_obstacle',
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'player_car', 1: 'opponent_car'},
        'nc': 2
    }
    
    with open('datasets/car_obstacle/data.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print("Created data.yaml")
    print(yaml.dump(data, default_flow_style=False))

create_data_yaml()