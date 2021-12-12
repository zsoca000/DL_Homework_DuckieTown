#====================================IMPORT====================================

from tensorflow import keras
from keras.models import save, load_model
from PIL import Image
import argparse
import sys
from keras.preprocessing.image import img_to_array
import gym
import numpy as np
import pyglet
import prep
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
from os import path,remove
from prep import preprocess

#====================================DATAS=====================================

path = 'datas/' 
model = load_model(path + 'reinf_learning_model')
time_step = 5

#==================================PARSING=====================================

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default="Duckietown-udem1-v0")
parser.add_argument("--map-name", default="zigzag_dists")  # MAPNAME
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--seed", default=1, type=int, help="seed")
args = parser.parse_args()

if args.env_name and args.env_name.find("Duckietown") != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
        camera_rand=args.camera_rand,
        dynamics_rand=args.dynamics_rand,
    )
else:
    env = gym.make(args.env_name)


#=============================BUTTON_PRESS-->ACTION==============================

# convert button press to action

def button2action(button = 0):
      
  wheel_distance = 0.102
  min_rad = 0.065
  action = np.array([0.0, 0.0]) 
      
  if button == 0: # UP
    action += np.array([0.44, 0.0])
    print('UP',end='\r')
  if button == 1: # DOWN
    action -= np.array([0.44, 0])
    print('DOWN',end='\r')
  if button == 2: # LEFT
    action += np.array([0, 1])
    print('LEFT',end='\r')
  if button == 3: # RIGHT
    action -= np.array([0, 1])
    print('RIGHT',end='\r')
  if button == 4: # SPACE
    action = np.array([0, 0])
    print('SPACE',end='\r')

  v1 = 1.5*action[0]
  v2 = 3*action[1]
  # Limit radius of curvature
  if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
    # adjust velocities evenly such that condition is fulfilled
    delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
    v1 += abs(delta_v)
    v2 -= delta_v

  action[0] = v1
  action[1] = v2
         
  return action

#===============================ACTUALLY_THE_PLAY=====================================

while True:
   env.reset()
   env.render()
   state = []
  
   # starting state
   for i in range(5):
     action = button2action()
     observation, reward, done, info = env.step(action)

     state.append( img_to_array ( preprocess(observation) ) )    
     
   done = False
   tot_reward = 0.0
   
   while not done:
      env.render()  # render         
      
      # refresh the state with the predicted button commands
      state_np = np.array(state).reshape(1,time_step,40,80,3)
      state_np /= 255
      y_pred = model.predict(state_np) 
      
      button = np.argmax(y_pred)
      action = button2action(button)
      
      observation, reward, done, info = env.step(action)
      state.append( img_to_array ( preprocess(observation) ) ) 
      state.pop(0)
      
      
      tot_reward += reward
      
      print('Game ended! Total reward: {}'.format(reward))
      
      
#=================================================================================
      
