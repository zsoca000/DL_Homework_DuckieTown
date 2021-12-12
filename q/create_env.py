from q_prep import preprocess
from collections import deque            # For storing moves 
import numpy as np
from PIL import Image
import gym                                # To train our network

import argparse
import sys
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
from os import path, remove

# bemenet: button=0..4 - hogy melyik gombot nyomtuk le
# kimenet: milyen PWM jelet kell beadni az env.step-be
def button2action(button=0):
  
	action = np.array([0.0, 0.0])
	wheel_distance = 0.102
	min_rad = 0.065

	if button == 0: # UP
		action += np.array([0.44, 0.0])
		#print('UP',end="\r")
	if button == 1: # DOWN
	    action -= np.array([0.44, 0])
	    #print('DOWN',end="\r")
	if button == 2: # LEFT
	    action += np.array([0, 1])
	    #print('LEFT',end="\r")
	if button == 3: # RIGHT
	    action -= np.array([0, 1])
	    #print('RIGHT',end="\r")
	if button == 4: # SPACE
	    action = np.array([0, 0])
	    #print('SPACE',end="\r")
         
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

# létrehozzuk a környezetet (env)
def create_env():
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

  env.reset()
  #env.render()

  return env

# bemenet: button=0..4
# végrehajtjuk az env.stepet
# visszatérünk ennek a kimeneteivel
def do_action(env, button = 4): 
    action = button2action(button)
    obs, reward, done, info = env.step(action)  
    state = preprocess(obs)
    #print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))
    return state, reward, done, info
