from PIL import Image
import argparse
import sys

import gym
import numpy as np
import pyglet
import prep
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
from os import path,remove



# ==================================================================================================

dir = 'datas/' 


if path.isfile(dir + 'names.npy'):
  datas = np.load(dir + 'names.npy')
  remove(dir + 'names.npy')
  datas = np.append(datas, datas[len(datas)-1]+1)
else:
  datas = np.array([0])

#print(type(datas))
#exit()

np.save(dir + 'names',datas)

#====================================================================================================

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
env.render()


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)



def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    
    wheel_distance = 0.102
    min_rad = 0.065

    button = np.zeros(5)
    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action += np.array([0.44, 0.0])
        button[0] = 1
    if key_handler[key.DOWN]:
        action -= np.array([0.44, 0])
        button[1] = 1
    if key_handler[key.LEFT]:
        action += np.array([0, 1])
        button[2] = 1
    if key_handler[key.RIGHT]:
        action -= np.array([0, 1])
        button[3] = 1
    if key_handler[key.SPACE]:
        action = np.array([0, 0])
        button[4] = 1

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

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    """print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))"""

#======================================================================================================
    datas = np.load(dir+'names.npy')
    randname = str(datas[len(datas)-1])
    
    x = prep.preprocess(obs)
    #img_x = Image.fromarray(x)
    
    y = button
    count = str(env.unwrapped.step_count)
    namex = dir + 'x/' + randname + '_' + count + '.jpg'
    namey = dir + 'y/' + randname + '_' + count 
    
    x.save(namex)
    np.save(namey, y)

    if done:
        datas = np.load(dir + 'names.npy')
        remove(dir + 'names.npy')
        datas = np.append(datas, datas[len(datas)-1]+1)
        
        print("============================================================================")
        print(datas)
        print("============================================================================")
        
        np.save(dir + 'names', datas)
        
        print("done!")
        env.reset()
        env.render()
#======================================================================================================
    
    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
