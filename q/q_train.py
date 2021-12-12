# Importok
import create_env
from create_env import create_env, button2action, do_action
from q_model import q_model
from q_prep import preprocess
from collections import deque
import numpy as np
from random import sample
from keras.models import save, load_model

# Model beolvasása. Itt továbbtanítunk. De a 14. és 15. sor megcserélésével indíthatunk új tanulást.
dir_ = 'datas/'
input_shape = (40, 80, 3)
#model,input_shape = q_model()
model=load_model(dir_ + 'q_model')

# Hyper Parameters
epochs = 100
observetime = 500                          # Number of timesteps we will be acting on the game and observing results
epsilon = 0.7                              # Probability of doing a random move
decay = 0.98				   # Ezzel csökkentjük epsilont
gamma = 0.5                                # Discounted future reward. How much we care about steps further in time
mb_size = 50                               # Learning minibatch size

env = create_env()

for epoch in range(epochs):	# végigmegyünk az epochokon
	
	# FIRST STEP: Knowing what each action does (Observing)
	# környezet (env) és változók létrehozása
	env.reset()
	
	obs = env.render('rgb_array')
	state = preprocess(obs)
	state = np.expand_dims(state,axis=0)
	done = False
	D = deque()   
	# actionok végrehajtása
	for t in range(observetime):
			env.render()
			if np.random.rand() <= epsilon: # random action
					Q = np.random.uniform(low=0.0, high=1.0, size=(5,))
			else: # action a háló alapján
					state = state.reshape(1,40,80,3)
					Q = model.predict(state/255)

			button = np.argmax(Q)
			action = button2action(button)
			state_new, reward, done, info = do_action(env, button)
			D.append((state, action, reward, state_new, done))         # 'Remember' action and consequence
			state = state_new         # Update state
			if done:
					env.reset()           # Restart game if it's finished
					state = preprocess(x = env.render('rgb_array'))

	# SECOND STEP: Learning from the observations (Experience replay)
	
	#env.reset()
	
	minibatch = sample(list(D), mb_size)                              # Sample some moves

	x_train_shape = (mb_size,) + input_shape
	x_train = np.zeros(x_train_shape)
	y_train = np.zeros((mb_size, 5))

	for i in range(0, mb_size):
			# beolvasás a minibatchből
			state = minibatch[i][0]
			action = np.argmax(minibatch[i][1])
			reward = minibatch[i][2]
			state_new = minibatch[i][3]
			done = minibatch[i][4]

	# Build Bellman equation for the Q function
			x_train[i] = state
			state = state.reshape(1,40,80,3)
			y_train[i] = model.predict(state/255)
			state_new = state_new.reshape(1,40,80,3)
			Q_sa = model.predict(state_new/255)
			
			if done:
					y_train[i, action] = reward
			else:
					y_train[i, action] = reward + gamma * np.max(Q_sa)

			# Train network to output the Q function
			model.train_on_batch(x_train, y_train)
	
	epsilon *= decay
	print('--------------------------------------------------------------')
	print('Epoch ', str(epoch), ' finished. next epsilon = ', str(epsilon))
	print('--------------------------------------------------------------')
	
	if epoch%25 == 0:	# néha biztonsági mentést készítünk
		model.save(dir_ + 'q_model')
		print("   saved")
