#Python3
import gym
import numpy as np
import random
from keras.layers import Dense,Conv2D,Multiply,Input,Flatten,Lambda
from collections import deque
from keras.optimizers import RMSprop
from keras.models import Model,clone_model
from keras import backend as K
from skimage.color import rgb2gray
from skimage.transform import resize
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"

env=gym.make('PongDeterministic-v4')
INPUT_SHAPE=(84,84,4)

def pre_process_image(img):
    img=np.uint8(resize(rgb2gray(img), (84, 84), mode='constant') * 255)
    return img

image_input=Input(shape=(84,84,4))
action_done=Input(shape=(5,))

Q_predictor =Lambda(lambda x: x / 255.0)(image_input)
Q_predictor=Conv2D(16,(8,8),strides=(4,4),activation='relu')(Q_predictor)
Q_predictor=Conv2D(32,(4,4),strides=(2,2),activation='relu')(Q_predictor)
Q_predictor=Flatten()(Q_predictor)
Q_predictor=Dense(256,activation='relu')(Q_predictor)
Q_predictor=Dense(5)(Q_predictor)

final_output=Multiply()([Q_predictor,action_done])

model=Model(inputs=[image_input,action_done],outputs=final_output)
model.load_weights('atari_model_pong.h5')

def action_todo(curr_state):
    global model
    predictions=model.predict([curr_state, np.ones(5).reshape(1, 5)])
    predictions=predictions[0]
    print(predictions,np.argmax(predictions))
    time.sleep(0.05)
    return np.argmax(predictions)

num_episodes=0

while num_episodes<100000:
    done = False
    dead = False
    num_episodes=num_episodes+1
    print("Episode",num_episodes,"started")
    env.reset()
    life_left=5
    #skip few frames at the beginning of the episode
    observation,reward,done,info=env.step(1)
    observation=pre_process_image(observation)
    observation=np.stack((observation,observation,observation,observation),axis=2)
    frame_history=np.reshape(observation,(1,84,84,4))
    while not done:
        env.render()
        action=action_todo(frame_history)
        action_asked=action+1
        observation,reward,done,info=env.step(action_asked)
        observation=pre_process_image(observation)
        observation=np.reshape(observation,(1,84,84,1))
        next_frame=np.append(observation,frame_history[:,:,:,:3],axis=3)
        dead=False
        if life_left>info['ale.lives']:
            dead=True
            life_left=info['ale.lives']

        if not dead:
            frame_history=next_frame
