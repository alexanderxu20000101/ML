import gym
#env = gym.make('Taxi-v1')
env = gym.make('MountainCar-v0')
env = gym.make('CartPole-v1')
env = gym.make('Acrobot-v1')

# env = gym.make('Ant-v1')  # 使用这个，需要安装mujoco的包（c++写的，通常用在linux上）
#           # win上用的不顺畅，资料不多。 这个包要收费的。
#env = gym.make('Assault-v0')   #用这个需要 安装atari包


# env = gym.make('MountainCarContinuous-v0')



observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)