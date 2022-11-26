import gym

def monitoring():
    from gym.wrappers.monitoring.video_recorder import VideoRecorder
    video_path = "models/test_video.mp4"

    env = gym.make('CarRacing-v2', render_mode='rgb_array')
    # https://www.anyscale.com/blog/an-introduction-to-reinforcement-learning-with-openai-gym-rllib-and-google
    # https://www.programcreek.com/python/example/95356/gym.monitoring.VideoRecorder
    # https://github.com/openai/gym/blob/master/gym/wrappers/monitoring/video_recorder.py
    video = VideoRecorder(env, path = video_path)
    # returns an initial observation
    env.reset()
    for i in range(200):
        env.render()
        video.capture_frame()
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        if done:
            break
        # Not printing this time
        print("step", i, reward)

    video.close()
    env.close()

def wrapper_video():
    env = gym.make('CarRacing-v2', render_mode='rgb_array')
    # https://www.gymlibrary.dev/api/wrappers/
    # https://github.com/openai/gym/blob/master/gym/wrappers/record_video.py
    env = gym.wrappers.RecordVideo(env, "recording")
    env.reset()
    done = False
    while not done:
        obs, rew, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        
wrapper_video()

"""
https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/vec_env/vec_video_recorder.html#VecVideoRecorder
https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/vec_env/base_vec_env.html
https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
https://www.gymlibrary.dev/content/vectorising/
https://www.gymlibrary.dev/api/wrappers/

trigger
https://stackoverflow.com/questions/71656396/how-to-set-a-trigger-for-when-to-record-video-open-ai-gym
"""