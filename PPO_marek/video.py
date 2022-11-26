import gym

def monitoring():
    from gym.wrappers.monitoring.video_recorder import VideoRecorder
    video_path = "models/test_video.mp4"

    env = gym.make('CarRacing-v2', render_mode='rgb_array')

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
    env = gym.wrappers.RecordVideo(env, "recording")
    env.reset()
    done = False
    while not done:
        obs, rew, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        
wrapper_video()
