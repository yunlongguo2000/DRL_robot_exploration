import os
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.transform import resize
import random
import numpy as np
import matplotlib.pyplot as plt
from tf_networks import LSTMNetwork
from tensorboardX import SummaryWriter
import robot_simulation as robot

# Select mode
TRAIN = False
PLOT = True

# Training environment parameters
ACTIONS = 50  # Number of valid actions
GAMMA = 0.99  # Decay rate of past observations
OBSERVE = 1e4  # Timesteps to observe before training
EXPLORE = 2e6  # Frames over which to anneal epsilon
REPLAY_MEMORY = 1000  # Number of previous transitions to remember
BATCH = 8  # Size of minibatch
h_size = 512  # Size of hidden cells of LSTM
trace_length = 8  # Memory length
FINAL_RATE = 0  # Final value of dropout rate
INITIAL_RATE = 0.9  # Initial value of dropout rate
TARGET_UPDATE = 25000  # Update frequency of the target network

network_dir = "../saved_networks/" + "rnn_" + str(ACTIONS)
if not os.path.exists(network_dir):
    os.makedirs(network_dir)
if TRAIN:
    log_dir = "../log/" + "rnn_" + str(ACTIONS)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExperienceBuffer:
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append(episode[point:point + trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size * trace_length, 5])

def padd_eps(eps_buff):
    if len(eps_buff) < trace_length:
        s = torch.zeros(1, 1, 84, 84).to(device)
        a = torch.zeros(ACTIONS).to(device)
        r = 0
        s1 = torch.zeros(1, 1, 84, 84).to(device)
        d = True
        for i in range(0, trace_length - len(eps_buff)):
            eps_buff.append(torch.tensor([s, a, r, s1, d]).to(device))
    return eps_buff

def start():
    policy_net = LSTMNetwork(ACTIONS, h_size).to(device)
    target_net = LSTMNetwork(ACTIONS, h_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    # Initialize a training environment
    robot_explo = robot.Robot(0, TRAIN, PLOT)
    myBuffer = ExperienceBuffer(REPLAY_MEMORY)
    step_t = 0
    drop_rate = INITIAL_RATE
    total_reward = np.empty([0, 0])
    init_state = (torch.zeros(1, h_size).to(device), torch.zeros(1, h_size).to(device))
    finish_all_map = False

    # Tensorboard
    if TRAIN:
        writer = SummaryWriter(log_dir=log_dir)

    # Saving and loading networks
    if TRAIN:
        pass  # No need to load the network when training from scratch
    else:
        try:
            checkpoint = torch.load(os.path.join(network_dir, 'rnn.pth'))
            policy_net.load_state_dict(checkpoint)
            print("Successfully loaded the network weights")
        except FileNotFoundError:
            print("Could not find old network weights, starting from scratch")

    # Get the first state by doing nothing and preprocess the image to 84x84x1
    x_t = robot_explo.begin()
    x_t = resize(x_t, (84, 84))
    s_t = torch.from_numpy(np.reshape(x_t, (1, 1, 84, 84))).float().to(device)
    state = init_state
    a_t_coll = []
    episodeBuffer = []

    while TRAIN and step_t <= EXPLORE:
        # Scale down dropout rate
        if drop_rate > FINAL_RATE and step_t > OBSERVE:
            drop_rate -= (INITIAL_RATE - FINAL_RATE) / EXPLORE

        # Choose an action by uncertainty
        readout_t, state1 = policy_net(s_t, trace_length, 1, state)
        readout_t = readout_t[0]
        readout_t[a_t_coll] = float("-inf")
        action_index = torch.argmax(readout_t).item()

        # Run the selected action and observe next state and reward
        x_t1, r_t, terminal, complete, re_locate, collision_index, _ = robot_explo.step(action_index)
        x_t1 = resize(x_t1, (84, 84))
        x_t1 = np.reshape(x_t1, (1, 1, 84, 84))
        s_t1 = torch.from_numpy(x_t1).float().to(device)
        finish = terminal

        # Store the transition
        episodeBuffer.append(torch.tensor([s_t, torch.tensor([action_index]), torch.tensor([r_t]), s_t1, torch.tensor([terminal])]).to(device))

        if step_t > OBSERVE:
            # Update target network
            if step_t % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Reset the recurrent layer's hidden state
            state_train = (torch.zeros(BATCH, h_size).to(device), torch.zeros(BATCH, h_size).to(device))

            # Sample a minibatch to train on
            trainBatch = myBuffer.sample(BATCH, trace_length)

            # Get the batch variables
            s_j_batch = torch.cat([s for s, _, _, _, _ in trainBatch]).to(device)
            a_batch = torch.LongTensor([a for _, a, _, _, _ in trainBatch]).to(device)
            r_batch = torch.Tensor([r for _, _, r, _, _ in trainBatch]).to(device)
            s_j1_batch = torch.cat([s1 for _, _, _, s1, _ in trainBatch]).to(device)
            terminal_batch = torch.Tensor([t for _, _, _, _, t in trainBatch]).to(device)

            readout_j1_batch, _ = target_net(s_j1_batch, trace_length, BATCH, state_train)
            y_batch = r_batch + GAMMA * torch.max(readout_j1_batch, dim=1)[0] * (1 - terminal_batch)

            # Perform gradient step
            optimizer.zero_grad()
            readout_batch, _ = policy_net(s_j_batch, trace_length, BATCH, state_train)
            q_values = readout_batch.gather(1, a_batch.unsqueeze(1)).squeeze()
            loss = criterion(q_values, y_batch)
            loss.backward()
            optimizer.step()

            # Update tensorboard
            new_average_reward = np.average(total_reward[len(total_reward) - 10000:])
            writer.add_scalar('average reward', new_average_reward, step_t)

        step_t += 1
        total_reward = np.append(total_reward, r_t)

        # Save progress
        if step_t == 2e4 or step_t == 2e5 or step_t == 2e6:
            torch.save(policy_net.state_dict(), os.path.join(network_dir, 'rnn.pth'))

        print("TIMESTEP", step_t, "/ DROPOUT", drop_rate, "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %e" % torch.max(readout_t).item(), "/ Terminal", finish, "\n")

        # Reset the environment
        if finish:
            bufferArray = torch.stack(padd_eps(episodeBuffer))
            episodeBuffer = []
            myBuffer.add(bufferArray)
            if complete:
                x_t = robot_explo.begin()
            if re_locate:
                x_t, re_locate_complete, _ = robot_explo.rescuer()
                if re_locate_complete:
                    x_t = robot_explo.begin()
            x_t = resize(x_t, (84, 84))
            s_t = torch.from_numpy(np.reshape(x_t, (1, 1, 84, 84))).float().to(device)
            a_t_coll = []
            state = init_state
            continue

        state = state1
        s_t = s_t1

    while not TRAIN and not finish_all_map:
        # Choose an action by policy
        readout_t, state1 = policy_net(s_t, trace_length, 1, state)
        readout_t = readout_t[0]
        readout_t[a_t_coll] = float("-inf")
        action_index = torch.argmax(readout_t).item()

        # Run the selected action and observe next state and reward
        x_t1, r_t, terminal, complete, re_locate, collision_index, finish_all_map = robot_explo.step(action_index)
        x_t1 = resize(x_t1, (84, 84))
        x_t1 = np.reshape(x_t1, (1, 1, 84, 84))
        s_t1 = torch.from_numpy(x_t1).float().to(device)
        finish = terminal

        step_t += 1
        print("TIMESTEP", step_t, "/ ACTION", action_index, "/ REWARD", r_t, "/ Terminal", finish, "\n")

        if finish:
            a_t_coll = []
            if complete:
                x_t = robot_explo.begin()
            if re_locate:
                x_t, re_locate_complete, finish_all_map = robot_explo.rescuer()
                if re_locate_complete:
                    x_t = robot_explo.begin()
            x_t = resize(x_t, (84, 84))
            s_t = torch.from_numpy(np.reshape(x_t, (1, 1, 84, 84))).float().to(device)
            continue

        # Avoid collision next time
        if collision_index:
            a_t_coll.append(action_index)
            continue
        a_t_coll = []
        s_t = s_t1

if __name__ == "__main__":
    start()
    if PLOT:
        plt.show()