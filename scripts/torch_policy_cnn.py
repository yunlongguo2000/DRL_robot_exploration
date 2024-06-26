import os
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.transform import resize
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from torch_networks import CNNNetwork
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
REPLAY_MEMORY = 10000  # Number of previous transitions to remember
BATCH = 64  # Size of minibatch
FINAL_RATE = 0  # Final value of dropout rate
INITIAL_RATE = 0.9  # Initial value of dropout rate
TARGET_UPDATE = 25000  # Update frequency of the target network

network_dir = "../saved_networks/" + "cnn_" + str(ACTIONS)
if not os.path.exists(network_dir):
    os.makedirs(network_dir)
if TRAIN:
    log_dir = "../log/" + "cnn_" + str(ACTIONS)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def start():
    policy_net = CNNNetwork(ACTIONS).to(device)
    target_net = CNNNetwork(ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    # Initialize a training environment
    robot_explo = robot.Robot(0, TRAIN, PLOT)
    step_t = 0
    drop_rate = INITIAL_RATE
    total_reward = np.empty([0, 0])
    finish_all_map = False

    # Store the previous observations in replay memory
    D = deque()

    # Tensorboard
    if TRAIN:
        writer = SummaryWriter(log_dir=log_dir)

    # Saving and loading networks
    if TRAIN:
        pass  # No need to load the network when training from scratch
    else:
        try:
            checkpoint = torch.load(os.path.join(network_dir, 'cnn_2000000.pth'))
            policy_net.load_state_dict(checkpoint)
            print("Successfully loaded the network weights")
        except FileNotFoundError:
            print("Could not find old network weights, starting from scratch")

    # Get the first state by doing nothing and preprocess the image to 80x80x1
    x_t = robot_explo.begin()
    x_t = resize(x_t, (84, 84))
    s_t = torch.from_numpy(np.reshape(x_t, (1, 1, 84, 84))).float().to(device)
    a_t_coll = []

    while TRAIN and step_t <= EXPLORE:
        # Scale down dropout rate
        if drop_rate > FINAL_RATE and step_t > OBSERVE:
            drop_rate -= (INITIAL_RATE - FINAL_RATE) / EXPLORE

        # Choose an action by uncertainty
        readout_t = policy_net(s_t)[0]
        readout_t[a_t_coll] = float("-inf")
        action_index = torch.argmax(readout_t).item()

        # Run the selected action and observe next state and reward
        x_t1, r_t, terminal, complete, re_locate, collision_index, _ = robot_explo.step(action_index)
        x_t1 = resize(x_t1, (84, 84))
        x_t1 = np.reshape(x_t1, (1, 1, 84, 84))
        s_t1 = torch.from_numpy(x_t1).float().to(device)
        finish = terminal

        # Store the transition
        D.append((s_t, torch.tensor([action_index]), torch.tensor([r_t]), s_t1, torch.tensor([terminal])))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if step_t > OBSERVE:
            # Update target network
            if step_t % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # Get the batch variables
            s_j_batch = torch.cat([s for s, _, _, _, _ in minibatch]).to(device)
            a_batch = torch.LongTensor([a for _, a, _, _, _ in minibatch]).to(device)
            r_batch = torch.Tensor([r for _, _, r, _, _ in minibatch]).to(device)
            s_j1_batch = torch.cat([s1 for _, _, _, s1, _ in minibatch]).to(device)
            terminal_batch = torch.Tensor([t for _, _, _, _, t in minibatch]).to(device)

            readout_j1_batch = target_net(s_j1_batch)
            y_batch = r_batch + GAMMA * torch.max(readout_j1_batch, dim=1)[0] * (1 - terminal_batch)

            # Perform gradient step
            optimizer.zero_grad()
            q_values = policy_net(s_j_batch).gather(1, a_batch.unsqueeze(1)).squeeze()
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
            torch.save(policy_net.state_dict(), os.path.join(network_dir, 'cnn_{}.pth'.format(step_t)))

        print("TIMESTEP", step_t, "/ DROPOUT", drop_rate, "/ ACTION", action_index, "/ REWARD", r_t, "/ Terminal", finish, "\n")

        # Reset the environment
        if finish:
            if complete:
                x_t = robot_explo.begin()
            if re_locate:
                x_t, re_locate_complete, _ = robot_explo.rescuer()
                if re_locate_complete:
                    x_t = robot_explo.begin()
            x_t = resize(x_t, (84, 84))
            s_t = torch.from_numpy(np.reshape(x_t, (1, 1, 84, 84))).float().to(device)
            a_t_coll = []
            continue

        s_t = s_t1

    while not TRAIN and not finish_all_map:
        # Choose an action by policy
        readout_t = policy_net(s_t)[0]
        readout_t[a_t_coll] = float("-inf")
        action_index = torch.argmax(readout_t).item()

        # Run the selected action and observe next state and reward
        x_t1, r_t, terminal, complete, re_locate, collision_index, finish_all_map = robot_explo.step(action_index)
        x_t1 = resize(x_t1, (84, 84))
        x_t1 = np.reshape(x_t1, (1, 1, 84, 84))
        s_t1 = torch.from_numpy(x_t1).float().to(device)
        finish = terminal

        step_t += 1
        print("TIMESTEP", step_t, "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %e" % torch.max(readout_t).item(), "/ Terminal", finish, "\n")

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