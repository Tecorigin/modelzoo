#!/usr/bin/env python3
import cv2
import random
import numpy as np
import argparse
from DRL.evaluator import Evaluator
from utils.util import *
from utils.tensorboard import TensorBoard
import time
import torch

exp = os.path.abspath('.').split('/')[-1]
writer = TensorBoard('../train_log/{}'.format(exp))
os.system('ln -sf ../train_log/{} ./log'.format(exp))
os.system('mkdir ./model')

def train(agent, env, evaluate):
    train_times = args.train_times
    env_batch = args.env_batch
    validate_interval = args.validate_interval
    max_step = args.max_step
    debug = args.debug
    episode_train_times = args.episode_train_times
    resume = args.resume
    output = args.output
    time_stamp = time.time()
    step = episode = episode_steps = 0
    tot_reward = 0.
    observation = None
    noise_factor = args.noise_factor
    
    # 添加loss日志文件
    loss_log_file = f"training_loss_log_{int(time.time())}.txt"
    with open(loss_log_file, 'w') as f:
        f.write("episode,step,train_step,critic_loss,actor_loss,Q_value\n")
    
    while step <= train_times:
        step += 1
        episode_steps += 1
        # reset if it is the start of episode
        if observation is None:
            observation = env.reset()
            agent.reset(observation, noise_factor)    
        action = agent.select_action(observation, noise_factor=noise_factor)
        observation, reward, done, _ = env.step(action)
        agent.observe(reward, observation, done, step)
        if (episode_steps >= max_step and max_step):
            if step > args.warmup:
                # [optional] evaluate
                if episode > 0 and validate_interval > 0 and episode % validate_interval == 0:
                    reward, dist = evaluate(env, agent.select_action, debug=debug)
                    if debug: prRed('Step_{:07d}: mean_reward:{:.3f} mean_dist:{:.3f} var_dist:{:.3f}'.format(step - 1, np.mean(reward), np.mean(dist), np.var(dist)))
                    writer.add_scalar('validate/mean_reward', np.mean(reward), step)
                    writer.add_scalar('validate/mean_dist', np.mean(dist), step)
                    writer.add_scalar('validate/var_dist', np.var(dist), step)
                    agent.save_model(output)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            tot_Q = 0.
            tot_value_loss = 0.
            tot_policy_loss = 0.  # 添加策略loss变量
            
            # 修改：移除step > args.warmup条件，让所有step都进行训练和打印
            if step >= 1:  # 从第一个step开始
                if step < 10000 * max_step:
                    lr = (3e-4, 1e-3)
                elif step < 20000 * max_step:
                    lr = (1e-4, 3e-4)
                else:
                    lr = (3e-5, 1e-4)
                for i in range(episode_train_times):
                    # 修改这里以获取更多的返回值
                    train_result = agent.update_policy(lr)
                    
                    # 更安全的返回值处理
                    if isinstance(train_result, tuple):
                        if len(train_result) >= 3:
                            Q, value_loss, policy_loss = train_result[0], train_result[1], train_result[2]
                        elif len(train_result) >= 2:
                            Q, value_loss = train_result[0], train_result[1]
                            policy_loss = torch.tensor(0.0)
                        else:
                            Q = train_result[0] if len(train_result) > 0 else torch.tensor(0.0)
                            value_loss = torch.tensor(0.0)
                            policy_loss = torch.tensor(0.0)
                    else:
                        # 如果不是元组，假设是单个值
                        Q = train_result if train_result is not None else torch.tensor(0.0)
                        value_loss = torch.tensor(0.0)
                        policy_loss = torch.tensor(0.0)
                    
                    # 转换为numpy值
                    try:
                        Q_val = Q.data.cpu().numpy() if hasattr(Q, 'data') else float(Q)
                    except:
                        Q_val = float(Q) if Q is not None else 0.0
                        
                    try:
                        value_loss_val = value_loss.data.cpu().numpy() if hasattr(value_loss, 'data') else float(value_loss)
                    except:
                        value_loss_val = float(value_loss) if value_loss is not None else 0.0
                        
                    try:
                        policy_loss_val = policy_loss.data.cpu().numpy() if hasattr(policy_loss, 'data') else float(policy_loss)
                    except:
                        policy_loss_val = float(policy_loss) if policy_loss is not None else 0.0
                    
                    tot_Q += Q_val
                    tot_value_loss += value_loss_val
                    tot_policy_loss += policy_loss_val
                    
                    # 打印每个训练step的loss，增加小数位数
                    if debug:
                        print(f'Episode {episode:3d} | TrainStep {i+1:2d} | CriticLoss: {value_loss_val:.9f} | ActorLoss: {policy_loss_val:.9f} | Q: {Q_val:.9f}')
                    
                    # 保存到日志文件，保留更多精度
                    with open(loss_log_file, 'a') as f:
                        f.write(f"{episode},{step},{i+1},{value_loss_val:.9f},{policy_loss_val:.9f},{Q_val:.9f}\n")
                
                # 记录平均loss到tensorboard（同样移除warmup条件）
                writer.add_scalar('train/critic_lr', lr[0], step)
                writer.add_scalar('train/actor_lr', lr[1], step)
                writer.add_scalar('train/Q', tot_Q / episode_train_times, step)
                writer.add_scalar('train/critic_loss', tot_value_loss / episode_train_times, step)
                writer.add_scalar('train/actor_loss', tot_policy_loss / episode_train_times, step)
                
                # 打印该episode的平均loss，增加小数位数
                if debug:
                    avg_critic_loss = tot_value_loss / episode_train_times
                    avg_actor_loss = tot_policy_loss / episode_train_times
                    avg_Q = tot_Q / episode_train_times
                    print(f'Episode {episode:3d} | Avg CriticLoss: {avg_critic_loss:.9f} | Avg ActorLoss: {avg_actor_loss:.9f} | Avg Q: {avg_Q:.9f}')
                
            if debug: 
                prBlack('#{}: steps:{} interval_time:{:.2f} train_time:{:.2f}'.format(episode, step, train_time_interval, time.time()-time_stamp)) 
            time_stamp = time.time()
            # reset
            observation = None
            episode_steps = 0
            episode += 1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning to Paint')

    # hyper-parameter
    parser.add_argument('--warmup', default=400, type=int, help='timestep without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.95**5, type=float, help='discount factor')
    parser.add_argument('--batch_size', default=96, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=800, type=int, help='replay memory size')
    parser.add_argument('--env_batch', default=96, type=int, help='concurrent environment number')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
    parser.add_argument('--noise_factor', default=0, type=float, help='noise level for parameter space noise')
    parser.add_argument('--validate_interval', default=50, type=int, help='how many episodes to perform a validation')
    parser.add_argument('--validate_episodes', default=5, type=int, help='how many episode to perform during validation')
    parser.add_argument('--train_times', default=2000000, type=int, help='total traintimes')
    parser.add_argument('--episode_train_times', default=10, type=int, help='train times for each episode')    
    parser.add_argument('--resume', default=None, type=str, help='Resuming model path for testing')
    parser.add_argument('--output', default='./model', type=str, help='Resuming model path for testing')
    parser.add_argument('--debug', dest='debug', action='store_true', help='print some info')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    
    args = parser.parse_args()    
    args.output = get_output_folder(args.output, "Paint")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    from DRL.ddpg import DDPG
    from DRL.multi import fastenv
    fenv = fastenv(args.max_step, args.env_batch, writer)
    agent = DDPG(args.batch_size, args.env_batch, args.max_step, \
                 args.tau, args.discount, args.rmsize, \
                 writer, args.resume, args.output)
    evaluate = Evaluator(args, writer)
    print('observation_space', fenv.observation_space, 'action_space', fenv.action_space)
    train(agent, fenv, evaluate)