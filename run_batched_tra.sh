export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25


CMD0="python main.py --run_group=Debug --seed=0 --env_name=cube-double-v0  \
--dataset_path=/global/scratch/users/vmyers/ogcrl/manipspace/cube-double-noisy-v0.npz \
--train_steps=1000000 --eval_interval=100000 --save_interval=1000000 --log_interval=5000 \
--eval_episodes=50 --video_episodes=1 --agent=algos/tra.py --agent.actor_p_trajgoal=1.0 \
--agent.actor_p_randomgoal=0.0 --agent.use_q=False --agent.actor_loss=awr --agent.alpha=0 \
--agent.discount=0.9"

CMD1="python main.py --run_group=Debug --seed=1 --env_name=cube-double-v0  \
--dataset_path=/global/scratch/users/vmyers/ogcrl/manipspace/cube-double-noisy-v0.npz \
--train_steps=1000000 --eval_interval=100000 --save_interval=1000000 --log_interval=5000 \
--eval_episodes=50 --video_episodes=1 --agent=algos/tra.py --agent.actor_p_trajgoal=1.0 \
--agent.actor_p_randomgoal=0.0 --agent.use_q=False --agent.actor_loss=awr --agent.alpha=0 \
--agent.discount=0.9"

CMD2="python main.py --run_group=Debug --seed=2 --env_name=cube-double-v0  \
--dataset_path=/global/scratch/users/vmyers/ogcrl/manipspace/cube-double-noisy-v0.npz \
--train_steps=1000000 --eval_interval=100000 --save_interval=1000000 --log_interval=5000 \
--eval_episodes=50 --video_episodes=1 --agent=algos/tra.py --agent.actor_p_trajgoal=1.0 \
--agent.actor_p_randomgoal=0.0 --agent.use_q=False --agent.actor_loss=awr --agent.alpha=0 \
--agent.discount=0.9"

CMD3="python main.py --run_group=Debug --seed=3 --env_name=cube-double-v0  \
--dataset_path=/global/scratch/users/vmyers/ogcrl/manipspace/cube-double-noisy-v0.npz \
--train_steps=1000000 --eval_interval=100000 --save_interval=1000000 --log_interval=5000 \
--eval_episodes=50 --video_episodes=1 --agent=algos/tra.py --agent.actor_p_trajgoal=1.0 \
--agent.actor_p_randomgoal=0.0 --agent.use_q=False --agent.actor_loss=awr --agent.alpha=0 \
--agent.discount=0.9"

$CMD0 &
$CMD1 &
$CMD2 &
$CMD3 &
