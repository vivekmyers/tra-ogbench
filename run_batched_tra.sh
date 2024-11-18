export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25

NAME="$1"
DATA_NAME="$2"

CMD0="python main.py --run_group=Debug --seed=0 --env_name=$NAME \
	    --train_steps=1000000 --eval_interval=100000 --save_interval=1000000 --log_interval=5000 \
	        --eval_episodes=50 --video_episodes=1 --agent=agents/tra.py --dataset_path=/global/scratch/users/seohong/data/ogcrl/$DATA_NAME --agent.actor_p_trajgoal=1.0 \
		    --agent.actor_p_randomgoal=0.0 --agent.alpha=0 --agent.discount=0.9 --agent.encoder=impala_small"
CMD1="python main.py --run_group=Debug --seed=1 --env_name=$NAME \
	    --train_steps=1000000 --eval_interval=100000 --save_interval=1000000 --log_interval=5000 \
	        --eval_episodes=50 --video_episodes=1 --agent=agents/tra.py --dataset_path=/global/scratch/users/seohong/data/ogcrl/$DATA_NAME --agent.actor_p_trajgoal=1.0 \
		    --agent.actor_p_randomgoal=0.0 --agent.alpha=0 --agent.discount=0.9 --agent.encoder=impala_small"

CMD2="python main.py --run_group=Debug --seed=2 --env_name=$NAME \
	    --train_steps=1000000 --eval_interval=100000 --save_interval=1000000 --log_interval=5000 \
	        --eval_episodes=50 --video_episodes=1 --agent=agents/tra.py --dataset_path=/global/scratch/users/seohong/data/ogcrl/$DATA_NAME --agent.actor_p_trajgoal=1.0 \
		    --agent.actor_p_randomgoal=0.0 --agent.alpha=0 --agent.discount=0.9 --agent.encoder=impala_small"
CMD3="python main.py --run_group=Debug --seed=3 --env_name=$NAME \
	    --train_steps=1000000 --eval_interval=100000 --save_interval=1000000 --log_interval=5000 \
	        --eval_episodes=50 --video_episodes=1 --agent=agents/tra.py --dataset_path=/global/scratch/users/seohong/data/ogcrl/$DATA_NAME --agent.actor_p_trajgoal=1.0 \
		    --agent.actor_p_randomgoal=0.0 --agent.alpha=0 --agent.discount=0.9 --agent.encoder=impala_small"

$CMD0 &
$CMD1 &
$CMD2 &
$CMD3 &

