NAME="$1"
DATA_NAME="$2"
SEED="$3"

CMD="python main.py --run_group=Debug --seed=$SEED --env_name=$NAME \
	 --train_steps=500000 --eval_interval=100000 --save_interval=1000000 --log_interval=5000 \
	  --eval_episodes=50 --video_episodes=1 --agent=agents/tra.py --dataset_path=/global/scratch/users/seohong/data/ogcrl/$DATA_NAME --agent.actor_p_trajgoal=1.0 \
	   --agent.actor_p_randomgoal=0.0 --agent.alpha=0 --agent.batch_size=256 --agent.alignment=0.01 --agent.discount=0.99 --agent.encoder=impala_small"

$CMD
