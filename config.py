dic_agent_conf = {
    "ACTOR_LEARNING_RATE": 0.001,
    "CRITIC_LEARNING_RATE": 0.001,
    "UPDATE_PERIOD": 300,
    "BATCH_SIZE": 20,
    "GAMMA": 0.9,
    "PATIENCE": 10,
    "NUM_LAYERS": 2,
    "D_DENSE": 32,
    "ACTOR_LOSS": "Clipped",  # or "KL-DIVERGENCE"
    "CRITIC_LOSS": "mean_squared_error",
    "OPTIMIZER": "Adam"
}

dic_env_conf = {
    "LIST_FEATURE_NAME": [],
    "D_FEATURE_NAME": [],
    "ACTION_DIM": 2,
    "ACTION_RANGE": "0-1", # or "-1~1"
}

dic_path ={

}

dic_exp_conf = {

}