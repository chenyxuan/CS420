from deeplight_agent import DeepLightAgent
from sumo_agent import SumoAgent
import time

deeplight = DeepLightAgent(num_phases = 9)
round = 1
step = 4000

deeplight.load_model('train_0_final')

start_time = time.time()
for rnd in range(round):
    s_agent = SumoAgent()
    for i in range(step):
        state_dict = s_agent.get_observation()

        action_dict = {}
        for key, value in state_dict.items():
            action = deeplight.choose(value, certain = True)
            action_dict[key] = action

        reward_dict = s_agent.take_action(action_dict)

        next_state_dict = s_agent.get_observation()

        for key, value in state_dict.items():
            deeplight.remeber(state_dict[key],
                              action_dict[key],
                              reward_dict[key],
                              next_state_dict[key])

        deeplight.update_network(s_agent.get_time(), rnd == 0)
        deeplight.update_network_bar()
        if i % 100 == 0:
            print rnd, i
            print time.time() - start_time
            print s_agent.get_score()
            start_time = time.time()
            deeplight.save_model("train_{0}_{1}".format(rnd, i))
    deeplight.save_model("train_{0}_final".format(rnd))