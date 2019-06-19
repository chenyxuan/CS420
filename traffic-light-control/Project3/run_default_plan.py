import cityflow

config = "data/2x2_demo/config.json"

num_step = 6000
eng = cityflow.Engine(config, thread_num=4)

for step in range(num_step):
    eng.next_step()

    current_time = eng.get_current_time()  # return a double, time past in seconds
    lane_vehicle_count = eng.get_lane_vehicle_count()  # return a dict, {lane_id: lane_count, ...}
    lane_waiting_vehicle_count = eng.get_lane_waiting_vehicle_count()  # return a dict, {lane_id: lane_waiting_count, ...}
    lane_vehicles = eng.get_lane_vehicles()  # return a dict, {lane_id: [vehicle1_id, vehicle2_id, ...], ...}
    vehicle_speed = eng.get_vehicle_speed()  # return a dict, {vehicle_id: vehicle_speed, ...}
    vehicle_distance = eng.get_vehicle_distance()  # return a dict, {vehicle_id: vehicle_distance, ...}
    if step % 100 == 0:
    	print step
    	print(eng.get_score())

print num_step
print(eng.get_score())

