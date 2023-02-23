from collections import defaultdict
import json



config_file = '/home/ashutosh/updated_codes/awesome-sogbofa/env/assets/dubins.json'
result_files = {'cem':'/home/ashutosh/updated_codes/awesome-sogbofa/results/continuous-dubins-car-w-velocity-state-v0/evaluation/turtlebot_cem_planning_updates' , \
'mppi' : '/home/ashutosh/updated_codes/awesome-sogbofa/results/continuous-dubins-car-w-velocity-state-v0/evaluation/turtlebot_mppi_planning_updates', \
'DSSPD' : '/home/ashutosh/updated_codes/awesome-sogbofa/results/continuous-dubins-car-w-velocity-state-v0/evaluation/turtlebot_sogbofa_planning_updates'}

MAX_SPEED = 0.3 ## or 0.3

with open(config_file, 'r') as f:
    config_data = f.read()
    config_json = json.loads(config_data)

maps = config_json["maps"]

map_names = maps.keys()

def find_optimal_steps(map_name , num):
    dist = (maps[map_name]['x'] - maps[map_name]['goal_x'])** 2 + (maps[map_name]['y'] - maps[map_name]['goal_y'])** 2 
    dist = dist ** 0.5 

    time_taken = dist / MAX_SPEED

    x = time_taken/ 0.15

    if num == 400:
        return 400

    return int(x)



success_by_map_percentages = defaultdict(dict)
steps_by_map_percentage = defaultdict(dict)

for k in result_files.keys():
    result = result_files[k]

    steps_by_map = defaultdict(list)
    success_by_map = defaultdict(list)

    with open(result + "/logs/output.log" , 'r') as f:
        f = f.readlines()

    for line in f:
        for map in map_names:
            if map == 'cave' or map == 'u':
                continue
            if map + ' ' == line[:len(map) + 1]:
                steps = int(line.replace(' ' , '').split(',')[1])
                steps_by_map[map].append(min(steps , 400))
                success = 1 if steps < 400 else 0 
                success_by_map[map].append(success)


    success_by_map_percentages[k] = {k : round(sum(v)/len(v) , 2) * 100 for k , v in success_by_map.items()}
    steps_by_map_percentage[k] = {k : round(sum(v)/len(v) , 0) for k , v in steps_by_map.items()}


content = ''

for map in map_names:
    if map == "cave" or map == 'u':
        continue
    row = map + '&'
    for alg in ['cem' , 'mppi' , 'DSSPD']:
        row += str(success_by_map_percentages[alg][map]) + '&' + str(steps_by_map_percentage[alg][map]) + '&'
    row = row[:-1] + r'\\ \hline '
    content += row 




content = ''
for alg in ['cem' , 'mppi' , 'DSSPD']:
    success_rate = 0
    steps_rate = 0
    count = 0
    row = alg  + '& '
    for map in map_names:
        if map == 'cave' or map == 'u':
            continue
        success_rate += success_by_map_percentages[alg][map]
        steps_rate += steps_by_map_percentage[alg][map]/find_optimal_steps(map , steps_by_map_percentage[alg][map])
        count += 1
    
    row += str(round(success_rate/count , 2)) + '&' + str(round(steps_rate/count , 2)) + r'\\ \hline '
    content += row

print(content)