import json
roadnet_file = ('/start/manhattan/4/flow_4.json')
f = open(roadnet_file, 'r')
a = json.load(f)
f.close()
c = []
k = 0
# for v in a:
#     v1 = v.copy()
#     c.append(v1)
#     if k< 3:
#         for j in range(1,2):
            
#             v1 = v.copy()
#             v1['startTime'] = v['startTime']+5*j
#             v1['endTime'] = v['endTime']+5*j
#             c.append(v1)
#     k = (k+1)%10
# newFile = ('/start/manhattan/16_3/anon_16_3_newyork_1.3.json')
# b = json.dumps(c)
# f = open(newFile, 'w')
# f.write(b)
# f.close()