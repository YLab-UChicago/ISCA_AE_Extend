import csv

class Injection():
    model = ''
    stage = ''
    fmodel = ''
    target_worker = -1
    target_layer = ''
    target_epoch = -1
    target_step =  -1
    inj_pos = []
    inj_values = []
    learning_rate = -1
    seed = 123

def get_array(string):
    out = []
    for elem in string.split('/'):
        out.append(int(elem))
    return out

def read_injection(file_name):
    inj = Injection()
    with open(file_name, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            name = row[0]

            if name == 'model':
                inj.model = row[1]
            elif name == 'stage':
                inj.stage = row[1]
            elif name == 'fmodel':
                inj.fmodel = row[1]
            elif name == 'target_worker':
                inj.target_worker = int(row[1])
            elif name == 'target_layer':
                inj.target_layer = row[1]
            elif name == 'target_epoch':
                inj.target_epoch = int(row[1])
            elif name == 'target_step':
                inj.target_step = int(row[1])
            elif name == 'learning_rate':
                inj.learning_rate = float(row[1])
            elif name == 'inj_pos':
                for i in range(1, len(row)):
                    inj.inj_pos.append(get_array(row[i]))
            elif name == 'inj_values':
                for i in range(1, len(row)):
                    inj.inj_values.append(float(row[i]))
    return inj

'''
inj = read_injection("injections/example.csv")
print(inj.stage)
print(inj.fmodel)
print(inj.target_worker)
print(inj.target_layer)
print(inj.target_epoch)
print(inj.target_step)
print(inj.inj_pos)
print(inj.inj_values)
'''
