import psycopg2 
import numpy as np

def conn_db():
    try:
        conn = psycopg2.connect(database='train_inject', user='yiizy', password='21stCenszMan!', host='yanjing-compute1.cs.uchicago.edu')
    except:
        print('Unable to connect to the database.')
    return conn


class DBStats:
    def __init__(self, network, phase, dataset):
        self.network = network
        self.phase = phase
        self.dataset = dataset
        self.seed = 0.
        self.date = ""
        self.device = ""
        self.stage = ""
        self.fmodel = "" 
        self.fname = ""

        self.final_epoch = -1
        self.final_train_loss = 0.
        self.final_val_loss = 0.
        self.final_train_acc = 0.
        self.final_val_acc = 0.

        self.target_worker = -1
        self.target_layer = ""
        self.target_train_loss = 0.
        self.target_train_acc = 0.
        self.target_epoch = -1
        self.target_step = -1

        self.epoch_nan = -1
        self.step_nan = -1

        self.inj_pos = []
        self.inj_values = []
        self.grad_diffs = []
        self.grad_maxes = []
        self.step_train_losses = []
        self.step_train_acc = []
        self.epoch_train_losses = []
        self.epoch_val_losses = []
        self.epoch_train_acc = []
        self.epoch_val_acc = []

        self.step_max_neurons = []
        self.step_max_gradients = []

        self.kernel_norms = []

    def valid_str(self, d, inlist=False):
        if np.isfinite(d):
            return str(d)
        elif np.isnan(d):
            if not inlist:
                return "\'NAN\'"
            else:
                return "NAN"
        elif np.isinf(d):
            if not inlist:
                return "\'INF\'"
            else:
                return "INF"
        return ""
        

    def list2str(self, l):
        if not len(l):
            return "\'{}\'"
        inj_pos_str = "\'{"
        for elem in l:
            inj_pos_str += self.valid_str(elem, True) + ','
        inj_pos_str = inj_pos_str[:-1] + "}\'"
        return inj_pos_str

    def list2D2str(self, l):
        if not len(l):
            return "\'{}\'"
        inj_pos_str = "\'{"
        for elem in l:
            elem_str = "{"
            for e in elem:
                elem_str += self.valid_str(e, True) + ','
            elem_str = elem_str[:-1] + "}"
            if not len(elem):
                elem_str = "{}"
            inj_pos_str += elem_str + ','
        inj_pos_str = inj_pos_str[:-1] + "}\'"
        return inj_pos_str


    def setvar(self):
        var_dict = vars(self)
        for var in var_dict:
            if var in ['inj_pos']:
                var_dict[var] = self.list2D2str(var_dict[var])
            if var in ['step_max_neurons', 'step_max_gradients', 'inj_values', 'grad_diffs', 'grad_maxes', 'step_train_losses', \
                    'step_train_acc', 'epoch_train_losses', 'epoch_val_losses', \
                    'epoch_train_acc', 'epoch_val_acc', 'kernel_norms']:
                var_dict[var] = self.list2str(var_dict[var])
            if var in ['final_train_loss', 'final_val_loss', 'final_train_acc', 'final_val_acc', 'target_train_loss', 'target_train_acc']:
                var_dict[var] = self.valid_str(var_dict[var])
    

    def push(self):
        conn = conn_db()
        cur = conn.cursor()

        self.setvar()

        if 'replay' in self.phase:
            return

        if self.phase == 'train':
            exec_str = """INSERT INTO golden_train (network, dataset, date, \
                    final_train_loss, final_val_loss, final_train_acc, final_val_acc, \
                    step_train_losses, step_train_acc, epoch_train_losses, epoch_val_losses, \
                    epoch_train_acc, epoch_val_acc, l2_vars, \
                    step_max_neurons, step_max_gradients) VALUES ('{}', '{}', '{}', \
                    {}, {}, {}, {},
                    {}, {}, {}, {},
                    {}, {}, {},
                    {}, {});""".format(
                    self.network, self.dataset, self.date,
                    self.final_train_loss, self.final_val_loss, self.final_train_acc, self.final_val_acc,
                    self.step_train_losses, self.step_train_acc, self.epoch_train_losses, self.epoch_val_losses,
                    self.epoch_train_acc, self.epoch_val_acc, self.kernel_norms,
                    self.step_max_neurons, self.step_max_gradients)

        elif 'inject' in self.phase:
            exec_str = """INSERT INTO new_inject (network, dataset, date, device, stage, seed, \
                    fmodel, fname, final_epoch, final_train_loss, \
                    final_val_loss, final_train_acc, final_val_acc, \
                    target_worker, target_layer, target_train_loss, \
                    target_train_acc, target_epoch, target_step, epoch_nan, step_nan, \
                    inj_pos, inj_values, grad_diffs, grad_maxes,
                    step_train_losses, step_train_acc, epoch_train_losses,\
                    epoch_val_losses, epoch_train_acc, epoch_val_acc,\
                    step_max_neurons, step_max_gradients) \
                    VALUES ('{}', '{}', '{}', '{}', '{}', {},\
                    '{}', '{}', {}, {}, \
                    {}, {}, {}, \
                    {}, '{}', {}, \
                    {}, {}, {}, {}, {}, \
                    {}, {}, {}, {}, \
                    {}, {}, {}, \
                    {}, {}, {}, \
                    {}, {});""".format(
                    self.network, self.dataset, self.date, self.device, self.stage, self.seed, \
                    self.fmodel, self.fname, self.final_epoch, self.final_train_loss, \
                    self.final_val_loss, self.final_train_acc, self.final_val_acc, \
                    self.target_worker, self.target_layer, self.target_train_loss, \
                    self.target_train_acc, self.target_epoch, self.target_step, self.epoch_nan, self.step_nan,
                    self.inj_pos, self.inj_values, self.grad_diffs, self.grad_maxes, \
                    self.step_train_losses, self.step_train_acc, self.epoch_train_losses, \
                    self.epoch_val_losses, self.epoch_train_acc, self.epoch_val_acc,
                    self.step_max_neurons, self.step_max_gradients)
                    #print(exec_str)

        cur.execute(exec_str)
        conn.commit()


 


