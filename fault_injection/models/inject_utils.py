from enum import Enum
import tensorflow as tf
import numpy as np
import struct

class InjType(Enum):
    INPUT = 1
    INPUT_16 = 2
    WT = 3
    WT_16 = 4
    RBFLIP = 5
    RD = 6
    RD_CORRECT = 7
    ZERO = 8
    N16_RD = 9
    N16_RD_CORRECT = 10
    RD_GLB = 11
    RD_CORRECT_GLB = 12
    N64_INPUT = 13
    N64_WT = 14
    N64_INPUT_16 = 15
    N64_WT_16 = 16
    N64_INPUT_GLB = 17
    N64_WT_GLB = 18


def is_bflip(inj_type):
    return inj_type in [InjType.INPUT, InjType.INPUT_16, InjType.WT, InjType.WT_16, InjType.RBFLIP]

def is_random(inj_type):
    return inj_type in [InjType.RD, InjType.N16_RD, InjType.RD_GLB]

def is_zero(inj_type):
    return inj_type in [InjType.ZERO]

def is_correct(inj_type):
    return inj_type in [InjType.RD_CORRECT, InjType.N16_RD_CORRECT, InjType.RD_CORRECT_GLB, InjType.N64_INPUT, InjType.N64_WT, InjType.N64_INPUT_16, InjType.N64_WT_16, InjType.N64_INPUT_GLB, InjType.N64_WT_GLB]


def is_input_target(inj_type):
    return inj_type in [InjType.INPUT, InjType.INPUT_16, InjType.N64_INPUT, InjType.N64_INPUT_16, InjType.N64_INPUT_GLB]

def is_weight_target(inj_type):
    return inj_type in [InjType.WT, InjType.WT_16, InjType.N64_WT, InjType.N64_WT_16, InjType.N64_WT_GLB]

def is_output_target(inj_type):
    return inj_type in [InjType.RD, InjType.RBFLIP, InjType.RD_CORRECT, InjType.ZERO, InjType.N16_RD, InjType.N16_RD_CORRECT, InjType.RD_GLB, InjType.RD_CORRECT_GLB]


def num_inj(inj_type):
    if inj_type in [InjType.INPUT, InjType.INPUT_16, InjType.WT, InjType.WT_16, InjType.RD, InjType.RBFLIP, InjType.RD_CORRECT, InjType.ZERO]:
        return 1, 1
    elif inj_type in [InjType.N16_RD, InjType.N16_RD_CORRECT]:
        return 16, 1
    elif inj_type in [InjType.RD_GLB, InjType.RD_CORRECT_GLB]:
        return 16, 1000 
    elif inj_type in [InjType.N64_INPUT, InjType.N64_WT, InjType.N64_INPUT_16, InjType.N64_WT_16]:
        return 64, 1
    elif inj_type in [InjType.N64_INPUT_GLB, InjType.N64_WT_GLB]:
        return 64, 1000


class InjState(Enum):
    IDLE = 1
    GOLDEN = 2
    INJECT = 3


class InjArgs():
    def __init__(self, inj_replica, inj_layer, inj_type, golden_weights, golden_output, mask=None, delta=None):
        if not isinstance(inj_type, InjType):
            print("ERROR: Invalid injection type!")
            exit(12)
        self.inj_replica = inj_replica
        self.inj_layer = inj_layer
        self.inj_type = inj_type
        self.golden_weights = golden_weights
        self.golden_output = golden_output

        # Two numpy arrays
        self.inj_mask = mask
        self.inj_delta = delta


def record(train_recorder, text):
    if train_recorder:
        train_recorder.write(text)
        train_recorder.flush()


def bin2fp32(bin_str):
    assert len(bin_str) == 32
    return struct.unpack('!f',struct.pack('!I', int(bin_str, 2)))[0]

def fp322bin(value):
    return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', value))


def choose_inj_pos(target, inj_type, train_recorder, db_st):
    np.random.seed(None)
    shape = target.shape


    def random_positions(shape, train_recorder, n_inj, n_repeat):
        positions = []
        l = len(shape)

        total = np.product(shape) / n_inj
        start = np.random.randint(total)
        end = np.random.randint(start+1, total) if n_repeat != 1 else start + 1
        start_pos = np.unravel_index(start * n_inj, shape)
        end_pos = np.unravel_index(end * n_inj, shape)
        for flat in range(start * n_inj, end * n_inj):
            positions.append(np.unravel_index(flat, shape))
        return positions

        record(train_recorder, "Start injection at position {}\n".format(start_pos))
        record(train_recorder, "End injection at position {}\n".format(end_pos))

    n_inj, n_repeat = num_inj(inj_type)
    positions = random_positions(shape, train_recorder, n_inj, n_repeat)

    def flip_one_bit(target):
        np.random.seed(None)
        bin_target = fp322bin(target)
        flip = np.random.randint(32)
        bin_output = ""
        for i in range(32):
            if i == flip:
                bin_output += ('1' if bin_target[i] == '0' else '0')
            else:
                bin_output += bin_target[i]
        return bin2fp32(bin_output) 

    def get_random_value():
        np.random.seed(None)
        one_bin = ''
        result = 0
        while one_bin == '' or not np.isfinite(result):
            one_bin = ''
            for _ in range(32):
                one_bin += str(np.random.randint(0,2))
            result = struct.unpack('!f',struct.pack('!I', int(one_bin, 2)))[0]
        return result

    def get_random_correct(target):
        shape = target.shape
        rd_pos = np.unravel_index(np.random.randint(np.product(shape)), shape)
        return target[rd_pos]

    mask = np.ones(shape)
    delta = np.zeros(shape)

    db_st.inj_pos = positions
    for pos in positions:
        ori_val = target[pos]

        if is_random(inj_type):
            val_delta = get_random_value()
        elif is_bflip(inj_type):
            val_delta = flip_one_bit(ori_val)
        elif is_correct(inj_type):
            val_delta = get_random_correct(target)
        else:
            val_delta = 0

        # Modified for worst case
        # Randomly choose a value between 1e16 to 1.3e18
        #val_delta = np.random.uniform(low=1e16, high=1e18)
        mask[pos] = 0
        delta[pos] = val_delta

        record(train_recorder, "Position is {}, Golden data is {}, inject data is {}\n".format(pos, ori_val, val_delta))
        db_st.inj_values.append(val_delta)

    return mask, delta




def choose_random_layer(model, phase):

    resnet18_fwrd_layers = [
            "conv1",
            "basicblock_1_basic_0_conv1",
            "basicblock_1_basic_0_conv2",
            "basicblock_1_basic_1_conv1",
            "basicblock_1_basic_1_conv2",
            "basicblock_2_basic_0_downsample",
            "basicblock_2_basic_0_conv1",
            "basicblock_2_basic_0_conv2",
            "basicblock_2_basic_1_conv1",
            "basicblock_2_basic_1_conv2",
            "basicblock_3_basic_0_downsample",
            "basicblock_3_basic_0_conv1",
            "basicblock_3_basic_0_conv2",
            "basicblock_3_basic_1_conv1",
            "basicblock_3_basic_1_conv2",
            "basicblock_4_basic_0_downsample",
            "basicblock_4_basic_0_conv1",
            "basicblock_4_basic_0_conv2",
            "basicblock_4_basic_1_conv1",
            "basicblock_4_basic_1_conv2"
            ]

    resnet18_bkwd_layers = [
            "basicblock_4_basic_1_conv2_grad_in",
            "basicblock_4_basic_1_conv2_grad_wt",
            "basicblock_4_basic_1_conv1_grad_in",
            "basicblock_4_basic_1_conv1_grad_wt",
            "basicblock_4_basic_0_downsample_grad_in",
            "basicblock_4_basic_0_downsample_grad_wt",
            "basicblock_4_basic_0_conv2_grad_in",
            "basicblock_4_basic_0_conv2_grad_wt",
            "basicblock_4_basic_0_conv1_grad_in",
            "basicblock_4_basic_0_conv1_grad_wt",
            "basicblock_3_basic_1_conv2_grad_in",
            "basicblock_3_basic_1_conv2_grad_wt",
            "basicblock_3_basic_1_conv1_grad_in",
            "basicblock_3_basic_1_conv1_grad_wt",
            "basicblock_3_basic_0_downsample_grad_in",
            "basicblock_3_basic_0_downsample_grad_wt",
            "basicblock_3_basic_0_conv2_grad_in",
            "basicblock_3_basic_0_conv2_grad_wt",
            "basicblock_3_basic_0_conv1_grad_in",
            "basicblock_3_basic_0_conv1_grad_wt",
            "basicblock_2_basic_1_conv2_grad_in",
            "basicblock_2_basic_1_conv2_grad_wt",
            "basicblock_2_basic_1_conv1_grad_in",
            "basicblock_2_basic_1_conv1_grad_wt",
            "basicblock_2_basic_0_downsample_grad_in",
            "basicblock_2_basic_0_downsample_grad_wt",
            "basicblock_2_basic_0_conv2_grad_in",
            "basicblock_2_basic_0_conv2_grad_wt",
            "basicblock_2_basic_0_conv1_grad_in",
            "basicblock_2_basic_0_conv1_grad_wt",
            "basicblock_1_basic_1_conv2_grad_in",
            "basicblock_1_basic_1_conv2_grad_wt",
            "basicblock_1_basic_1_conv1_grad_in",
            "basicblock_1_basic_1_conv1_grad_wt",
            "basicblock_1_basic_0_conv2_grad_in",
            "basicblock_1_basic_0_conv2_grad_wt",
            "basicblock_1_basic_0_conv1_grad_in",
            "basicblock_1_basic_0_conv1_grad_wt",
            "conv1_grad_in",
            "conv1_grad_wt"
            ]

    effnet_fwrd_layers = [
            "conv2",
            "block7_mbconv_0_conv2",
            "block7_mbconv_0_se_expand_conv",
            "block7_mbconv_0_se_reduce_conv",
            "block7_mbconv_0_conv1",
            "block6_mbconv_3_conv2",
            "block6_mbconv_3_se_expand_conv",
            "block6_mbconv_3_se_reduce_conv",
            "block6_mbconv_3_conv1",
            "block6_mbconv_2_conv2",
            "block6_mbconv_2_se_expand_conv",
            "block6_mbconv_2_se_reduce_conv",
            "block6_mbconv_2_conv1",
            "block6_mbconv_1_conv2",
            "block6_mbconv_1_se_expand_conv",
            "block6_mbconv_1_se_reduce_conv",
            "block6_mbconv_1_conv1",
            "block6_mbconv_0_conv2",
            "block6_mbconv_0_se_expand_conv",
            "block6_mbconv_0_se_reduce_conv",
            "block6_mbconv_0_conv1",
            "block5_mbconv_2_conv2",
            "block5_mbconv_2_se_expand_conv",
            "block5_mbconv_2_se_reduce_conv",
            "block5_mbconv_2_conv1",
            "block5_mbconv_1_conv2",
            "block5_mbconv_1_se_expand_conv",
            "block5_mbconv_1_se_reduce_conv",
            "block5_mbconv_1_conv1",
            "block5_mbconv_0_conv2",
            "block5_mbconv_0_se_expand_conv",
            "block5_mbconv_0_se_reduce_conv",
            "block5_mbconv_0_conv1",
            "block4_mbconv_2_conv2",
            "block4_mbconv_2_se_expand_conv",
            "block4_mbconv_2_se_reduce_conv",
            "block4_mbconv_2_conv1",
            "block4_mbconv_1_conv2",
            "block4_mbconv_1_se_expand_conv",
            "block4_mbconv_1_se_reduce_conv",
            "block4_mbconv_1_conv1",
            "block4_mbconv_0_conv2",
            "block4_mbconv_0_se_expand_conv",
            "block4_mbconv_0_se_reduce_conv",
            "block4_mbconv_0_conv1",
            "block3_mbconv_1_conv2",
            "block3_mbconv_1_se_expand_conv",
            "block3_mbconv_1_se_reduce_conv",
            "block3_mbconv_1_conv1",
            "block3_mbconv_0_conv2",
            "block3_mbconv_0_se_expand_conv",
            "block3_mbconv_0_se_reduce_conv",
            "block3_mbconv_0_conv1",
            "block2_mbconv_1_conv2",
            "block2_mbconv_1_se_expand_conv",
            "block2_mbconv_1_se_reduce_conv",
            "block2_mbconv_1_conv1",
            "block2_mbconv_0_conv2",
            "block2_mbconv_0_se_expand_conv",
            "block2_mbconv_0_se_reduce_conv",
            "block2_mbconv_0_conv1",
            "block1_mbconv_0_conv2",
            "block1_mbconv_0_se_expand_conv",
            "block1_mbconv_0_se_reduce_conv",
            "block1_mbconv_0_conv1",
            "conv1"
    ]

    effnet_bkwd_layers = [
            "conv2_grad_in",
            "block7_mbconv_0_conv2_grad_in",
            "block7_mbconv_0_se_expand_conv_grad_in",
            "block7_mbconv_0_se_reduce_conv_grad_in",
            "block7_mbconv_0_conv1_grad_in",
            "block6_mbconv_3_conv2_grad_in",
            "block6_mbconv_3_se_expand_conv_grad_in",
            "block6_mbconv_3_se_reduce_conv_grad_in",
            "block6_mbconv_3_conv1_grad_in",
            "block6_mbconv_2_conv2_grad_in",
            "block6_mbconv_2_se_expand_conv_grad_in",
            "block6_mbconv_2_se_reduce_conv_grad_in",
            "block6_mbconv_2_conv1_grad_in",
            "block6_mbconv_1_conv2_grad_in",
            "block6_mbconv_1_se_expand_conv_grad_in",
            "block6_mbconv_1_se_reduce_conv_grad_in",
            "block6_mbconv_1_conv1_grad_in",
            "block6_mbconv_0_conv2_grad_in",
            "block6_mbconv_0_se_expand_conv_grad_in",
            "block6_mbconv_0_se_reduce_conv_grad_in",
            "block6_mbconv_0_conv1_grad_in",
            "block5_mbconv_2_conv2_grad_in",
            "block5_mbconv_2_se_expand_conv_grad_in",
            "block5_mbconv_2_se_reduce_conv_grad_in",
            "block5_mbconv_2_conv1_grad_in",
            "block5_mbconv_1_conv2_grad_in",
            "block5_mbconv_1_se_expand_conv_grad_in",
            "block5_mbconv_1_se_reduce_conv_grad_in",
            "block5_mbconv_1_conv1_grad_in",
            "block5_mbconv_0_conv2_grad_in",
            "block5_mbconv_0_se_expand_conv_grad_in",
            "block5_mbconv_0_se_reduce_conv_grad_in",
            "block5_mbconv_0_conv1_grad_in",
            "block4_mbconv_2_conv2_grad_in",
            "block4_mbconv_2_se_expand_conv_grad_in",
            "block4_mbconv_2_se_reduce_conv_grad_in",
            "block4_mbconv_2_conv1_grad_in",
            "block4_mbconv_1_conv2_grad_in",
            "block4_mbconv_1_se_expand_conv_grad_in",
            "block4_mbconv_1_se_reduce_conv_grad_in",
            "block4_mbconv_1_conv1_grad_in",
            "block4_mbconv_0_conv2_grad_in",
            "block4_mbconv_0_se_expand_conv_grad_in",
            "block4_mbconv_0_se_reduce_conv_grad_in",
            "block4_mbconv_0_conv1_grad_in",
            "block3_mbconv_1_conv2_grad_in",
            "block3_mbconv_1_se_expand_conv_grad_in",
            "block3_mbconv_1_se_reduce_conv_grad_in",
            "block3_mbconv_1_conv1_grad_in",
            "block3_mbconv_0_conv2_grad_in",
            "block3_mbconv_0_se_expand_conv_grad_in",
            "block3_mbconv_0_se_reduce_conv_grad_in",
            "block3_mbconv_0_conv1_grad_in",
            "block2_mbconv_1_conv2_grad_in",
            "block2_mbconv_1_se_expand_conv_grad_in",
            "block2_mbconv_1_se_reduce_conv_grad_in",
            "block2_mbconv_1_conv1_grad_in",
            "block2_mbconv_0_conv2_grad_in",
            "block2_mbconv_0_se_expand_conv_grad_in",
            "block2_mbconv_0_se_reduce_conv_grad_in",
            "block2_mbconv_0_conv1_grad_in",
            "block1_mbconv_0_conv2_grad_in",
            "block1_mbconv_0_se_expand_conv_grad_in",
            "block1_mbconv_0_se_reduce_conv_grad_in",
            "block1_mbconv_0_conv1_grad_in",
            "conv1_grad_in",

            "conv2_grad_wt",
            "block7_mbconv_0_conv2_grad_wt",
            "block7_mbconv_0_se_expand_conv_grad_wt",
            "block7_mbconv_0_se_reduce_conv_grad_wt",
            "block7_mbconv_0_conv1_grad_wt",
            "block6_mbconv_3_conv2_grad_wt",
            "block6_mbconv_3_se_expand_conv_grad_wt",
            "block6_mbconv_3_se_reduce_conv_grad_wt",
            "block6_mbconv_3_conv1_grad_wt",
            "block6_mbconv_2_conv2_grad_wt",
            "block6_mbconv_2_se_expand_conv_grad_wt",
            "block6_mbconv_2_se_reduce_conv_grad_wt",
            "block6_mbconv_2_conv1_grad_wt",
            "block6_mbconv_1_conv2_grad_wt",
            "block6_mbconv_1_se_expand_conv_grad_wt",
            "block6_mbconv_1_se_reduce_conv_grad_wt",
            "block6_mbconv_1_conv1_grad_wt",
            "block6_mbconv_0_conv2_grad_wt",
            "block6_mbconv_0_se_expand_conv_grad_wt",
            "block6_mbconv_0_se_reduce_conv_grad_wt",
            "block6_mbconv_0_conv1_grad_wt",
            "block5_mbconv_2_conv2_grad_wt",
            "block5_mbconv_2_se_expand_conv_grad_wt",
            "block5_mbconv_2_se_reduce_conv_grad_wt",
            "block5_mbconv_2_conv1_grad_wt",
            "block5_mbconv_1_conv2_grad_wt",
            "block5_mbconv_1_se_expand_conv_grad_wt",
            "block5_mbconv_1_se_reduce_conv_grad_wt",
            "block5_mbconv_1_conv1_grad_wt",
            "block5_mbconv_0_conv2_grad_wt",
            "block5_mbconv_0_se_expand_conv_grad_wt",
            "block5_mbconv_0_se_reduce_conv_grad_wt",
            "block5_mbconv_0_conv1_grad_wt",
            "block4_mbconv_2_conv2_grad_wt",
            "block4_mbconv_2_se_expand_conv_grad_wt",
            "block4_mbconv_2_se_reduce_conv_grad_wt",
            "block4_mbconv_2_conv1_grad_wt",
            "block4_mbconv_1_conv2_grad_wt",
            "block4_mbconv_1_se_expand_conv_grad_wt",
            "block4_mbconv_1_se_reduce_conv_grad_wt",
            "block4_mbconv_1_conv1_grad_wt",
            "block4_mbconv_0_conv2_grad_wt",
            "block4_mbconv_0_se_expand_conv_grad_wt",
            "block4_mbconv_0_se_reduce_conv_grad_wt",
            "block4_mbconv_0_conv1_grad_wt",
            "block3_mbconv_1_conv2_grad_wt",
            "block3_mbconv_1_se_expand_conv_grad_wt",
            "block3_mbconv_1_se_reduce_conv_grad_wt",
            "block3_mbconv_1_conv1_grad_wt",
            "block3_mbconv_0_conv2_grad_wt",
            "block3_mbconv_0_se_expand_conv_grad_wt",
            "block3_mbconv_0_se_reduce_conv_grad_wt",
            "block3_mbconv_0_conv1_grad_wt",
            "block2_mbconv_1_conv2_grad_wt",
            "block2_mbconv_1_se_expand_conv_grad_wt",
            "block2_mbconv_1_se_reduce_conv_grad_wt",
            "block2_mbconv_1_conv1_grad_wt",
            "block2_mbconv_0_conv2_grad_wt",
            "block2_mbconv_0_se_expand_conv_grad_wt",
            "block2_mbconv_0_se_reduce_conv_grad_wt",
            "block2_mbconv_0_conv1_grad_wt",
            "block1_mbconv_0_conv2_grad_wt",
            "block1_mbconv_0_se_expand_conv_grad_wt",
            "block1_mbconv_0_se_reduce_conv_grad_wt",
            "block1_mbconv_0_conv1_grad_wt",
            "conv1_grad_wt"
    ]

    densenet_fwrd_layers = [
            "densenet_conv",
            "densenet_dense_block_1_bottleneck_0_conv1",
            "densenet_dense_block_1_bottleneck_0_conv2",
            "densenet_dense_block_1_bottleneck_1_conv1",
            "densenet_dense_block_1_bottleneck_1_conv2",
            "densenet_dense_block_1_bottleneck_2_conv1",
            "densenet_dense_block_1_bottleneck_2_conv2",
            "densenet_dense_block_1_bottleneck_3_conv1",
            "densenet_dense_block_1_bottleneck_3_conv2",
            "densenet_dense_block_1_bottleneck_4_conv1",
            "densenet_dense_block_1_bottleneck_4_conv2",
            "densenet_dense_block_1_bottleneck_5_conv1",
            "densenet_dense_block_1_bottleneck_5_conv2",
            "densenet_transition_1_conv",
            "densenet_dense_block_2_bottleneck_0_conv1",
            "densenet_dense_block_2_bottleneck_0_conv2",
            "densenet_dense_block_2_bottleneck_1_conv1",
            "densenet_dense_block_2_bottleneck_1_conv2",
            "densenet_dense_block_2_bottleneck_2_conv1",
            "densenet_dense_block_2_bottleneck_2_conv2",
            "densenet_dense_block_2_bottleneck_3_conv1",
            "densenet_dense_block_2_bottleneck_3_conv2",
            "densenet_dense_block_2_bottleneck_4_conv1",
            "densenet_dense_block_2_bottleneck_4_conv2",
            "densenet_dense_block_2_bottleneck_5_conv1",
            "densenet_dense_block_2_bottleneck_5_conv2",
            "densenet_dense_block_2_bottleneck_6_conv1",
            "densenet_dense_block_2_bottleneck_6_conv2",
            "densenet_dense_block_2_bottleneck_7_conv1",
            "densenet_dense_block_2_bottleneck_7_conv2",
            "densenet_dense_block_2_bottleneck_8_conv1",
            "densenet_dense_block_2_bottleneck_8_conv2",
            "densenet_dense_block_2_bottleneck_9_conv1",
            "densenet_dense_block_2_bottleneck_9_conv2",
            "densenet_dense_block_2_bottleneck_10_conv1",
            "densenet_dense_block_2_bottleneck_10_conv2",
            "densenet_dense_block_2_bottleneck_11_conv1",
            "densenet_dense_block_2_bottleneck_11_conv2",
            "densenet_transition_2_conv",
            "densenet_dense_block_3_bottleneck_0_conv1",
            "densenet_dense_block_3_bottleneck_0_conv2",
            "densenet_dense_block_3_bottleneck_1_conv1",
            "densenet_dense_block_3_bottleneck_1_conv2",
            "densenet_dense_block_3_bottleneck_2_conv1",
            "densenet_dense_block_3_bottleneck_2_conv2",
            "densenet_dense_block_3_bottleneck_3_conv1",
            "densenet_dense_block_3_bottleneck_3_conv2",
            "densenet_dense_block_3_bottleneck_4_conv1",
            "densenet_dense_block_3_bottleneck_4_conv2",
            "densenet_dense_block_3_bottleneck_5_conv1",
            "densenet_dense_block_3_bottleneck_5_conv2",
            "densenet_dense_block_3_bottleneck_6_conv1",
            "densenet_dense_block_3_bottleneck_6_conv2",
            "densenet_dense_block_3_bottleneck_7_conv1",
            "densenet_dense_block_3_bottleneck_7_conv2",
            "densenet_dense_block_3_bottleneck_8_conv1",
            "densenet_dense_block_3_bottleneck_8_conv2",
            "densenet_dense_block_3_bottleneck_9_conv1",
            "densenet_dense_block_3_bottleneck_9_conv2",
            "densenet_dense_block_3_bottleneck_10_conv1",
            "densenet_dense_block_3_bottleneck_10_conv2",
            "densenet_dense_block_3_bottleneck_11_conv1",
            "densenet_dense_block_3_bottleneck_11_conv2",
            "densenet_dense_block_3_bottleneck_12_conv1",
            "densenet_dense_block_3_bottleneck_12_conv2",
            "densenet_dense_block_3_bottleneck_13_conv1",
            "densenet_dense_block_3_bottleneck_13_conv2",
            "densenet_dense_block_3_bottleneck_14_conv1",
            "densenet_dense_block_3_bottleneck_14_conv2",
            "densenet_dense_block_3_bottleneck_15_conv1",
            "densenet_dense_block_3_bottleneck_15_conv2",
            "densenet_dense_block_3_bottleneck_16_conv1",
            "densenet_dense_block_3_bottleneck_16_conv2",
            "densenet_dense_block_3_bottleneck_17_conv1",
            "densenet_dense_block_3_bottleneck_17_conv2",
            "densenet_dense_block_3_bottleneck_18_conv1",
            "densenet_dense_block_3_bottleneck_18_conv2",
            "densenet_dense_block_3_bottleneck_19_conv1",
            "densenet_dense_block_3_bottleneck_19_conv2",
            "densenet_dense_block_3_bottleneck_20_conv1",
            "densenet_dense_block_3_bottleneck_20_conv2",
            "densenet_dense_block_3_bottleneck_21_conv1",
            "densenet_dense_block_3_bottleneck_21_conv2",
            "densenet_dense_block_3_bottleneck_22_conv1",
            "densenet_dense_block_3_bottleneck_22_conv2",
            "densenet_dense_block_3_bottleneck_23_conv1",
            "densenet_dense_block_3_bottleneck_23_conv2",
            "densenet_transition_3_conv",
            "densenet_dense_block_4_bottleneck_0_conv1",
            "densenet_dense_block_4_bottleneck_0_conv2",
            "densenet_dense_block_4_bottleneck_1_conv1",
            "densenet_dense_block_4_bottleneck_1_conv2",
            "densenet_dense_block_4_bottleneck_2_conv1",
            "densenet_dense_block_4_bottleneck_2_conv2",
            "densenet_dense_block_4_bottleneck_3_conv1",
            "densenet_dense_block_4_bottleneck_3_conv2",
            "densenet_dense_block_4_bottleneck_4_conv1",
            "densenet_dense_block_4_bottleneck_4_conv2",
            "densenet_dense_block_4_bottleneck_5_conv1",
            "densenet_dense_block_4_bottleneck_5_conv2",
            "densenet_dense_block_4_bottleneck_6_conv1",
            "densenet_dense_block_4_bottleneck_6_conv2",
            "densenet_dense_block_4_bottleneck_7_conv1",
            "densenet_dense_block_4_bottleneck_7_conv2",
            "densenet_dense_block_4_bottleneck_8_conv1",
            "densenet_dense_block_4_bottleneck_8_conv2",
            "densenet_dense_block_4_bottleneck_9_conv1",
            "densenet_dense_block_4_bottleneck_9_conv2",
            "densenet_dense_block_4_bottleneck_10_conv1",
            "densenet_dense_block_4_bottleneck_10_conv2",
            "densenet_dense_block_4_bottleneck_11_conv1",
            "densenet_dense_block_4_bottleneck_11_conv2",
            "densenet_dense_block_4_bottleneck_12_conv1",
            "densenet_dense_block_4_bottleneck_12_conv2",
            "densenet_dense_block_4_bottleneck_13_conv1",
            "densenet_dense_block_4_bottleneck_13_conv2",
            "densenet_dense_block_4_bottleneck_14_conv1",
            "densenet_dense_block_4_bottleneck_14_conv2",
            "densenet_dense_block_4_bottleneck_15_conv1",
            "densenet_dense_block_4_bottleneck_15_conv2"
    ]

    densenet_bkwd_layers = [
            "densenet_conv_grad_in",
            "densenet_dense_block_1_bottleneck_0_conv1_grad_in",
            "densenet_dense_block_1_bottleneck_0_conv2_grad_in",
            "densenet_dense_block_1_bottleneck_1_conv1_grad_in",
            "densenet_dense_block_1_bottleneck_1_conv2_grad_in",
            "densenet_dense_block_1_bottleneck_2_conv1_grad_in",
            "densenet_dense_block_1_bottleneck_2_conv2_grad_in",
            "densenet_dense_block_1_bottleneck_3_conv1_grad_in",
            "densenet_dense_block_1_bottleneck_3_conv2_grad_in",
            "densenet_dense_block_1_bottleneck_4_conv1_grad_in",
            "densenet_dense_block_1_bottleneck_4_conv2_grad_in",
            "densenet_dense_block_1_bottleneck_5_conv1_grad_in",
            "densenet_dense_block_1_bottleneck_5_conv2_grad_in",
            "densenet_transition_1_conv_grad_in",
            "densenet_dense_block_2_bottleneck_0_conv1_grad_in",
            "densenet_dense_block_2_bottleneck_0_conv2_grad_in",
            "densenet_dense_block_2_bottleneck_1_conv1_grad_in",
            "densenet_dense_block_2_bottleneck_1_conv2_grad_in",
            "densenet_dense_block_2_bottleneck_2_conv1_grad_in",
            "densenet_dense_block_2_bottleneck_2_conv2_grad_in",
            "densenet_dense_block_2_bottleneck_3_conv1_grad_in",
            "densenet_dense_block_2_bottleneck_3_conv2_grad_in",
            "densenet_dense_block_2_bottleneck_4_conv1_grad_in",
            "densenet_dense_block_2_bottleneck_4_conv2_grad_in",
            "densenet_dense_block_2_bottleneck_5_conv1_grad_in",
            "densenet_dense_block_2_bottleneck_5_conv2_grad_in",
            "densenet_dense_block_2_bottleneck_6_conv1_grad_in",
            "densenet_dense_block_2_bottleneck_6_conv2_grad_in",
            "densenet_dense_block_2_bottleneck_7_conv1_grad_in",
            "densenet_dense_block_2_bottleneck_7_conv2_grad_in",
            "densenet_dense_block_2_bottleneck_8_conv1_grad_in",
            "densenet_dense_block_2_bottleneck_8_conv2_grad_in",
            "densenet_dense_block_2_bottleneck_9_conv1_grad_in",
            "densenet_dense_block_2_bottleneck_9_conv2_grad_in",
            "densenet_dense_block_2_bottleneck_10_conv1_grad_in",
            "densenet_dense_block_2_bottleneck_10_conv2_grad_in",
            "densenet_dense_block_2_bottleneck_11_conv1_grad_in",
            "densenet_dense_block_2_bottleneck_11_conv2_grad_in",
            "densenet_transition_2_conv_grad_in",
            "densenet_dense_block_3_bottleneck_0_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_0_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_1_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_1_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_2_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_2_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_3_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_3_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_4_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_4_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_5_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_5_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_6_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_6_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_7_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_7_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_8_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_8_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_9_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_9_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_10_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_10_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_11_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_11_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_12_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_12_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_13_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_13_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_14_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_14_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_15_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_15_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_16_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_16_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_17_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_17_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_18_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_18_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_19_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_19_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_20_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_20_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_21_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_21_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_22_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_22_conv2_grad_in",
            "densenet_dense_block_3_bottleneck_23_conv1_grad_in",
            "densenet_dense_block_3_bottleneck_23_conv2_grad_in",
            "densenet_transition_3_conv_grad_in",
            "densenet_dense_block_4_bottleneck_0_conv1_grad_in",
            "densenet_dense_block_4_bottleneck_0_conv2_grad_in",
            "densenet_dense_block_4_bottleneck_1_conv1_grad_in",
            "densenet_dense_block_4_bottleneck_1_conv2_grad_in",
            "densenet_dense_block_4_bottleneck_2_conv1_grad_in",
            "densenet_dense_block_4_bottleneck_2_conv2_grad_in",
            "densenet_dense_block_4_bottleneck_3_conv1_grad_in",
            "densenet_dense_block_4_bottleneck_3_conv2_grad_in",
            "densenet_dense_block_4_bottleneck_4_conv1_grad_in",
            "densenet_dense_block_4_bottleneck_4_conv2_grad_in",
            "densenet_dense_block_4_bottleneck_5_conv1_grad_in",
            "densenet_dense_block_4_bottleneck_5_conv2_grad_in",
            "densenet_dense_block_4_bottleneck_6_conv1_grad_in",
            "densenet_dense_block_4_bottleneck_6_conv2_grad_in",
            "densenet_dense_block_4_bottleneck_7_conv1_grad_in",
            "densenet_dense_block_4_bottleneck_7_conv2_grad_in",
            "densenet_dense_block_4_bottleneck_8_conv1_grad_in",
            "densenet_dense_block_4_bottleneck_8_conv2_grad_in",
            "densenet_dense_block_4_bottleneck_9_conv1_grad_in",
            "densenet_dense_block_4_bottleneck_9_conv2_grad_in",
            "densenet_dense_block_4_bottleneck_10_conv1_grad_in",
            "densenet_dense_block_4_bottleneck_10_conv2_grad_in",
            "densenet_dense_block_4_bottleneck_11_conv1_grad_in",
            "densenet_dense_block_4_bottleneck_11_conv2_grad_in",
            "densenet_dense_block_4_bottleneck_12_conv1_grad_in",
            "densenet_dense_block_4_bottleneck_12_conv2_grad_in",
            "densenet_dense_block_4_bottleneck_13_conv1_grad_in",
            "densenet_dense_block_4_bottleneck_13_conv2_grad_in",
            "densenet_dense_block_4_bottleneck_14_conv1_grad_in",
            "densenet_dense_block_4_bottleneck_14_conv2_grad_in",
            "densenet_dense_block_4_bottleneck_15_conv1_grad_in",
            "densenet_dense_block_4_bottleneck_15_conv2_grad_in", 

            "densenet_conv_grad_wt",
            "densenet_dense_block_1_bottleneck_0_conv1_grad_wt",
            "densenet_dense_block_1_bottleneck_0_conv2_grad_wt",
            "densenet_dense_block_1_bottleneck_1_conv1_grad_wt",
            "densenet_dense_block_1_bottleneck_1_conv2_grad_wt",
            "densenet_dense_block_1_bottleneck_2_conv1_grad_wt",
            "densenet_dense_block_1_bottleneck_2_conv2_grad_wt",
            "densenet_dense_block_1_bottleneck_3_conv1_grad_wt",
            "densenet_dense_block_1_bottleneck_3_conv2_grad_wt",
            "densenet_dense_block_1_bottleneck_4_conv1_grad_wt",
            "densenet_dense_block_1_bottleneck_4_conv2_grad_wt",
            "densenet_dense_block_1_bottleneck_5_conv1_grad_wt",
            "densenet_dense_block_1_bottleneck_5_conv2_grad_wt",
            "densenet_transition_1_conv_grad_wt",
            "densenet_dense_block_2_bottleneck_0_conv1_grad_wt",
            "densenet_dense_block_2_bottleneck_0_conv2_grad_wt",
            "densenet_dense_block_2_bottleneck_1_conv1_grad_wt",
            "densenet_dense_block_2_bottleneck_1_conv2_grad_wt",
            "densenet_dense_block_2_bottleneck_2_conv1_grad_wt",
            "densenet_dense_block_2_bottleneck_2_conv2_grad_wt",
            "densenet_dense_block_2_bottleneck_3_conv1_grad_wt",
            "densenet_dense_block_2_bottleneck_3_conv2_grad_wt",
            "densenet_dense_block_2_bottleneck_4_conv1_grad_wt",
            "densenet_dense_block_2_bottleneck_4_conv2_grad_wt",
            "densenet_dense_block_2_bottleneck_5_conv1_grad_wt",
            "densenet_dense_block_2_bottleneck_5_conv2_grad_wt",
            "densenet_dense_block_2_bottleneck_6_conv1_grad_wt",
            "densenet_dense_block_2_bottleneck_6_conv2_grad_wt",
            "densenet_dense_block_2_bottleneck_7_conv1_grad_wt",
            "densenet_dense_block_2_bottleneck_7_conv2_grad_wt",
            "densenet_dense_block_2_bottleneck_8_conv1_grad_wt",
            "densenet_dense_block_2_bottleneck_8_conv2_grad_wt",
            "densenet_dense_block_2_bottleneck_9_conv1_grad_wt",
            "densenet_dense_block_2_bottleneck_9_conv2_grad_wt",
            "densenet_dense_block_2_bottleneck_10_conv1_grad_wt",
            "densenet_dense_block_2_bottleneck_10_conv2_grad_wt",
            "densenet_dense_block_2_bottleneck_11_conv1_grad_wt",
            "densenet_dense_block_2_bottleneck_11_conv2_grad_wt",
            "densenet_transition_2_conv_grad_wt",
            "densenet_dense_block_3_bottleneck_0_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_0_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_1_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_1_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_2_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_2_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_3_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_3_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_4_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_4_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_5_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_5_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_6_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_6_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_7_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_7_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_8_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_8_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_9_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_9_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_10_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_10_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_11_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_11_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_12_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_12_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_13_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_13_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_14_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_14_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_15_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_15_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_16_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_16_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_17_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_17_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_18_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_18_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_19_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_19_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_20_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_20_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_21_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_21_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_22_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_22_conv2_grad_wt",
            "densenet_dense_block_3_bottleneck_23_conv1_grad_wt",
            "densenet_dense_block_3_bottleneck_23_conv2_grad_wt",
            "densenet_transition_3_conv_grad_wt",
            "densenet_dense_block_4_bottleneck_0_conv1_grad_wt",
            "densenet_dense_block_4_bottleneck_0_conv2_grad_wt",
            "densenet_dense_block_4_bottleneck_1_conv1_grad_wt",
            "densenet_dense_block_4_bottleneck_1_conv2_grad_wt",
            "densenet_dense_block_4_bottleneck_2_conv1_grad_wt",
            "densenet_dense_block_4_bottleneck_2_conv2_grad_wt",
            "densenet_dense_block_4_bottleneck_3_conv1_grad_wt",
            "densenet_dense_block_4_bottleneck_3_conv2_grad_wt",
            "densenet_dense_block_4_bottleneck_4_conv1_grad_wt",
            "densenet_dense_block_4_bottleneck_4_conv2_grad_wt",
            "densenet_dense_block_4_bottleneck_5_conv1_grad_wt",
            "densenet_dense_block_4_bottleneck_5_conv2_grad_wt",
            "densenet_dense_block_4_bottleneck_6_conv1_grad_wt",
            "densenet_dense_block_4_bottleneck_6_conv2_grad_wt",
            "densenet_dense_block_4_bottleneck_7_conv1_grad_wt",
            "densenet_dense_block_4_bottleneck_7_conv2_grad_wt",
            "densenet_dense_block_4_bottleneck_8_conv1_grad_wt",
            "densenet_dense_block_4_bottleneck_8_conv2_grad_wt",
            "densenet_dense_block_4_bottleneck_9_conv1_grad_wt",
            "densenet_dense_block_4_bottleneck_9_conv2_grad_wt",
            "densenet_dense_block_4_bottleneck_10_conv1_grad_wt",
            "densenet_dense_block_4_bottleneck_10_conv2_grad_wt",
            "densenet_dense_block_4_bottleneck_11_conv1_grad_wt",
            "densenet_dense_block_4_bottleneck_11_conv2_grad_wt",
            "densenet_dense_block_4_bottleneck_12_conv1_grad_wt",
            "densenet_dense_block_4_bottleneck_12_conv2_grad_wt",
            "densenet_dense_block_4_bottleneck_13_conv1_grad_wt",
            "densenet_dense_block_4_bottleneck_13_conv2_grad_wt",
            "densenet_dense_block_4_bottleneck_14_conv1_grad_wt",
            "densenet_dense_block_4_bottleneck_14_conv2_grad_wt",
            "densenet_dense_block_4_bottleneck_15_conv1_grad_wt",
            "densenet_dense_block_4_bottleneck_15_conv2_grad_wt"
    ]

    nfnet_fwrd_layers  = [
            "NF_ResNet_block_0_conv0",
            "NF_ResNet_block_0_conv1",
            "NF_ResNet_block_0_conv2",
            "NF_ResNet_block_0_conv_shortcut",
            "NF_ResNet_block_1_conv0",
            "NF_ResNet_block_1_conv1",
            "NF_ResNet_block_1_conv2",
            "NF_ResNet_block_2_conv0",
            "NF_ResNet_block_2_conv1",
            "NF_ResNet_block_2_conv2",
            "NF_ResNet_block_2_conv_shortcut",
            "NF_ResNet_block_3_conv0",
            "NF_ResNet_block_3_conv1",
            "NF_ResNet_block_3_conv2",
            "NF_ResNet_block_4_conv0",
            "NF_ResNet_block_4_conv1",
            "NF_ResNet_block_4_conv2",
            "NF_ResNet_block_4_conv_shortcut",
            "NF_ResNet_block_5_conv0",
            "NF_ResNet_block_5_conv1",
            "NF_ResNet_block_5_conv2",
            "NF_ResNet_block_6_conv0",
            "NF_ResNet_block_6_conv1",
            "NF_ResNet_block_6_conv2",
            "NF_ResNet_block_6_conv_shortcut",
            "NF_ResNet_block_7_conv0",
            "NF_ResNet_block_7_conv1",
            "NF_ResNet_block_7_conv2"
    ]

    nfnet_bkwd_layers = [
            "NF_ResNet_block_0_conv0_grad_in",
            "NF_ResNet_block_0_conv1_grad_in",
            "NF_ResNet_block_0_conv2_grad_in",
            "NF_ResNet_block_0_conv_shortcut_grad_in",
            "NF_ResNet_block_1_conv0_grad_in",
            "NF_ResNet_block_1_conv1_grad_in",
            "NF_ResNet_block_1_conv2_grad_in",
            "NF_ResNet_block_2_conv0_grad_in",
            "NF_ResNet_block_2_conv1_grad_in",
            "NF_ResNet_block_2_conv2_grad_in",
            "NF_ResNet_block_2_conv_shortcut_grad_in",
            "NF_ResNet_block_3_conv0_grad_in",
            "NF_ResNet_block_3_conv1_grad_in",
            "NF_ResNet_block_3_conv2_grad_in",
            "NF_ResNet_block_4_conv0_grad_in",
            "NF_ResNet_block_4_conv1_grad_in",
            "NF_ResNet_block_4_conv2_grad_in",
            "NF_ResNet_block_4_conv_shortcut_grad_in",
            "NF_ResNet_block_5_conv0_grad_in",
            "NF_ResNet_block_5_conv1_grad_in",
            "NF_ResNet_block_5_conv2_grad_in",
            "NF_ResNet_block_6_conv0_grad_in",
            "NF_ResNet_block_6_conv1_grad_in",
            "NF_ResNet_block_6_conv2_grad_in",
            "NF_ResNet_block_6_conv_shortcut_grad_in",
            "NF_ResNet_block_7_conv0_grad_in",
            "NF_ResNet_block_7_conv1_grad_in",
            "NF_ResNet_block_7_conv2_grad_in",

            "NF_ResNet_block_0_conv0_grad_wt",
            "NF_ResNet_block_0_conv1_grad_wt",
            "NF_ResNet_block_0_conv2_grad_wt",
            "NF_ResNet_block_0_conv_shortcut_grad_wt",
            "NF_ResNet_block_1_conv0_grad_wt",
            "NF_ResNet_block_1_conv1_grad_wt",
            "NF_ResNet_block_1_conv2_grad_wt",
            "NF_ResNet_block_2_conv0_grad_wt",
            "NF_ResNet_block_2_conv1_grad_wt",
            "NF_ResNet_block_2_conv2_grad_wt",
            "NF_ResNet_block_2_conv_shortcut_grad_wt",
            "NF_ResNet_block_3_conv0_grad_wt",
            "NF_ResNet_block_3_conv1_grad_wt",
            "NF_ResNet_block_3_conv2_grad_wt",
            "NF_ResNet_block_4_conv0_grad_wt",
            "NF_ResNet_block_4_conv1_grad_wt",
            "NF_ResNet_block_4_conv2_grad_wt",
            "NF_ResNet_block_4_conv_shortcut_grad_wt",
            "NF_ResNet_block_5_conv0_grad_wt",
            "NF_ResNet_block_5_conv1_grad_wt",
            "NF_ResNet_block_5_conv2_grad_wt",
            "NF_ResNet_block_6_conv0_grad_wt",
            "NF_ResNet_block_6_conv1_grad_wt",
            "NF_ResNet_block_6_conv2_grad_wt",
            "NF_ResNet_block_6_conv_shortcut_grad_wt",
            "NF_ResNet_block_7_conv0_grad_wt",
            "NF_ResNet_block_7_conv1_grad_wt",
            "NF_ResNet_block_7_conv2_grad_wt",
    ]


    table_dict = {'resnet18_fwrd_inject': resnet18_fwrd_layers,
                  'resnet18_bkwd_inject': resnet18_bkwd_layers,

                  'effnet_fwrd_inject': effnet_fwrd_layers,
                  'effnet_bkwd_inject': effnet_bkwd_layers,

                  'densenet_fwrd_inject': densenet_fwrd_layers,
                  'densenet_bkwd_inject': densenet_bkwd_layers,

                  'nfnet_fwrd_inject': nfnet_fwrd_layers,
                  'nfnet_bkwd_inject': nfnet_bkwd_layers
            }
    
    if '_' in model:
        model = model[:model.find('_')]
    target_list = table_dict[model + "_" + phase]

    #return target_list[np.random.randint(len(target_list))]

    if 'fwrd' in phase:
        return target_list[np.random.randint(len(target_list))]
    else:
        target_layer = None
        while not target_layer or 'grad_in' not in target_layer:
            target_layer = target_list[np.random.randint(len(target_list))]
        return target_layer



def get_inj_args(inj_type, strategy, inj_layer, inputs, kernels, outputs, train_recorder, db_st):
    np.random.seed(None)
    inj_replica = np.random.randint(strategy.num_replicas_in_sync)
    db_st.target_worker = inj_replica
    record(train_recorder, "Inject worker: {}\n".format(inj_replica))
    record(train_recorder, "Inject layer: {}\n".format(inj_layer))

    if is_input_target(inj_type):
        target = inputs.values[inj_replica].numpy()
    elif is_weight_target(inj_type):
        if type(kernels) == list:
            target = kernels[0].values[inj_replica].numpy()
        else:
            target = kernels.values[inj_replica].numpy()
    elif is_output_target(inj_type):
        target = outputs.values[inj_replica].numpy()
    else:
        print("ERROR: Unsupported inject type!")
        exit(2)

    record(train_recorder, "Shape for target layer is {}\n".format(target.shape))

    mask, delta = choose_inj_pos(target, inj_type, train_recorder, db_st)

    if type(kernels) == list:
        golden_weights = []
        for elem in kernels:
            wt_em = elem.values[0].numpy()
            golden_weights.append(wt_em)
    else:
        golden_weights = kernels.values[0].numpy()

    np_array = np.zeros(strategy.num_replicas_in_sync, dtype=bool)
    np_array[inj_replica] = True
    inj_flag_dataset = tf.data.Dataset.from_tensor_slices(np_array).repeat().batch(strategy.num_replicas_in_sync)
    inj_flag_iterator = iter(strategy.experimental_distribute_dataset(inj_flag_dataset))
    inj_flag = next(inj_flag_iterator)

    return InjArgs(inj_replica, inj_layer, inj_type, golden_weights, outputs, mask, delta), inj_flag



def get_gpu_inj_args(inj_type, inj_layer, inputs, kernels, outputs, train_recorder, db_st):
    np.random.seed(None)
    inj_replica = 0
    db_st.target_worker = inj_replica
    record(train_recorder, "Inject worker: {}\n".format(inj_replica))
    record(train_recorder, "Inject layer: {}\n".format(inj_layer))

    if is_input_target(inj_type):
        target = inputs.numpy()
    elif is_weight_target(inj_type):
        if type(kernels) == list:
            target = kernels[0].numpy()
        else:
            target = kernels.numpy()
    elif is_output_target(inj_type):
        target = outputs.numpy()
    else:
        print("ERROR: Unsupported inject type!")
        exit(2)

    record(train_recorder, "Shape for target layer is {}\n".format(target.shape))

    mask, delta = choose_inj_pos(target, inj_type, train_recorder, db_st)

    if type(kernels) == list:
        golden_weights = []
        for elem in kernels:
            wt_em = elem.numpy()
            golden_weights.append(wt_em)
    else:
        golden_weights = kernels.numpy()

    '''
    np_array = np.zeros(strategy.num_replicas_in_sync, dtype=bool)
    np_array[inj_replica] = True
    inj_flag_dataset = tf.data.Dataset.from_tensor_slices(np_array).repeat().batch(strategy.num_replicas_in_sync)
    inj_flag_iterator = iter(strategy.experimental_distribute_dataset(inj_flag_dataset))
    inj_flag = next(inj_flag_iterator)
    '''

    inj_flag = True

    return InjArgs(inj_replica, inj_layer, inj_type, golden_weights, outputs, mask, delta), inj_flag


def set_replay_pos(target, rp, train_recorder):
    shape = target.shape

    positions = [tuple(elem) for elem in rp.inj_pos]

    mask = np.ones(shape)
    delta = np.zeros(shape)

    for i in range(len(positions)):
        pos = positions[i]
        ori_val = target[pos]
        val_delta = rp.inj_values[i]

        mask[pos] = 0
        delta[pos] = val_delta

        record(train_recorder, "Position is {}, Golden data is {}, inject data is {}\n".format(pos, ori_val, val_delta))
    return mask, delta


def get_replay_args(inj_type, rp, strategy, inj_layer, inputs, kernels, outputs, train_recorder):
    np.random.seed(None)
    #inj_replica = np.random.randint(strategy.num_replicas_in_sync)
    inj_replica = rp.target_worker
    record(train_recorder, "Inject worker: {}\n".format(inj_replica))
    record(train_recorder, "Inject layer: {}\n".format(inj_layer))

    if is_input_target(inj_type):
        target = inputs.values[inj_replica].numpy()
    elif is_weight_target(inj_type):
        if type(kernels) == list:
            target = kernels[0].values[inj_replica].numpy()
        else:
            target = kernels.values[inj_replica].numpy()
    elif is_output_target(inj_type):
        target = outputs.values[inj_replica].numpy()
    else:
        print("ERROR: Unsupported inject type!")
        exit(2)

    record(train_recorder, "Shape for target layer is {}\n".format(target.shape))

    mask, delta = set_replay_pos(target, rp, train_recorder)

    if type(kernels) == list:
        golden_weights = []
        for elem in kernels:
            wt_em = elem.values[0].numpy()
            golden_weights.append(wt_em)
    else:
        golden_weights = kernels.values[0].numpy()

    np_array = np.zeros(strategy.num_replicas_in_sync, dtype=bool)
    np_array[inj_replica] = True
    inj_flag_dataset = tf.data.Dataset.from_tensor_slices(np_array).repeat().batch(strategy.num_replicas_in_sync)
    inj_flag_iterator = iter(strategy.experimental_distribute_dataset(inj_flag_dataset))
    inj_flag = next(inj_flag_iterator)

    return InjArgs(inj_replica, inj_layer, inj_type, golden_weights, outputs, mask, delta), inj_flag


def multi_tensor_reduce_max(t_list):
    maxes = []
    for t in t_list:
        maxes.append(tf.reduce_max(tf.abs(t)))
    return tf.reduce_max(maxes)
