import json
import numpy as np

save_dir = ""

def generate_states():
    params_camera = {
    "LCX": 1.955241548733e+003,
    "LCY": 1.556154461988e+003,
    "LFX": 4.809718190137e+004,
    "LFY": 4.765150703202e+004,
    "LTX": 0.0,
    "LTY": 0.0,
    "LTZ": 0.0,
    "LRot": [
        [
            1.0,
            0.0,
            0.0
        ],
        [
            0.0,
            1.0,
            0.0
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ],
    "LResolution": [
        3000, 4096
    ],

    "RCX": 1.979338466035e+003,
    "RCY": 1.358101978259e+003,
    "RFX": 3.955349186047e+004,
    "RFY": 3.881524048294e+004,
    "RTX": -7.395848297013e+002,
    "RTY": -1.462066037464e+001,
    "RTZ": -2.388431027659e+002,
    "RRot": [
        [
            9.315488131923e-001, -7.050266506112e-003, 3.635479368422e-001
        ],
        [
            3.193855908553e-003, 9.999320904861e-001, 1.120775180355e-002
        ],
        [
            -3.636022661157e-001, -9.279448165028e-003, 9.315080697006e-001
        ]
    ],
    "RResolution": [
        3000, 4096
    ],
    }

    params_dataset = {
        "Zcl": 2030,                # in mm, distance between Ow and Os in Zw direction
        "num_trainset": 100,
        "assumed_range": 60.0,      # Assumed spatial range in x or y direction, in mm, on Os surface. Influencing the speckle size.
        "disp_amp_uv": 0.01,        # Displacement amplitude of in-plane displacement you wanted, in mm.
        "disp_amp_w": 0.05,         # Displacement amplitude of off-plane displacement, in mm.
        "dataset_savepath": "./Sample/",  # Dataset savepath.
        "box_size":(1024, 1024),    # Image block size you wanted.
        "start_pos":(1000, 1500),    # The start point of the cropped area in the image pixel coordinate.
        "bkcolor":"white"           # Image background color. "black" for white speckle on black background; "white" for black speckle on white background.
    }

    params_camera.update(params_dataset)
    json_str = json.dumps(params_camera, indent=4)
    with open(save_dir+"States_box.json", 'w') as json_file:
        json_file.write(json_str)
    return params_dataset


def generate_seeds(params_dataset, out_name):
    seed_array = np.zeros(shape=(params_dataset["num_trainset"], 17 * 15 + 8))
    amp_w = params_dataset["disp_amp_w"]
    amp_uv = params_dataset["disp_amp_uv"]
    assumed_range = params_dataset["assumed_range"]

    # Change the speckle density here if you want
    densityarray = [list(np.linspace(1.3, 1.7, num=4)), list(np.linspace(1.0, 1.4, num=4))]

    # speckle_size = [1.5, 1.99, 2.5, 3, 3.5]
    speckle_size = [1.5, 1.8]   # , 2.4

    for i in range(params_dataset["num_trainset"]):
        amp_list_uv = amp_uv * np.random.randint(5, 15, size=15) / 30
        amp_list_w = amp_w * np.random.randint(5, 15, size=15) / 30
        # Five inplane/outplane displacements; For 3 Displacement
        for j in range(15):         # 3 Curved surface * 5 Displacement-of-summation for each dataset.
            # Type of in-plane displacement
            seed_array[i, j * 11] = np.random.rand() > 0.5
            # Linear
            if not seed_array[i, j * 11]:
                A = np.random.rand() / assumed_range
                B = np.random.rand() / assumed_range
                C = 1 - np.abs(A) * assumed_range - np.abs(B) * assumed_range
                D = np.random.rand() / assumed_range
                E = np.random.rand() / assumed_range
                F = 1 - np.abs(A) * assumed_range - np.abs(B) * assumed_range
                seed_array[i, j * 11 + 1:j * 11 + 11] = amp_list_uv[j] * A, amp_list_uv[j] * B, amp_list_uv[j] * C, amp_list_uv[j] * D, amp_list_uv[j] * E, amp_list_uv[j] * F, 1e11, 1e11, 1e11, 1e11# 平面位移情况
            # sin product
            else:
                A = np.random.normal(1, 0.2)
                B = 2 * np.pi / (assumed_range * 8 / np.random.randint(10, 30))
                C = np.random.rand() * 2 * np.pi
                D = 2 * np.pi / (assumed_range * 7 / np.random.randint(10, 20))
                E = np.random.rand() * 2 * np.pi
                F = np.random.normal(1, 0.2)
                G = 2 * np.pi / (assumed_range * 8 / np.random.randint(10, 30))
                H = np.random.rand() * 2 * np.pi
                I = 2 * np.pi / (assumed_range * 7 / np.random.randint(10, 20))
                J = np.random.rand() * 2 * np.pi
                seed_array[i, j * 11 + 1:j * 11 + 11] = amp_list_uv[j] * A, B, C, D, E, amp_list_uv[j] * F, G, H, I, J# 正弦位移情况

            # Type of off-plane displacement
            seed_array[i, 11*15 + j * 6] = np.random.rand() > 0.5
            # Linear
            if not seed_array[i, 11*15 + j * 6]:
                A = np.random.rand() / assumed_range
                B = np.random.rand() / assumed_range
                C = 1 - np.abs(A) * assumed_range - np.abs(B) * assumed_range
                seed_array[i, 11*15 + j * 6 + 1:11*15 + j * 6 + 6] = amp_list_w[j] * A, amp_list_w[j] * B, amp_list_w[j] * C, 1e11, 1e11# 平面位移情况
            # sin product
            else:
                A = np.random.normal(1, 0.2)
                B = 2 * np.pi / (assumed_range * 8 / np.random.randint(10, 30))
                C = np.random.rand() * 2 * np.pi
                D = 2 * np.pi / (assumed_range * 7 / np.random.randint(10, 20))
                E = np.random.rand() * 2 * np.pi
                seed_array[i, 11*15 + j * 6 + 1:11*15 + j * 6 + 6] = amp_list_w[j] * A, B, C, D, E# 正弦位移情况


        k = i    # One speckle pattern for 1 displacement fields

        # Speckle size
        seed_array[i, 17 * 15 + 1] = speckle_size[(k%8)//4]    # 散斑大小
        # Speckle density
        seed_array[i, 17 * 15 + 2] = densityarray[(k%8)//4][k%4] / 8        # 散斑密度
        # Random Seeds for image generator
        seed_array[i, 17 * 15 + 3:] = 5 * k, 5 * k + 1, 5 * k + 2, 5 * k + 3, 5 * k + 4

    np.savetxt(save_dir+out_name, seed_array, delimiter=",")

if __name__ == '__main__':
    params_dataset = generate_states()
    generate_seeds(params_dataset, "Seeds_exp_box.csv")