import json
import numpy as np

save_dir = "./Seeds/"

def generate_states():
    params_camera = {
        "LCX": 324,
        "LCY": 243,
        "LFX": 2.00e+004,
        "LFY": 2.00e+004,
        "LK1": 2.341435984559e+000,
        "LK2": 2.537860298854e+003,
        "LK3": 1.058687686738e+001,
        "LTX": 0.000000000000e+000,
        "LTY": 0.000000000000e+000,
        "LTZ": 0.000000000000e+000,
        "LRot": [[1.000000000000e+000, 0.000000000000e+000, 0.000000000000e+000],
                 [0.000000000000e+000, 1.000000000000e+000, 0.000000000000e+000],
                 [0.000000000000e+000, 0.000000000000e+000, 1.000000000000e+000]],
        "LResolution": (486, 648),
        "LEulerAngle": (0, 0, 0),

        "RCX": 324,
        "RCY": 243,
        "RFX": 2.00e+004,
        "RFY": 2.00e+004,
        "RK1": 7.085011634771e+000,
        "RK2": -1.840555488318e+003,
        "RK3": 2.309452655375e+000,
        "RTX": -3.00e+002,
        "RTY": -0.0,
        "RTZ": 172.2051,
        "RRot": [[0.5, 0, 0.8660254],
                 [0, 1, 0],
                 [-0.8660254, 0, 0.5]],
        "RResolution": (486, 648),
        "REulerAngle": (-0.00882608, 0.438867, -0.0169612),     #in Rad
    }

    params_dataset = {
        "num_trainset": 1000,
        "num_testset": 100,
        "disp_num_xy": 5,
        "disp_num_z": 5,
        "assumed_range": 10,          # in mm, on Os surface
        "disp_amp": 0.05,           # in mm
        "dataset_savepath": "E:/DATA/3DDIC_DL/",
    }

    params_camera.update(params_dataset)
    json_str = json.dumps(params_camera, indent=4)
    with open(save_dir+"States.json", 'w') as json_file:
        json_file.write(json_str)
    return params_dataset

def generate_seeds(params_dataset, out_name):
    seed_array = np.zeros(shape=(params_dataset["num_trainset"], 93))
    amp = params_dataset["disp_amp"]
    assumed_range = params_dataset["assumed_range"]

    densityarray = [list(np.linspace(1, 2.5, num=5)), list(np.linspace(0.7, 1.6, num=5)), list(np.linspace(1.8, 3.5, num=5)), list(np.linspace(1.6, 3.2, num=5)), list(np.linspace(1.2, 2.5, num=5)), ]
    speckle_size = [1.5, 1.99, 2.5, 3, 3.5]

    for i in range(params_dataset["num_trainset"]):
        amp_list = amp * np.random.randint(4, 10, size=5) / 20
        # Five inplane/outplane displacements
        for j in range(5):
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
                seed_array[i, j * 11 + 1:j * 11 + 11] = amp_list[j] * A, amp_list[j] * B, amp_list[j] * C, amp_list[j] * D, amp_list[j] * E, amp_list[j] * F, 1e11, 1e11, 1e11, 1e11# 平面位移情况
            # sin product
            else:
                A = np.random.normal(1, 0.2)
                B = 2 * np.pi / (assumed_range * 5 / np.random.randint(1, 30))
                C = np.random.rand() * 2 * np.pi
                D = 2 * np.pi / (assumed_range * 5 / np.random.randint(1, 20))
                E = np.random.rand() * 2 * np.pi
                F = np.random.normal(1, 0.2)
                G = 2 * np.pi / (assumed_range * 5 / np.random.randint(1, 30))
                H = np.random.rand() * 2 * np.pi
                I = 2 * np.pi / (assumed_range * 5 / np.random.randint(1, 20))
                J = np.random.rand() * 2 * np.pi
                seed_array[i, j * 11 + 1:j * 11 + 11] = amp_list[j] * A, B, C, D, E, amp_list[j] * F, G, H, I, J# 正弦位移情况

            # Type of off-plane displacement
            seed_array[i, 55 + j * 6] = np.random.rand() > 0.5
            # Linear
            if not seed_array[i, 55 + j * 6]:
                A = np.random.rand() / assumed_range
                B = np.random.rand() / assumed_range
                C = 1 - np.abs(A) * assumed_range - np.abs(B) * assumed_range
                seed_array[i, 55 + j * 6 + 1:55 + j * 6 + 6] = amp_list[j] * A, amp_list[j] * B, amp_list[j] * C, 1e11, 1e11# 平面位移情况
            # sin product
            else:
                A = np.random.normal(1, 0.2)
                B = 2 * np.pi / (assumed_range * 5 / np.random.randint(1, 30))
                C = np.random.rand() * 2 * np.pi
                D = 2 * np.pi / (assumed_range * 5 / np.random.randint(1, 20))
                E = np.random.rand() * 2 * np.pi
                seed_array[i, 55 + j * 6 + 1:55 + j * 6 + 6] = amp_list[j] * A, B, C, D, E# 正弦位移情况

        k = i//3    # One speckle pattern for 3 displacement fields

        # Speckle size
        seed_array[i, 85 + 1] = speckle_size[(k%25)//5]    # 散斑大小
        # Speckle density
        seed_array[i, 85 + 2] = densityarray[(k%25)//5][k%5] / 8        # 散斑密度
        # Random Seeds for image generator
        seed_array[i, 85 + 3:] = 5 * k, 5 * k + 1, 5 * k + 2, 5 * k + 3, 5 * k + 4

    np.savetxt(save_dir+out_name, seed_array, delimiter=",")

if __name__ == '__main__':
    params_dataset = generate_states()
    generate_seeds(params_dataset, "Seeds_*.csv")