# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import numpy as np
from sys import platform
import argparse

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../bin/python/openpose/Release');
        os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../x64/Release;' + dir_path + '/../bin;'
        import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../examples/media/COCO_val2014_000000000192.jpg",
                        help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()


    # Process Image
    def posemodl(image):
        datum = op.Datum()
        # imageToProcess = cv2.imread(args[0].image_path)
        datum.cvInputData = image
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # Display Image
        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
        # cv2.waitKey(0)
        kp = datum.poseKeypoints
        return kp, datum.cvOutputData


    def gcnmodel(kps, net):
        count_None = 0
        new_kps = []
        for kp in kps:
            count_None += 1 if kp is None else 0
            new_kps.append(kp[0]) if kp is not None else np.zeros((18, 3))
        if count_None >= 5:
            return False
        kps = np.array(kps)  # T, 18, 3
        kps = torch.tensor(kps)
        # W, H = (320, 180) if scene_name is not 'Home_01'or'Home_02' else (320, 240)
        kps = noralization(kps, 656, 368)  # 3, 15, 18,  | 20, 18, 3
        kps = kps.permute(2, 0, 1)
        out = net(kps.unsqueeze(0).cuda())

        return out > 0.5


    def camer_input():
        kps = []
        import torch
        import net.st_gcn as GCN
        gcn_net = GCN.Model(in_channels=3,
                            num_class=1,
                            graph_args={'layout': 'openpose', 'strategy': 'spatial'},
                            edge_importance_weighting=True)
        model_path = ''
        gcn_net.load_state_dict(torch.load(model_path))
        gcn_net.cuda().eval()

        while 1:
            # get a frame
            ret, frame = cap.read()
            kp, outframe = posemodel(frame)
            kps.append(kp)
            if len(kps) == 20:
                fallsate = gcnmodel(kps, gcn_net)
            elif len(kps) > 20:
                kps.pop(0)
                fallsate = gcnmodel(kps, gcn_net)
            else:
                fallsate = False
            print('Fall state: ', fallsate)
            cv2.imshow("capture", outframe)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return


    camer_input()

except Exception as e:
    print(e)
    sys.exit(-1)
