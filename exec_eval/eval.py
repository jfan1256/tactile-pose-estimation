import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import yaml
import pickle
import datetime
import numpy as np

from tqdm import tqdm
from pathlib import Path

from utils.system import get_configs
from utils.display import print_header
from exec_gen.gen_vid import generate_vid
from exec_train.utils_model import set_seed
from exec_gen.gen_image import generate_image
from exec_eval.utils_eval import setup_data, setup_model, model_generate, get_keypoint_spatial_dis

# Main
def main(configs):
    # Setup model
    model = setup_model(configs)

    # Setup data
    test_dataloader = setup_data(configs)

    # Start training
    print_header("Start Evaluating")
    start_time = time.time()

    # Setup image data
    tactile_GT = np.empty((1, 96, 96))
    heatmap_GT = np.empty((1, 21, 20, 20, 18))
    heatmap_pred = np.empty((1, 21, 20, 20, 18))
    keypoint_GT = np.empty((1, 21, 3))
    keypoint_pred = np.empty((1, 21, 3))
    
    # Setup video data
    tactile_GT_v = np.empty((1, 96, 96))
    heatmap_GT_v = np.empty((1, 21, 20, 20, 18))
    heatmap_pred_v = np.empty((1, 21, 20, 20, 18))
    keypoint_GT_v = np.empty((1, 21, 3))
    keypoint_pred_v = np.empty((1, 21, 3))
    keypoint_GT_log = np.empty((1, 21, 3))
    keypoint_pred_log = np.empty((1, 21, 3))

    count = 0
    for i, batch in tqdm(enumerate(test_dataloader), desc='Test'):
        # Pass through model
        heatmap_out, keypoint_out, tactile, heatmap, keypoint = model_generate(model, batch, configs)

        # Export image
        if configs.exp_image:
            base = 0
            image_data = [heatmap.cpu().data.numpy().reshape(-1, 21, 20, 20, 18),
                          heatmap_out.cpu().data.numpy().reshape(-1, 21, 20, 20, 18),
                          keypoint.cpu().data.numpy().reshape(-1, 21, 3),
                          keypoint_out.cpu().data.numpy().reshape(-1, 21, 3),
                          tactile.cpu().data.numpy().reshape(-1, 96, 96)]

            # Generate image
            generate_image(image_data, configs.exp_dir + 'predictions/image/', i, base)

        # Export video
        if configs.exp_video:
            if i > 50 and i < 60:
                heatmap_GT_v = np.append(heatmap_GT, heatmap.cpu().data.numpy().reshape(-1, 21, 20, 20, 18), axis=0)
                heatmap_pred_v = np.append(heatmap_pred, heatmap_out.cpu().data.numpy().reshape(-1, 21, 20, 20, 18), axis=0)
                keypoint_GT_v = np.append(keypoint_GT, keypoint.cpu().data.numpy().reshape(-1, 21, 3), axis=0)
                keypoint_pred_v = np.append(keypoint_pred, keypoint_out.cpu().data.numpy().reshape(-1, 21, 3), axis=0)
                tactile_GT_v = np.append(tactile_GT, tactile.cpu().data.numpy().reshape(-1, 96, 96), axis=0)

        # Export L2
        if configs.exp_L2:
            keypoint_GT_log = np.append(keypoint_GT_log, keypoint.cpu().data.numpy().reshape(-1, 21, 3), axis=0)
            keypoint_pred_log = np.append(keypoint_pred_log, keypoint_out.cpu().data.numpy().reshape(-1, 21, 3), axis=0)

        # Export data
        if configs.exp_data:
            heatmap_GT = np.append(heatmap_GT, heatmap.cpu().data.numpy().reshape(-1, 21, 20, 20, 18), axis=0)
            heatmap_pred = np.append(heatmap_pred, heatmap_out.cpu().data.numpy().reshape(-1, 21, 20, 20, 18), axis=0)
            keypoint_GT = np.append(keypoint_GT, keypoint.cpu().data.numpy().reshape(-1, 21, 3), axis=0)
            keypoint_pred = np.append(keypoint_pred, keypoint_out.cpu().data.numpy().reshape(-1, 21, 3), axis=0)
            tactile_GT = np.append(tactile_GT, tactile.cpu().data.numpy().reshape(-1, 96, 96), axis=0)

            # Avoid overflow
            if i % 20 == 0 and i != 0:
                count += 1
                to_save = [heatmap_GT[1:, :, :, :, :], heatmap_pred[1:, :, :, :, :],
                          keypoint_GT[1:, :, :], keypoint_pred[1:, :, :],
                          tactile_GT[1:, :, :]]
                pickle.dump(to_save, open(configs.exp_dir + 'predictions/data/' + configs.ckpt + str(count) + '.p', "wb"))
                tactile_GT = np.empty((1, 96, 96))
                heatmap_GT = np.empty((1, 21, 20, 20, 18))
                heatmap_pred = np.empty((1, 21, 20, 20, 18))
                keypoint_GT = np.empty((1, 21, 3))
                keypoint_pred = np.empty((1, 21, 3))

    # Export L2
    if configs.exp_L2:
        dis = get_keypoint_spatial_dis(keypoint_GT_log[1:, :, :], keypoint_pred_log[1:, :, :])
        pickle.dump(dis, open(configs.exp_dir + 'predictions/L2/'+ configs.ckpt + '_dis.p', "wb"))
        print ("keypoint_dis_saved:", dis.shape)

    # Export video
    if configs.exp_video:
        to_save = [heatmap_GT_v[1:, :, :, :, :], heatmap_pred_v[1:, :, :, :, :],
                   keypoint_GT_v[1:, :, :], keypoint_pred_v[1:, :, :],
                   tactile_GT_v[1:, :, :]]

        print (to_save[0].shape, to_save[1].shape, to_save[2].shape, to_save[3].shape, to_save[4].shape)
        generate_vid(to_save, configs.exp_dir + 'predictions/video/' + configs.ckpt, heatmap=True)

    # Calculate total time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluating time {}'.format(total_time_str))
    return

if __name__ == '__main__':
    # Set seed
    set_seed(20050531)

    # Get configs
    configs = yaml.load(open(get_configs() / 'eval' / 'eval.yaml', 'r'), Loader=yaml.Loader)
    Path(configs['output_dir']).mkdir(parents=True, exist_ok=True); yaml.dump(configs, open(os.path.join(configs['output_dir'], 'configs.yaml'), 'w'))

    # Execute main
    main(configs)