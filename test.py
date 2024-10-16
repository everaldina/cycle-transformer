# This code is released under the CC BY-SA 4.0 license.

import glob
import os
import numpy as np
import pandas as pd
import pydicom
import torch
import json

from skimage.metrics import structural_similarity as ssim
from models import create_model
from options.train_options import TrainOptions
from options.test_options import TestOptions


def export_results(data, image_ids, scan_ids, result_dir, model_name):
    result_folder = os.path.join(result_dir, model_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    # Save results to csv 
    
    pd.DataFrame(
        {'image': image_ids, 
         'scan': scan_ids, 
         'mae_pre': data['mae']['pre'], 
         'mae_post': data['mae']['post'], 
         'rmse_pre': data['rmse']['pre'], 
         'rmse_post': data['rmse']['post'], 
         'ssim_pre': data['ssim']['pre'], 
         'ssim_post': data['ssim']['post']
        }
    ).to_csv(os.path.join(result_folder, 'results.csv'))
    
    pd.DataFrame(
        {
            'mae': {
                'pre': np.mean(data['mae']['pre']),
                'post': np.mean(data['mae']['post'])
            },
            'rmse': {
                'pre': np.mean(data['rmse']['pre']),
                'post': np.mean(data['rmse']['post'])
            },
            'ssim': {
                'pre': np.mean(data['ssim']['pre']),
                'post': np.mean(data['ssim']['post'])
            }
        }, index=[0]
    ).to_csv(os.path.join(result_folder, 'results_summary.csv'))
    
    # save results to json
    with open(os.path.join(result_folder, 'results.json'), 'w') as f:
        json.dump(data, f, indent=4)


def print_result(mae_pre, mae_post, rmse_pre, rmse_post, ssim_pre, ssim_post):
    print(f"MAE before {mae_pre}, after {mae_post}")
    print(f"RMSE before {rmse_pre}, after {rmse_post}")
    print(f"SSIM before {ssim_pre}, after {ssim_post}")
    


@torch.no_grad()
def compute_eval_metrics_gan(opt, result_dir="test_results", tagA='ARTERIAL', tagB='NATIVE', device='cpu'):
    # root_path - is the path to the raw Coltea-Lung-CT-100W data set.

    #opt = TrainOptions().parse()
    #opt.load_iter = 40
    opt.isTrain = False
    opt.device = device

    model = create_model(opt)
    model.setup(opt)
    gen = model.netG_A
    gen.eval()

    eval_dirs = pd.read_csv(os.path.join(opt.dataroot, 'test_data.csv'))
    eval_dirs = list(eval_dirs.iloc[:, 1])
    
    id_list = []
    scan_list = []

    mae_pre = []
    mae_post = []
    rmse_pre = []
    rmse_post = []
    ssim_pre = []
    ssim_post = []

    for path in glob.glob(os.path.join(opt.dataroot, 'Coltea-Lung-CT-100W/*')):
        image_id = path.split('/')[-1]
        if not image_id in eval_dirs:
            continue

        print(f"{'-'*10} Processing {image_id} {'-'*10}")
        for scan in glob.glob(os.path.join(path, tagA, 'DICOM', '*')):
            scan_name = scan.split('/')[-1]
            
            id_list.append(image_id)
            scan_list.append(scan_name)
            orig_img = pydicom.dcmread(scan).pixel_array
            native_img = pydicom.dcmread(scan.replace(tagA, tagB)).pixel_array

            # Scale native image
            native_img[native_img < 0] = 0
            native_img = native_img / 1e3
            native_img = native_img - 1

            # Scale original image, which is transform
            orig_img[orig_img < 0] = 0
            orig_img = orig_img / 1e3
            orig_img = orig_img - 1

            orig_img_in = np.expand_dims(orig_img, 0).astype(float)
            orig_img_in = torch.from_numpy(orig_img_in).float().to(device)
            orig_img_in = orig_img_in.unsqueeze(0)

            native_fake = gen(orig_img_in)[0, 0].detach().cpu().numpy()

            mae_pre.append(np.mean(np.abs(orig_img - native_img)))
            mae_post.append(np.mean(np.abs(native_fake - native_img)))

            rmse_pre.append(np.sqrt(np.mean((orig_img - native_img)**2)))
            rmse_post.append(np.sqrt(np.mean((native_fake - native_img)**2)))


            # data_range : float, optional
            # The data range of the input image (difference between maximum and minimum possible values). 
            # By default, this is estimated from the image data type. This estimate may be wrong for floating-point 
            # image data. Therefore it is recommended to always pass this scalar value explicitly (see note below).
            max_orig = orig_img.max()
            min_orig = orig_img.min()
            max_native = native_img.max()
            min_native = native_img.min()
            max_native_fake = native_fake.max()
            min_native_fake = native_fake.min()
            
            ssim_pre.append(ssim(orig_img, native_img, data_range= max(max_orig, max_native) - min(min_orig, min_native)))
            ssim_post.append(ssim(native_fake, native_img, data_range= max(max_native_fake, max_native) - min(min_native_fake, min_native)))
            #print_result(mae_pre[-1], mae_post[-1], rmse_pre[-1], rmse_post[-1], ssim_pre[-1], ssim_post[-1])

    

    data = {
        'mae': {
            'pre': mae_pre,
            'post': mae_post
        },
        'rmse': {
            'pre': rmse_pre,
            'post': rmse_post
        },
        'ssim': {
            'pre': ssim_pre,
            'post': ssim_post
        }
    }
    export_results(data, id_list, scan_list, result_dir, opt.name)
    

    mae_pre = np.mean(mae_pre)
    mae_post = np.mean(mae_post)
    rmse_pre = np.mean(rmse_pre)
    rmse_post = np.mean(rmse_post)
    ssim_pre = np.mean(ssim_pre)
    ssim_post = np.mean(ssim_post)

    print(f"MAE before {mae_pre}, after {mae_post}")
    print(f"RMSE before {rmse_pre}, after {rmse_post}")
    print(f"SSIM before {ssim_pre}, after {ssim_post}")


if __name__ == '__main__':
    opt = TrainOptions().parse()
    result_dir = "test_results"
    
    compute_eval_metrics_gan(
        opt=opt,
        device='cuda',
    )
