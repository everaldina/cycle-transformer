import glob, os
import pydicom
import numpy as np
import torch

from options.test_options import TestOptions
from models import create_model
from util.util import mkdir

def preprocess_image(image_array):
    # image_array[image_array < 0] = 0
    # image_array = image_array / 1e3
    # image_array = image_array - 1
    return image_array

def save_image(x, y, image_id, scan_id, path):
    
    pass

def main(config):
    opt = config['opt']
    device = config['device']
    opt.isTrain = False
    opt.device = device

    model = create_model(opt)
    model.setup(opt)
    gen = model.netG_A
    gen.eval()
    
    tagA = config['moving']
    tagB = config['fixed']
    
    mkdir(config['real_pair_folder'])
    mkdir(config['cytran_pair_folder'])
    
    image_list = []
    for path in glob.glob(os.path.join(opt.dataroot, 'Coltea-Lung-CT-100W/*')):
        image_id = path.split('/')[-1]
        image_list.append(image_id)

        print(f"{'-'*10} Processing {image_id} {'-'*10}")
        for scan in glob.glob(os.path.join(path, tagA, 'DICOM', '*')):
            scan_name = scan.split('/')[-1]
            src_img = pydicom.dcmread(scan).pixel_array
            target_img = pydicom.dcmread(scan.replace(tagA, tagB)).pixel_array
            
            src_img = preprocess_image(src_img)
            target_img = preprocess_image(target_img)

            src_img_in = np.expand_dims(src_img, 0).astype(float)
            src_img_in = torch.from_numpy(src_img_in).float().to(device)
            src_img_in = src_img_in.unsqueeze(0)

            fake_target = gen(src_img_in)[0, 0].detach().cpu().numpy()
            
            save_image(src_img, fake_target, image_id, scan_name, config['cytran_pair_folder'])
            save_image(src_img, target_img, image_id, scan_name, config['real_pair_folder'])
            
            


if __name__ == '__main__':
    opt = TestOptions().parse()
    config = {
        'fixed': "NATIVE",
        'moving': "ARTERIAL",
        'real_pair_folder': "/results",
        'cytran_pair_folder': "/results",
        'net_path': "/net.pth", 
        'opt': opt,
        'device': 'cuda'  
    }
    main(config)