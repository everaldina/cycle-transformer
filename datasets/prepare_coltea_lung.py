import argparse
import pandas as pd
import os
import json
import pickle

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataroot', type=str)
    parser.add_argument('train_list', type=str)
    parser.add_argument('test_list', type=str)
    parser.add_argument('eval_list', type=str)
    parser.add_argument('--results_dir', type=str, default='data_list')
    return parser.parse_args()

def get_ids(list_path):
    data = pd.read_csv(list_path)
    return list(data.iloc[:,1])

def load_dicom(path, phase):
    root = os.path.join(path, phase, 'DICOM')
    files = os.listdir(root)
    list_files = [os.path.join(root, x) for x in files]
    return sorted(list_files)
    
    
    

if __name__ == '__main__':
    args = load_args()
    
    phases = ['ARTERIAL', 'VENOUS', 'NATIVE']
    
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    subset = {
        'train': {
            'ids': get_ids(args.train_list),
            'data': {
                'ARTERIAL': [],
                'VENOUS': [],
                'NATIVE': []
            }
        },
        'test': {
            'ids': get_ids(args.test_list),
            'data': {
                'ARTERIAL': [],
                'VENOUS': [],
                'NATIVE': []
            }
        },
        'eval': {
            'ids': get_ids(args.eval_list),
            'data': {
                'ARTERIAL': [],
                'VENOUS': [],
                'NATIVE': []
            }
        }
    }
    
    dir_path_list = [os.path.join(args.dataroot, x) for x in os.listdir(args.dataroot) if os.path.isdir(os.path.join(args.dataroot, x))]
    
    # adding the paths to the subset dictionary
    for item in dir_path_list:
        dir_name = os.path.basename(item)
        
        for key in subset:
            if dir_name in subset[key]['ids']:
                subset[key]['data']['ARTERIAL'].extend(load_dicom(item, 'ARTERIAL'))
                subset[key]['data']['VENOUS'].extend(load_dicom(item, 'VENOUS'))
                subset[key]['data']['NATIVE'].extend(load_dicom(item, 'NATIVE'))
       
    # saving the data list
    for key in subset:
        with open(os.path.join(args.results_dir, key + '_list.json'), 'w') as f:
            json.dump(subset[key]['data'], f)
            
        with open(os.path.join(args.results_dir, key + '_list.pkl'), "wb") as f:
            pickle.dump(subset[key]['data'], f)
    
        
    
    
    
    
    
    
    
    


