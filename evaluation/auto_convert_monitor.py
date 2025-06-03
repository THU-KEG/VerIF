import subprocess
import time, os, json
import psutil
import wandb
import argparse
from datetime import datetime, timedelta
import re

current_path = os.getcwd()

SOURCE_DIR = '/workspace/zhenyu/common_libs/'

config = {
    "9b": {
        'tgi_template': SOURCE_DIR +'/template/9b_8k/tgi',
        'config_path': SOURCE_DIR + '/template/9b_8k/nemo',
        'server_warmup_time': 120,
        'chkp_save_time': 240,
    },
    "qw7b": {
        'tgi_template': SOURCE_DIR + '/template/qw25_7b/tgi_ori_rope',
        'config_path': SOURCE_DIR + '/template/qw25_7b/nemo',
        'server_warmup_time': 120,
        'chkp_save_time': 240,
    },
    "qw14b": {
        'tgi_template': SOURCE_DIR + '/template/qw25_14b/tgi_ori_rope',
        'config_path': SOURCE_DIR + '/template/qw25_14b/nemo',
        'server_warmup_time': 200,
        'chkp_save_time': 400,
    },
    "qw32b": {
        'tgi_template': SOURCE_DIR + '/template/qw25_32b/tgi_ori_rope',
        'config_path': SOURCE_DIR + '/template/qw25_32b/nemo',
        'server_warmup_time': 200,
        'chkp_save_time': 400,
    }
}

# convert_script = 'python3 server/text_generation_server/models/glm2/convert_mcore.py --model_dir {} --save_dir {}'
# convert_script = 'python3 server/text_generation_server/models/glm2/hf-tgi-convert.py --model_dir {} --save_dir {}'

# tgi_src_path = '/usr/src'

convert_script = "python3 convert/convert_chatglm_nemo_to_tgi.py --input_name_or_path {} --output_path {} --cpu-only"


def need_convert_tgi(iter_dir):
    
    path = os.path.join(iter_dir, 'TGI')
    if not os.path.exists(path):
        return True
    dirs = os.listdir(path)
    find_safetenors = False
    for d in dirs:
        if '.safetensors' in d:
            find_safetenors = True
            break
    
    complete = find_safetenors and 'config.json' in dirs and 'config.py' in dirs
    if not complete:
        return True
    return False


def is_folder_recent(folder_path, seconds=600):
    folder_stat = os.stat(folder_path)
    folder_create_time = datetime.fromtimestamp(folder_stat.st_ctime)
    
    now = datetime.now()
    delta = now - folder_create_time
    
    return delta < timedelta(seconds=seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--tgi_port', type=str, default="8080")
    parser.add_argument('--tgi_master_port', type=str, default="29500")
    parser.add_argument('--cuda_devices', type=str, default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()

    seen_iter_dirs = set()

    def get_step(sst):
        # 'epoch=0-step=100-train_rewards=0.334'
        step = re.search(r'step=(\d+)', sst).group(1)
        return int(step)

    while True:
        with open(args.config_path, 'r') as f:
            monitor_config = json.load(f)
            for key in monitor_config.keys():
                monitor_config[key]['config'] = config[monitor_config[key]['config']]
        
        for mgt_path in monitor_config.keys():
            config_this = monitor_config[mgt_path]['config']
            project = monitor_config[mgt_path]['project']

            if not mgt_path.endswith('checkpoints') and not mgt_path.endswith('/checkpoints'):
                mgt_path = os.path.join(mgt_path, 'checkpoints')

            if not os.path.exists(mgt_path):
                print(f"Path {mgt_path} not exists. Waiting for the first ckpt")
                break
            
            dirs = os.listdir(mgt_path)

            iter_dirs = []
            for d in dirs:
                # if d.startswith('iter_'):
                    # iter_dirs.append(d)
                if d.startswith("epoch=") or d.startswith("'epoch"):
                    iter_dirs.append(d)

            print(mgt_path)
            print(dirs)
            ### for openrlhf only
            # dirs = sorted(iter_dirs, key=lambda x: int(x.split('_')[-1][4:]))
            dirs = sorted(iter_dirs, key=lambda x: get_step(x))
            
            for d in dirs:
                iter_dir = os.path.join(mgt_path, d)

                if "unfinished" in iter_dir:
                    print(f"Skip {iter_dir} for unfinished")
                    break
                
                # If recent saved, skip and wait for totally saved
                mp_rank_dirs = [x for x in os.listdir(iter_dir) if "model." in x]
                mp_rank_dirs = sorted(mp_rank_dirs)
                dir_to_check_time = None

                for mp_rank_d in mp_rank_dirs:
                    if mp_rank_d.endswith('safetensors'):
                        dir_to_check_time = os.path.join(iter_dir, mp_rank_d)

                    # dir_to_check_time = 
                if dir_to_check_time is None:
                    dir_to_check_time = iter_dir


                
                if is_folder_recent(dir_to_check_time, seconds=config_this['chkp_save_time']):
                    print(f"Skip {iter_dir} for recent saved")
                    # force test is conducted step by step for wandb
                    break
                
                # Convert TGI
                if need_convert_tgi(iter_dir):
                    os.system('rm -r {}'.format(os.path.join(iter_dir, 'TGI')))
                    # os.system('rm -r {}/*.json'.format(iter_dir))
                    os.system('rm -r {}/*.py'.format(iter_dir))
                    os.system('rm -r {}/*.model'.format(iter_dir))
                    os.system("mkdir -p {}".format(os.path.join(iter_dir, 'model_weights')))
                    os.system("cp -r {}/model.* {}".format(iter_dir, os.path.join(iter_dir, 'model_weights')))

                    config_path = config_this['tgi_template']
                    if config_path.endswith('/'):
                        config_path = config_path[:-1]
                    os.system('cp {}/* {}'.format(config_path, iter_dir))
                    os.system('cp {}/* {}'.format(config_this['config_path'], iter_dir))
                    
                    
                    os.system('cp {}/common.pt {}'.format(iter_dir, os.path.join(iter_dir, "model_weights")))
                    os.system('cp {}/metadata.json {}'.format(iter_dir, os.path.join(iter_dir, "model_weights")))

                    # os.chdir(tgi_src_path)
                    this_convert_script = convert_script.format(iter_dir, os.path.join(iter_dir, 'TGI'))
                    print(this_convert_script)

                    os.system(this_convert_script)
                    os.chdir(current_path)

                    os.system('rm -r {}'.format(os.path.join(iter_dir, 'TGI', 'config.json')))
                    os.system('cp {}/* {}'.format(config_path, os.path.join(iter_dir, 'TGI')))

                    time.sleep(10)
                
        time.sleep(10)