import re
import subprocess
import time, os, json
import psutil
import wandb
import argparse
from datetime import datetime, timedelta

current_path = os.getcwd()

SIMPLE_EVALS_DIR = '/mnt/ph/glm-simple-evals'

# tLJURLIcLm3OvU7xgG2T85ZgMvVuOWoWtaxa3uKMNEU0lAUkBRo6Y7PXymcGi9ha
task_info_dict = {
    'simple_evals': {
        'dir': SIMPLE_EVALS_DIR,
        'trace': [
            'python3 evaluate.py --save_dir {model_path} --tgi_url {url} --gpt4_url {vllm_url} --checker vllm --tasks ifeval cello cfbench followbench arena_hard alignbench --proc_num 150 --model_name {model_name} --auto_extract_answer --max_new_tokens 7144'
        ],
        'eval_group': 'simple_evals',
        'sub_tasks': [
            "ifeval", "cello", "cfbench", "followbench", "arena_hard", "alignbench"
        ]
    }
}

config = {
    "9b": {
        'tgi_template': '/workspace/zhenyu/tgi-configs/9b_serving_model_template_original',
        'server_warmup_time': 120,
        'chkp_save_time': 240,
        'tgi_script': 'CUDA_VISIBLE_DEVICES={cuda_devices} text-generation-launcher --model-id {model_path} --num-shard 2 --port {port} --master-port {master_port} --max-concurrent-requests 409600 --max-input-length 8190 --max-total-tokens 8192 --max-batch-prefill-tokens 8192 --trust-remote-code --max-waiting-tokens 2 --cuda-memory-fraction 0.98 --block-size 32 --dtype bfloat16',
        'tasks': [
            'simple_evals'
        ]
    },
    "qw7b": {
        'tgi_template': '/workspace/zhenyu/common_libs/template/qw25_7b/tgi_ori_rope',
        'server_warmup_time': 120,
        'chkp_save_time': 240,
        'tgi_script': 'CUDA_VISIBLE_DEVICES={cuda_devices} text-generation-launcher --model-id {model_path} --num-shard 2 --port {port} --master-port {master_port} --max-concurrent-requests 409600 --max-input-length 8190 --max-total-tokens 8192 --max-batch-prefill-tokens 8192 --trust-remote-code --max-waiting-tokens 2 --cuda-memory-fraction 0.98 --block-size 32 --dtype bfloat16',
        'tasks': [
            'simple_evals'
        ]
    },
    "qw14b": {
        'tgi_template': '/workspace/zhenyu/common_libs/template/qw25_14b/tgi_ori_rope',
        'server_warmup_time': 180,
        'chkp_save_time': 300,
        'tgi_script': 'ENABLE_GPU_AFFINITY=1 DISABLE_PREFILL_LOGPROBS=1 TP_CHUNK_SIZE=8192 EMPTY_CACHE_LOOPS=128000 CUDA_VISIBLE_DEVICES={cuda_devices} text-generation-launcher --model-id {model_path} --num-shard 4 --port {port} --master-port {master_port} --max-concurrent-requests 409600 --max-input-length 16380 --max-total-tokens 16384 --max-batch-prefill-tokens 16384 --trust-remote-code --max-waiting-tokens 2 --cuda-memory-fraction 0.97 --block-size 32 --dtype bfloat16',
        'tasks': [
            'simple_evals'
        ]
    },
    "qw32b": {
        'tgi_template': '/workspace/zhenyu/common_libs/template/qw25_32b/tgi_ori_rope',
        'server_warmup_time': 200,
        'chkp_save_time': 300,
        'tgi_script': 'ENABLE_GPU_AFFINITY=1 DISABLE_PREFILL_LOGPROBS=1 TP_CHUNK_SIZE=8192 EMPTY_CACHE_LOOPS=128000 CUDA_VISIBLE_DEVICES={cuda_devices} text-generation-launcher --model-id {model_path} --num-shard 4 --port {port} --master-port {master_port} --max-concurrent-requests 409600 --max-input-length 8190 --max-total-tokens 8192 --max-batch-prefill-tokens 8192 --trust-remote-code --max-waiting-tokens 2 --cuda-memory-fraction 0.98 --block-size 32 --dtype bfloat16',
        'tasks': [
            'simple_evals'
        ]
    },
}

tgi_src_path = '/usr/src'


def get_need_eval_task_keys(iter_dir, tasks):
    ret = []
    for task_key in tasks:
        task_info = task_info_dict[task_key]
        for sub_task in task_info['sub_tasks']:
            if not os.path.exists(os.path.join(iter_dir, task_info['eval_group'], f'{sub_task}.json')):
                ret.append(task_key)
                break
            js = json.load(open(os.path.join(iter_dir, task_info['eval_group'], f'{sub_task}.json')))
            try:
                score = js['score']
            except:
                ret.append(task_key)
                break
    return ret


def is_folder_recent(folder_path, seconds=600):
    folder_stat = os.stat(folder_path)
    folder_create_time = datetime.fromtimestamp(folder_stat.st_ctime)
    
    now = datetime.now()
    delta = now - folder_create_time
    
    return delta < timedelta(seconds=seconds)


def export_wandb(iter_dir, task_info, project):
    *mgt_path, iter_name = iter_dir.split('/')
    mgt_path = '/'.join(mgt_path)
    exp_name = mgt_path.split('/')[-2]
    # step = int(iter_name.split('_')[-1][4:])
    step = get_step(iter_name)

    # else:
        # exp_name += "_exp"

    wandb_kwargs = {
        'dir': mgt_path,
        'group': exp_name,
        'project': project,
        'job_type': 'eval',
        # to be deleted
        # TODO:
        # 'name': exp_name + '-' + task_info['eval_group'],
        # 'id': exp_name + '-' + task_info['eval_group'],
        'name': exp_name,
        'id': exp_name + '-' + task_info['eval_group'],
        'resume': "allow",
        "entity": "glm-zero",
    }
    wandb.init(
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        **wandb_kwargs
    )
    for sub_task in task_info['sub_tasks']:
        js = json.load(open(os.path.join(iter_dir, task_info['eval_group'], f'{sub_task}.json')))
        if 'score' in js:
            rst = round(js['score'], 2)
        else:
            rst = -1
        wandb.log({sub_task: rst}, step=step)
    wandb.finish()



def get_step(sst):
    # 'epoch=0-step=100-train_rewards=0.334'
    step = re.search(r'step_(\d+)', sst).group(1)
    return int(step)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--tgi_port', type=str, default="8080")
    parser.add_argument('--tgi_master_port', type=str, default="29500")
    parser.add_argument('--cuda_devices', type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument('--vllm_url', type=str, default="http://172.18.200.150:8000/v1")
    parser.add_argument('--model_name', type=str, default="Meta-Llama-3.1-70B-Instruct")
    args = parser.parse_args()

    seen_iter_dirs = set()


    while True:
        with open(args.config_path, 'r') as f:
            monitor_config = json.load(f)
            for key in monitor_config.keys():
                monitor_config[key]['config'] = config[monitor_config[key]['config']]
        
        for mgt_path in monitor_config.keys():
            config_this = monitor_config[mgt_path]['config']
            project = monitor_config[mgt_path]['project']

            # if not mgt_path.endswith('checkpoints') and not mgt_path.endswith('/checkpoints'):
                # mgt_path = os.path.join(mgt_path, 'checkpoints')

            if not os.path.exists(mgt_path):
                print(f"Path {mgt_path} not exists. Waiting for the first ckpt")
                break

            dirs = os.listdir(mgt_path)
                
            iter_dirs = []
            for d in dirs:
                if d.startswith('step_'):
                    iter_dirs.append(d)
                # if d.startswith("epoch=") or d.startswith("'epoch="):
                    # iter_dirs.append(d)

            print(mgt_path)
            ### for openrlhf only
            # dirs = sorted(iter_dirs, key=lambda x: int(x.split('_')[-1][4:]))

            dirs = sorted(iter_dirs, key=lambda x: get_step(x))
            
            for d in dirs:
                iter_dir = os.path.join(mgt_path, d)

                if "unfinished" in iter_dir:
                    print(f"Skip {iter_dir} for unfinished")
                    break
                
                # Check if need eval
                need_eval_task_keys = get_need_eval_task_keys(iter_dir, tasks=config_this['tasks'])
                if len(need_eval_task_keys) < len(config_this['tasks']):
                    if iter_dir not in seen_iter_dirs:
                        already_eval_task_keys = set(config_this['tasks']) - set(need_eval_task_keys)
                        for task_key in already_eval_task_keys:
                            task_info = task_info_dict[task_key]
                            export_wandb(iter_dir, task_info, project)
                        seen_iter_dirs.add(iter_dir)

                if len(need_eval_task_keys) == 0:
                    continue
                
                # If recent saved, skip and wait for totally saved
                if not os.path.exists(os.path.join(iter_dir, "TGI")):
                    print(f"Cannot find TGI in " + iter_dir)
                    break

                mp_rank_dirs = os.listdir(iter_dir + "/TGI")
                mp_rank_dirs = sorted(mp_rank_dirs)

                dir_to_check_time = None
                for mp_rank_d in mp_rank_dirs:
                    if mp_rank_d.endswith('safetensors'):
                        dir_to_check_time = os.path.join(iter_dir, "TGI", mp_rank_d)
                    # dir_to_check_time = 

                if dir_to_check_time is None:
                    dir_to_check_time = iter_dir
                                
                if is_folder_recent(dir_to_check_time, seconds=config_this['chkp_save_time']):
                    print(f"Skip {iter_dir} for recent saved")
                    # force test is conducted step by step for wandb
                    break

                # Start TGI Server
                script = config_this['tgi_script'].format(
                    model_path=os.path.join(iter_dir, 'TGI'),
                    port=args.tgi_port,
                    master_port=args.tgi_master_port,
                    cuda_devices=args.cuda_devices
                )
                process = subprocess.Popen(script, shell=True)
                time.sleep(config_this['server_warmup_time'])

                # Start Eval
                for task_key in need_eval_task_keys:
                    task_info = task_info_dict[task_key]
                    os.chdir(task_info['dir'])
                    for trace in task_info['trace']:
                        os.system(
                            trace.format(
                                model_path=iter_dir,
                                url=f'http://127.0.0.1:{args.tgi_port}/generate',
                                model_name=args.model_name,
                                vllm_url=args.vllm_url
                            )
                        )
                    os.chdir(current_path)
                    export_wandb(iter_dir, task_info, project)
                
                # Kill TGI Server
                parent = psutil.Process(process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                os.system("pkill -f '--master-port {}'".format(args.tgi_master_port))
                time.sleep(20)
        time.sleep(10)