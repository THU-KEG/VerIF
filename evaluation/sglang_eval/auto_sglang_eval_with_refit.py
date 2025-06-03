import subprocess
import time, os, json
import psutil
import wandb
import argparse
import requests
from datetime import datetime, timedelta

current_path = os.getcwd()

task_info_dict = {
    'simple_evals': {
        'dir': '{simple_evals_dir}',
        'trace': [
            'python3 evaluate.py --save_dir {model_path} --sglang_url {sglang_url} --gpt4_url {vllm_url} --backbone sglang --checker vllm --tasks math500 aime omni-math --proc_num 150 --model_name {model_name} --auto_extract_answer --max_new_tokens 4096'
        ],
        'eval_group': 'simple_evals',
        'sub_tasks': [
            'math500', "aime", 'omni-math'
        ]
    }
}

config = {
    "qw7b": {
        'chkp_save_time': 240,
        'tp_size': 2,
        'tasks': [
            'simple_evals'
        ]
    },
    "qw14b": {
        'chkp_save_time': 300,
        'tp_size': 2,
        'tasks': [
            'simple_evals'
        ]
    }
}

sglang_launch_sctipt = 'python3 -m sglang.launch_server --model-path {model_path}  --port {port} --host 0.0.0.0 --tp {tp_size} --dp-size {dp_size}'

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

def export_wandb(iter_dir, task_info, project, wandb_entity):
    *mgt_path, iter_name = iter_dir.split('/')
    mgt_path = '/'.join(mgt_path)
    exp_name = mgt_path.split('/')[-2]
    step = int(iter_name.split('_')[-1])

    wandb_kwargs = {
        'dir': mgt_path,
        'group': exp_name,
        'project': project,
        'job_type': 'eval',
        'name': exp_name,
        'id': exp_name + '-' + task_info['eval_group'],
        'resume': "allow"
    }
    if wandb_entity is not None:
        wandb_kwargs['entity'] = wandb_entity
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

def check_sglang_health(port):
    health_url = 'http://127.0.0.1:{}/health'.format(port)

    try:
        response = requests.get(health_url)
        assert response.status_code == 200
        return True
    except:
        return False

def start_sglang(model_path, port, tp_size):
    dp_size = 8 // tp_size
    cmd = sglang_launch_sctipt.format(
        model_path=model_path,
        port=port,
        tp_size=tp_size,
        dp_size=dp_size
    )
    process = subprocess.Popen(cmd, shell=True)
    while not check_sglang_health(port):
        time.sleep(1)
    return process

def sglang_refit(model_path, port):
    refit_url = 'http://127.0.0.1:{}/update_weights'.format(port)
    refit_success = False
    try:
        data = {
            "model_path": model_path
        }
        response = requests.post(refit_url, json=data, timeout=600)
        print('DEBUG', response, response.json())
        assert response.json()["success"] is True
        refit_success = True
    except:
        pass
    return refit_success

def kill_sglang(process):
    if process is not None:
        process.kill()
    os.system("pkill -f 'sglang.launch_server'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--sglang_port', type=str, default="9999")
    parser.add_argument('--vllm_url', type=str, default="http://172.18.70.4:8000/v1")
    parser.add_argument('--model_name', type=str, default="Meta-Llama-3.1-70B-Instruct")
    args = parser.parse_args()

    seen_iter_dirs = set()
    current_model_type = "qw7b"
    sglang_process_dict = {'sglang_process': None}
    
    while True:
        with open(args.config_path, 'r') as f:
            monitor_config = json.load(f)
            for key in monitor_config.keys():
                monitor_config[key]['model_type'] = monitor_config[key]['config']
                monitor_config[key]['config'] = config[monitor_config[key]['config']]

        for mgt_path in monitor_config.keys():
            config_this = monitor_config[mgt_path]['config']
            model_type = monitor_config[mgt_path]['model_type']
            project = monitor_config[mgt_path]['wandb_project']
            wandb_entity = os.environ.get('WANDB_ENTITY', None)
            if 'wandb_entity' in monitor_config[mgt_path]:
                wandb_entity = monitor_config[mgt_path]['wandb_entity']

            if not mgt_path.endswith('sglang_hf_chkp') and not mgt_path.endswith('/sglang_hf_chkp'):
                mgt_path = os.path.join(mgt_path, 'sglang_hf_chkp')
            if not os.path.exists(mgt_path):
                print(f"Path {mgt_path} not exists. Waiting for the first ckpt")
                break


            dirs = os.listdir(mgt_path)

            iter_dirs = []
            for d in dirs:
                if d.startswith('step_'):
                    iter_dirs.append(d)
            dirs = sorted(iter_dirs, key=lambda x: int(x.split('_')[-1]))
            for d in dirs:
                iter_dir = os.path.join(mgt_path, d)

                # Check if need eval
                need_eval_task_keys = get_need_eval_task_keys(iter_dir, tasks=config_this['tasks'])
                if len(need_eval_task_keys) < len(config_this['tasks']):
                    if iter_dir not in seen_iter_dirs:
                        already_eval_task_keys = set(config_this['tasks']) - set(need_eval_task_keys)
                        for task_key in already_eval_task_keys:
                            task_info = task_info_dict[task_key]
                            export_wandb(iter_dir, task_info, project, wandb_entity)
                        seen_iter_dirs.add(iter_dir)
                if len(need_eval_task_keys) == 0:
                    continue

                # If recent saved, skip and wait for totally saved
                mp_rank_dirs = os.listdir(iter_dir)
                mp_rank_dirs = sorted(mp_rank_dirs)
                dir_to_check_time = None
                for mp_rank_d in mp_rank_dirs:
                    if mp_rank_d.endswith('.safetensors'):
                        dir_to_check_time = os.path.join(iter_dir, mp_rank_d)
                        break
                if dir_to_check_time is None:
                    dir_to_check_time = iter_dir
                if is_folder_recent(dir_to_check_time, seconds=config_this['chkp_save_time']):
                    # force test is conducted step by step for wandb
                    break
                
                # Start Sglang Server
                # 如果没有sglang，则肯定要启动
                if not check_sglang_health(args.sglang_port):
                    sglang_process = start_sglang(iter_dir, args.sglang_port, config_this['tp_size'])
                    sglang_process_dict['sglang_process'] = sglang_process
                    current_model_type = model_type
                else:
                    # 如果正在运行的sglang和当前的模型类型一致，则直接refit
                    refit_success = False
                    if current_model_type == model_type:
                        refit_success = sglang_refit(iter_dir, args.sglang_port)
                    # 如果refit不成功或者模型架构不一样，则先停止再启动
                    if not refit_success or current_model_type != model_type:
                        kill_sglang(sglang_process_dict['sglang_process'])
                        # 确保已经成功停止了
                        while check_sglang_health(args.sglang_port):
                            time.sleep(1)
                        sglang_process = start_sglang(iter_dir, args.sglang_port, config_this['tp_size'])
                        sglang_process_dict['sglang_process'] = sglang_process
                        current_model_type = model_type

                # Start Eval
                for task_key in need_eval_task_keys:
                    task_info = task_info_dict[task_key]
                    os.chdir(monitor_config[mgt_path]['simple_evals_dir'])
                    for trace in task_info['trace']:
                        os.system(
                            trace.format(
                                model_path=iter_dir,
                                sglang_url='http://127.0.0.1:{}/v1/chat/completions'.format(args.sglang_port),
                                vllm_url=args.vllm_url,
                                model_name=args.model_name,
                            )
                        )
                    os.chdir(current_path)
                    export_wandb(iter_dir, task_info, project, wandb_entity)
                time.sleep(5)
        time.sleep(5)