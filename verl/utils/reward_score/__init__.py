# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code


def _default_compute_score(prompt_str, data_source, solution_str, ground_truth, extra_info=None):
    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        from . import math
        res = math.compute_score(solution_str, ground_truth)

        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads'
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ['hiyouga/geometry3k']:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in ['ifeval']:
        from . import remote_if
        res = remote_if.compute_score(prompt_str, solution_str, ground_truth)
    # elif data_source in ['infinity-instruct']: # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     from . import remote_if_rm
    #     res = remote_if_rm.compute_score(prompt_str, solution_str, ground_truth)
    # elif data_source in ['wildchat']:
    #     from . import remote_if_verifier
    #     res = remote_if_verifier.compute_score(prompt_str, solution_str, ground_truth)
    elif data_source in ['hf_rm']:
        from . import remote_hf_rm
        res = remote_hf_rm.compute_score(prompt_str, solution_str, ground_truth)
    elif data_source in ["if_rm"]:
        from . import remote_if_rm
        res = remote_if_rm.compute_score(prompt_str, solution_str, ground_truth)
    elif data_source in ['if_verifier']:
        from . import remote_if_verifier
        res = remote_if_verifier.compute_score(prompt_str, solution_str, ground_truth)
    else:
        raise NotImplementedError

    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        # return float(res[0])
        return res
