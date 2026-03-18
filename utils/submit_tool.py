#!/usr/bin/env python3
import argparse
import sys
import os
import zipfile
import importlib.util
from pathlib import Path
import numpy as np

def verify_submission(submission_dir):
    """验证提交目录是否包含有效的策略"""
    print(f"Verifying submission in {submission_dir}...")
    
    sub_path = Path(submission_dir)
    policy_file = sub_path / 'policy.py'
    
    if not policy_file.exists():
        print(f"❌ Error: {policy_file} not found!")
        return False
        
    try:
        # 动态导入策略文件
        spec = importlib.util.spec_from_file_location("policy", str(policy_file))
        policy_module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(sub_path))
        spec.loader.exec_module(policy_module)
        
        if not hasattr(policy_module, 'CombatPolicy'):
            print("❌ Error: Class 'CombatPolicy' not found in policy.py!")
            return False
            
        print("✅ Found CombatPolicy class.")
        
        # 简单模拟实例化
        class MockSpace:
            def __init__(self, shape):
                self.shape = shape
                
        obs_space = MockSpace((100,))
        act_space = MockSpace((21,))
        
        policy = policy_module.CombatPolicy(obs_space, act_space)
        print("✅ Policy instantiated successfully.")
        
        # 测试 act 函数
        mock_obs = np.zeros(100)
        action = policy.act(mock_obs)
        if not isinstance(action, np.ndarray) or action.shape != (21,):
            print(f"❌ Error: act() must return np.ndarray of shape (21,), got {type(action)} with shape {getattr(action, 'shape', None)}")
            return False
            
        print("✅ Policy act() verified.")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed with error: {e}")
        return False
    finally:
        if str(sub_path) in sys.path:
            sys.path.remove(str(sub_path))

def pack_submission(submission_dir, output_file="submission.zip"):
    """打包提交文件"""
    if not verify_submission(submission_dir):
        print("❌ Cannot pack: Verification failed.")
        sys.exit(1)
        
    print(f"\nPacking {submission_dir} into {output_file}...")
    sub_path = Path(submission_dir)
    
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(submission_dir):
            for file in files:
                if file.endswith('.pyc') or '__pycache__' in root:
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, submission_dir)
                zf.write(file_path, arcname)
                print(f"  Added {arcname}")
                
    print(f"✅ Successfully created {output_file}")
    print("👉 You can now upload this file to the web platform.")

def main():
    parser = argparse.ArgumentParser(description="CombatBench Submission Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # verify command
    parser_verify = subparsers.add_parser("verify", help="Verify a submission directory")
    parser_verify.add_argument("dir", help="Directory containing policy.py")
    
    # pack command
    parser_pack = subparsers.add_parser("pack", help="Pack a submission directory")
    parser_pack.add_argument("dir", help="Directory containing policy.py")
    parser_pack.add_argument("--out", default="submission.zip", help="Output zip file name")
    
    args = parser_parse = parser.parse_args()
    
    if args.command == "verify":
        if verify_submission(args.dir):
            print("\n🎉 Submission is valid!")
        else:
            sys.exit(1)
    elif args.command == "pack":
        pack_submission(args.dir, args.out)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
