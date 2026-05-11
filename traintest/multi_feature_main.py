import argparse
import os
import sys
import json
import pandas as pd
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from so_detection import run_so_detection
    from icon_detection import run_icon_detection
    from smaliopcode_detection import run_smaliopcode_detection
    from apicall_detection import run_apicall_detection
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保 so_detection.py, icon_detection.py, smaliopcode_detection.py, apicall_detection.py 在同一目录下。")
    sys.exit(1)

def get_arg_parser():
    parser = argparse.ArgumentParser(description="多特征融合检测入口脚本")
    
    parser.add_argument('--root', type=str, default='/newdisk/liuzhuowu/lzw/apks_androzoo', help='包含所有APK家族子文件夹的根目录')
    parser.add_argument('--output_dir', type=str, default='/newdisk/liuzhuowu/baseline/androzoo_result', help='结果输出根目录')
    parser.add_argument('--intermediate_dir', type=str, default='/newdisk/liuzhuowu/baseline/temp', help='中间文件目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--workers', type=int, default=4, help='全局默认数据加载线程数 (若各模块未指定则使用此值)')
    parser.add_argument('--batch_size', type=int, default=32, help='全局默认批处理大小 (若各模块未指定则使用此值)')
    parser.add_argument('--no_progress', action='store_true', help='关闭进度条显示')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='采样比例 (0.0-1.0]')
    
    parser.add_argument('--skip_so', action='store_true', help='跳过 SO 特征检测')
    parser.add_argument('--skip_icon', action='store_true', help='跳过 Icon 特征检测')
    parser.add_argument('--skip_smaliopcode', action='store_true', help='跳过 SmaliOpcode 特征检测')
    parser.add_argument('--skip_apicall', action='store_true', help='跳过 ApiCall 特征检测')
    
    parser.add_argument('--so_epochs', type=int, default=5, help='SO Epochs')
    parser.add_argument('--so_batch_size', type=int, default=32, help='SO Batch Size')
    parser.add_argument('--so_lr', type=float, default=1e-4, help='SO Learning Rate')
    parser.add_argument('--so_threshold', type=float, default=0.85, help='SO Threshold')
    parser.add_argument('--so_workers', type=int, default=0, help='SO Workers')
    
    parser.add_argument('--icon_sample_fraction', type=float, default=0.25, help='Icon Sample Fraction')
    parser.add_argument('--icon_threshold', type=float, default=0.6, help='Icon Threshold')
    parser.add_argument('--icon_workers', type=int, default=8, help='Icon Workers')
    
    parser.add_argument('--smaliopcode_epochs', type=int, default=5, help='SmaliOpcode Epochs')
    parser.add_argument('--smaliopcode_batch_size', type=int, default=64, help='SmaliOpcode Batch Size')
    parser.add_argument('--smaliopcode_lr', type=float, default=0.001, help='SmaliOpcode Learning Rate')
    parser.add_argument('--smaliopcode_threshold', type=float, default=0.85, help='SmaliOpcode Threshold')
    parser.add_argument('--smaliopcode_workers', type=int, default=4, help='SmaliOpcode Workers')

    parser.add_argument('--apicall_gexf_name', type=str, default='community_processed_graph.gexf', help='ApiCall GEXF Name')
    parser.add_argument('--apicall_apk_subdirs', type=str, default='original_apk,repack_apk', help='ApiCall APK Subdirs')
    parser.add_argument('--apicall_sinkhorn_reg', type=float, default=0.1, help='ApiCall Sinkhorn Reg')
    parser.add_argument('--apicall_max_nodes', type=int, default=2000, help='ApiCall Max Nodes')
    parser.add_argument('--apicall_prefilter_margin', type=float, default=0.05, help='ApiCall Prefilter Margin')
    parser.add_argument('--apicall_threshold', type=float, default=0.85, help='ApiCall Threshold')
    parser.add_argument('--apicall_output_dir', type=str, default='./test', help='ApiCall Output Dir')
    
    return parser

def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
    run_output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    summary_metrics = {}
    
    print(f"=== 开始多特征检测任务 ===")
    print(f"根目录: {args.root}")
    print(f"输出目录: {run_output_dir}")
    print("=" * 30)

    if not args.skip_so:
        print("\n>>> 正在运行 SO 特征检测...")
        try:
            so_args = argparse.Namespace(
                root=args.root,
                result_root=os.path.join(run_output_dir, 'so'),
                intermediate_root=os.path.join(args.intermediate_dir, 'so') if args.intermediate_dir else os.path.join(run_output_dir, 'intermediate', 'so'),
                batch_size=args.so_batch_size,
                workers=args.so_workers,
                lr=args.so_lr, 
                epochs=args.so_epochs, 
                seed=args.seed,
                no_progress=args.no_progress,
                mode='train_test', 
                model_path=None,   
                threshold=args.so_threshold,
                margin=0.5,        
                neg_weight=2.0,    
                auto_threshold=False,
                min_recall=0.85,
                target_precision=0.95
            )
            metrics = run_so_detection(so_args)
            summary_metrics['so'] = metrics
            print(f"[SO] 完成。Metrics: {metrics}")
        except Exception as e:
            print(f"[SO] 运行失败: {e}")
            import traceback
            traceback.print_exc()
            summary_metrics['so'] = {'error': str(e)}
    else:
        print("\n>>> 跳过 SO 特征检测")

    if not args.skip_icon:
        print("\n>>> 正在运行 Icon 特征检测...")
        try:
            icon_args = argparse.Namespace(
                root=args.root,
                result_root=run_output_dir,
                intermediate_root=args.intermediate_dir if args.intermediate_dir else os.path.join(run_output_dir, 'intermediate'),
                batch_size=args.batch_size,
                workers=args.icon_workers,
                seed=args.seed,
                threshold=args.icon_threshold,
                no_progress=args.no_progress,
                sample_fraction=args.icon_sample_fraction
            )
            metrics = run_icon_detection(icon_args)
            summary_metrics['icon'] = metrics
            print(f"[Icon] 完成。Metrics: {metrics}")
        except Exception as e:
            print(f"[Icon] 运行失败: {e}")
            import traceback
            traceback.print_exc()
            summary_metrics['icon'] = {'error': str(e)}
    else:
        print("\n>>> 跳过 Icon 特征检测")

    if not args.skip_smaliopcode:
        print("\n>>> 正在运行 SmaliOpcode 特征检测...")
        try:
            smaliopcode_args = argparse.Namespace(
                root=args.root,
                result_root=os.path.join(run_output_dir, 'smaliopcode'),
                intermediate_root=os.path.join(args.intermediate_dir, 'smaliopcode') if args.intermediate_dir else os.path.join(run_output_dir, 'intermediate', 'smaliopcode'),
                batch_size=args.smaliopcode_batch_size,
                workers=args.smaliopcode_workers,
                lr=args.smaliopcode_lr,
                epochs=args.smaliopcode_epochs,
                seed=args.seed,
                no_progress=args.no_progress,
                sample_fraction=1.0
            )
            if hasattr(args, 'sample_fraction'):
                 smaliopcode_args.sample_fraction = args.sample_fraction
            
            smaliopcode_args.sample_fraction = 1.0 

            metrics = run_smaliopcode_detection(smaliopcode_args)
            summary_metrics['smaliopcode'] = metrics
            print(f"[SmaliOpcode] 完成。Metrics: {metrics}")
        except Exception as e:
            print(f"[SmaliOpcode] 运行失败: {e}")
            import traceback
            traceback.print_exc()
            summary_metrics['smaliopcode'] = {'error': str(e)}
    else:
        print("\n>>> 跳过 SmaliOpcode 特征检测")

    if not args.skip_apicall:
        print("\n>>> 正在运行 ApiCall 特征检测...")
        try:
            apicall_args = argparse.Namespace(
                root=args.root,
                output_dir=args.apicall_output_dir if args.apicall_output_dir else os.path.join(run_output_dir, 'apicall_results'),
                gexf_name=args.apicall_gexf_name, 
                seed=args.seed,
                sample_fraction=args.sample_fraction,
                apk_subdirs=args.apicall_apk_subdirs,
                max_nodes=args.apicall_max_nodes,
                threshold=args.apicall_threshold,
                sinkhorn_reg=args.apicall_sinkhorn_reg,
                prefilter_margin=args.apicall_prefilter_margin,
                no_progress=args.no_progress
            )
            metrics = run_apicall_detection(apicall_args)
            summary_metrics['apicall'] = metrics
            print(f"[ApiCall] 完成。Metrics: {metrics}")
        except Exception as e:
            print(f"[ApiCall] 运行失败: {e}")
            import traceback
            traceback.print_exc()
            summary_metrics['apicall'] = {'error': str(e)}
    else:
        print("\n>>> 跳过 ApiCall 特征检测")

    print("\n" + "=" * 30)
    print("=== 多特征检测任务汇总 ===")
    
    summary_file = os.path.join(run_output_dir, 'summary.json')
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_metrics, f, indent=4, ensure_ascii=False)
        print(f"汇总报告已保存至: {summary_file}")
    except Exception as e:
        print(f"保存汇总报告失败: {e}")

    for feature, result in summary_metrics.items():
        print(f"\n[{feature.upper()}] Results:")
        if isinstance(result, dict):
            for k, v in result.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {result}")

if __name__ == '__main__':
    main()
