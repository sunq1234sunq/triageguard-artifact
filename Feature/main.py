import os
import sys
import shutil
import subprocess
import argparse
import re
import csv
import difflib
import time
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
from concurrent.futures import ThreadPoolExecutor, as_completed

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import icon_extractor
import smaliopcode_extractor
import apicall_extractor
import apicall_enhance
import so_preprocessor
import so_extractor
from feature_config import FILTER_LIBRARIES

def decompile_apk(apk_path, output_dir):
    """使用apktool反编译APK文件"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        apk_name = os.path.splitext(os.path.basename(apk_path))[0]
        decompiled_path = os.path.join(output_dir, apk_name)
        
        if os.path.exists(decompiled_path):
            print(f"已存在反编译目录，跳过反编译: {decompiled_path}")
            return decompiled_path
        
        cmd = ["apktool", "d", apk_path, "-o", decompiled_path, "-f"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"APK反编译成功: {apk_path}")
            
            _preprocess_remove_libs(decompiled_path)
            
            return decompiled_path
        else:
            print(f"APK反编译失败: {apk_path}, 错误信息: {result.stderr}")
            return None
    except Exception as e:
        print(f"反编译APK时出错 {apk_path}: {e}")
        return None

def _preprocess_remove_libs(decompiled_path):
    """
    预处理：删除第三方库的 smali 文件
    """
    libs = [lib.replace('.', os.sep) for lib in FILTER_LIBRARIES]
    
    if not libs:
        return

    print("  正在执行Smali预过滤(删除第三方库)...")
    count = 0
    
    for root_dir in os.listdir(decompiled_path):
        if not root_dir.startswith('smali'):
            continue
            
        full_root = os.path.join(decompiled_path, root_dir)
        if not os.path.isdir(full_root):
            continue
            
        for lib_path in libs:
            target_dir = os.path.join(full_root, lib_path)
            if os.path.exists(target_dir):
                try:
                    shutil.rmtree(target_dir)
                    count += 1
                except:
                    pass
    
    if count > 0:
        print(f"  已移除 {count} 个第三方库目录")

def build_all_smali(decompiled_path):
    all_smali_dir = os.path.join(decompiled_path, 'all_smali')
    if os.path.exists(all_smali_dir):
        try:
            shutil.rmtree(all_smali_dir)
        except Exception:
            pass
    os.makedirs(all_smali_dir, exist_ok=True)
    libs = [lib.replace('\\', '/').strip('/') for lib in FILTER_LIBRARIES]
    sources = [d for d in os.listdir(decompiled_path) if d.startswith('smali') and os.path.isdir(os.path.join(decompiled_path, d))]
    copied = 0
    for src in sources:
        src_root = os.path.join(decompiled_path, src)
        for root, _, files in os.walk(src_root):
            for f in files:
                if not f.endswith('.smali'):
                    continue
                abs_path = os.path.join(root, f)
                rel = os.path.relpath(abs_path, src_root).replace('\\', '/')
                skip = False
                for lib in libs:
                    if rel.startswith(lib):
                        skip = True
                        break
                if skip:
                    continue
                dest_path = os.path.join(all_smali_dir, rel)
                dest_dir = os.path.dirname(dest_path)
                os.makedirs(dest_dir, exist_ok=True)
                try:
                    shutil.copy2(abs_path, dest_path)
                    copied += 1
                except Exception:
                    continue
    return all_smali_dir, copied

def process_icon_feature(apk_folder, output_folder):
    """处理图标特征"""
    try:
        icon_output_dir = os.path.join(output_folder, "icon")
        os.makedirs(icon_output_dir, exist_ok=True)
        
        icon_file = icon_extractor.collect_apk_icons(apk_folder)
        if icon_file:
            dest_path = os.path.join(icon_output_dir, os.path.basename(icon_file))
            shutil.copy2(icon_file, dest_path)
            print(f"图标特征处理完成: {apk_folder}")
            return True
        else:
            print(f"未找到图标: {apk_folder}")
            return False
    except Exception as e:
        print(f"处理图标特征时出错 {apk_folder}: {e}")
        return False

def process_smaliopcode_feature(apk_folder, output_folder):
    """处理OMM特征"""
    try:
        smaliopcode_output_dir = os.path.join(output_folder, "smaliopcode")
        os.makedirs(smaliopcode_output_dir, exist_ok=True)
        
        excel_path = os.path.join(current_dir, "res", "smaliopcode_opcode.xlsx")
        smali_directory = os.path.join(apk_folder, "all_smali")
        
        if not os.path.exists(smali_directory):
            smali_directory = os.path.join(apk_folder, "smali")
        
        if not os.path.exists(smali_directory):
            print(f"Smali目录不存在: {smali_directory}")
            return False
            
        analyzer = smaliopcode_extractor.SmaliOpcodeProcess(excel_path)
        analyzer.process_directory(smali_directory, smaliopcode_output_dir)
        
        generated_file = os.path.join(smaliopcode_output_dir, "dalvik.npy")
        if os.path.exists(generated_file):
            print(f"SmaliOpcode特征处理完成: {apk_folder}")
            return True
        else:
            print(f"SmaliOpcode特征文件未生成: {apk_folder}")
            return False
    except Exception as e:
        print(f"处理SmaliOpcode特征时出错 {apk_folder}: {e}")
        return False

def process_apicall_feature(apk_folder, output_folder):
    """处理SFCG特征"""
    try:
        apicall_output_dir = os.path.join(output_folder, "apicall")
        os.makedirs(apicall_output_dir, exist_ok=True)
        
        entities_txt = os.path.join(current_dir, "res", "apicall_entities.txt")
        entity_embedding_pkl = os.path.join(current_dir, "res", "apicall_embeddings.pkl")
        
        required_files = [entities_txt, entity_embedding_pkl]
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"缺少必要文件: {file_path}")
                return False
        
        s_nodes_txt = apicall_enhance.read_nodes_from_txt(entities_txt)
        s_nodes_with_vectors = apicall_enhance.load_nodes_with_vectors(entity_embedding_pkl)
        
        processor = apicall_extractor.ApiCallProcessor(
            directory=apk_folder,
            s_nodes_txt=s_nodes_txt,
            s_nodes_with_vectors=s_nodes_with_vectors,
            nodes_txt=s_nodes_txt,
            nodes_with_vectors=s_nodes_with_vectors
        )
        
        processor.run(apicall_output_dir)
        
        print(f"ApiCall特征处理完成: {apk_folder}")
        return True
    except Exception as e:
        print(f"处理ApiCall特征时出错 {apk_folder}: {e}")
        return False

def process_so_feature(apk_folder, output_folder):
    """处理SO特征"""
    try:
        so_output_dir = os.path.join(output_folder, "so")
        os.makedirs(so_output_dir, exist_ok=True)
        
        if not so_preprocessor.extract_so_files(apk_folder):
            print(f"提取SO文件失败: {apk_folder}")
            return False
            
        excel_path = os.path.join(current_dir, "res", "so_arm_opcode.xlsx")
        
        analyzer = so_extractor.SoSmaliOpcodeProcess(excel_path, "")
        success = analyzer.process_single_apk(apk_folder)
        
        if success:
            generated_file = os.path.join(apk_folder, "transition_probabilities.npy")
            if os.path.exists(generated_file):
                dest_path = os.path.join(so_output_dir, "transition_probabilities.npy")
                shutil.move(generated_file, dest_path)
            print(f"SO特征处理完成: {apk_folder}")
            return True
        else:
            print(f"SO特征处理失败: {apk_folder}")
            return False
    except Exception as e:
        print(f"处理SO特征时出错 {apk_folder}: {e}")
        return False

def process_single_apk(apk_file, output_base_dir, decompiled_base_dir, enable_timing=False, include_size=False):
    """处理单个APK文件，生成所有特征"""
    try:
        timing_data = {}
        start_total = time.time()
        
        apk_name = os.path.splitext(os.path.basename(apk_file))[0]
        timing_data['apk_name'] = apk_name
        
        if include_size:
            try:
                size_mb = os.path.getsize(apk_file) / (1024 * 1024)
                timing_data['apk_size_mb'] = f"{size_mb:.2f}"
            except Exception:
                timing_data['apk_size_mb'] = '/'

        output_folder = os.path.join(output_base_dir, apk_name)
        
        t0 = time.time()
        decompiled_folder = decompile_apk(apk_file, decompiled_base_dir)
        decompile_duration = time.time() - t0
        
        if decompiled_folder:
            timing_data['decompile_time'] = decompile_duration
        else:
            timing_data['decompile_time'] = '/'
            timing_data['icon_time'] = '/'
            timing_data['filter_smali_time'] = '/'
            timing_data['smaliopcode_matrix_time'] = '/'
            timing_data['apicall_graph_time'] = '/'
            timing_data['so_disasm_time'] = '/'
            timing_data['so_matrix_time'] = '/'
            timing_data['so_time'] = '/'
            timing_data['total_time'] = time.time() - start_total
            if enable_timing:
                return timing_data
            else:
                return False

        results = []
        
        t0 = time.time()
        if process_icon_feature(decompiled_folder, output_folder):
            timing_data['icon_time'] = time.time() - t0
            results.append(True)
        else:
            timing_data['icon_time'] = '/'
            results.append(False)

        t0 = time.time()
        all_smali_dir, _ = build_all_smali(decompiled_folder)
        timing_data['filter_smali_time'] = time.time() - t0

        t0 = time.time()
        smaliopcode_out = os.path.join(output_folder, "smaliopcode", "dalvik.npy")
        if os.path.exists(smaliopcode_out):
            print(f"SmaliOpcode矩阵已存在，跳过SmaliOpcode特征: {apk_name}")
            timing_data['smaliopcode_matrix_time'] = 0
            results.append(True)
        else:
            smaliopcode_output_dir = os.path.join(output_folder, "smaliopcode")
            os.makedirs(smaliopcode_output_dir, exist_ok=True)
            excel_path = os.path.join(current_dir, "res", "smaliopcode_opcode.xlsx")
            smali_directory = all_smali_dir
            if os.path.exists(smali_directory):
                analyzer = smaliopcode_extractor.SmaliOpcodeProcess(excel_path)
                analyzer.process_directory(smali_directory, smaliopcode_output_dir)
                timing_data['smaliopcode_matrix_time'] = time.time() - t0
                results.append(True)
            else:
                timing_data['smaliopcode_matrix_time'] = '/'
                results.append(False)

        t0 = time.time()
        apicall_output_dir = os.path.join(output_folder, "apicall")
        os.makedirs(apicall_output_dir, exist_ok=True)
        apicall_gexf = os.path.join(apicall_output_dir, "community_processed_graph.gexf")
        entities_txt = os.path.join(current_dir, "res", "apicall_entities.txt")
        entity_embedding_pkl = os.path.join(current_dir, "res", "apicall_embeddings.pkl")
        if os.path.exists(apicall_gexf):
             print(f"ApiCall图已存在，跳过ApiCall特征: {apk_name}")
             timing_data['apicall_graph_time'] = 0
             results.append(True)
        elif os.path.exists(entities_txt) and os.path.exists(entity_embedding_pkl):
             s_nodes_txt = apicall_enhance.read_nodes_from_txt(entities_txt)
             s_nodes_with_vectors = apicall_enhance.load_nodes_with_vectors(entity_embedding_pkl)
             processor = apicall_extractor.ApiCallProcessor(
                 directory=all_smali_dir,
                 s_nodes_txt=s_nodes_txt,
                 s_nodes_with_vectors=s_nodes_with_vectors,
                 nodes_txt=s_nodes_txt,
                 nodes_with_vectors=s_nodes_with_vectors
             )
             processor.run(apicall_output_dir)
             timing_data['apicall_graph_time'] = time.time() - t0
             results.append(True)
        else:
             timing_data['apicall_graph_time'] = '/'
             results.append(False)

        t0 = time.time()
        so_out = os.path.join(output_folder, "so", "transition_probabilities.npy")
        if os.path.exists(so_out):
            print(f"SO矩阵已存在，跳过SO特征: {apk_name}")
            timing_data['so_disasm_time'] = 0
            timing_data['so_matrix_time'] = 0
            timing_data['so_time'] = 0
            results.append(True)
        else:
            so_output_dir = os.path.join(output_folder, "so")
            os.makedirs(so_output_dir, exist_ok=True)
            t_dis = time.time()
            if so_preprocessor.extract_so_files(decompiled_folder):
                excel_path = os.path.join(current_dir, "res", "so_arm_opcode.xlsx")
                analyzer = so_extractor.SoSmaliOpcodeProcess(excel_path, "")
                so_directory = os.path.join(decompiled_folder, "so_files")
                txt_directory = os.path.join(decompiled_folder, "so_txt")
                if os.path.exists(txt_directory):
                    try:
                        shutil.rmtree(txt_directory)
                    except Exception:
                        pass
                os.makedirs(txt_directory, exist_ok=True)
                try:
                    analyzer.extract_disassemblies_from_folder(so_directory, txt_directory)
                    timing_data['so_disasm_time'] = time.time() - t_dis
                    t_mat = time.time()
                    analyzer.analyze_directory(txt_directory)
                    probs = analyzer.calculate_transition_probabilities()
                    analyzer.save_transition_probabilities(probs, so_output_dir)
                    timing_data['so_matrix_time'] = time.time() - t_mat
                    timing_data['so_time'] = time.time() - t0
                    results.append(True)
                except Exception:
                    timing_data['so_disasm_time'] = '/'
                    timing_data['so_matrix_time'] = '/'
                    timing_data['so_time'] = '/'
                    results.append(False)
            else:
                timing_data['so_disasm_time'] = '/'
                timing_data['so_matrix_time'] = '/'
                timing_data['so_time'] = '/'
                results.append(False)
        
        success_count = sum(results)
        
        timing_data['total_time'] = time.time() - start_total
        
        if enable_timing:
            return timing_data
        else:
            return True
    except Exception as e:
        print(f"处理APK时出错 {apk_file}: {e}")
        return False if not enable_timing else None

def main(input_apk_dir, output_dir, decompiled_dir, max_workers=4, enable_timing=False, include_size=False):
    """
    主函数：处理APK文件，为每个APK反编译并生成四个特征文件
    
    参数:
    input_apk_dir: 包含APK文件的输入目录
    output_dir: 输出目录，将按照APK分别存放特征文件
    decompiled_dir: 反编译文件的存储目录
    max_workers: 最大并发处理数
    enable_timing: 是否启用计时并保存结果
    include_size: 是否统计APK大小
    """
    if not os.path.exists(input_apk_dir):
        print(f"输入目录不存在: {input_apk_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(decompiled_dir, exist_ok=True)
    
    apk_files = []
    for item in os.listdir(input_apk_dir):
        item_path = os.path.join(input_apk_dir, item)
        if os.path.isfile(item_path) and item.lower().endswith('.apk'):
            apk_files.append(item_path)
    print(f"找到 {len(apk_files)} 个APK文件（默认全量）")
    
    apk_files_to_process = []
    skipped_count = 0
    for apk_path in apk_files:
        apk_name = os.path.splitext(os.path.basename(apk_path))[0]
        out_apk_dir = os.path.join(output_dir, apk_name)
        smaliopcode_done = os.path.exists(os.path.join(out_apk_dir, "smaliopcode", "dalvik.npy"))
        so_done = os.path.exists(os.path.join(out_apk_dir, "so", "transition_probabilities.npy"))
        if smaliopcode_done and so_done:
            skipped_count += 1
        else:
            apk_files_to_process.append(apk_path)
    print(f"断点续跑过滤：将处理 {len(apk_files_to_process)} 个APK，跳过 {skipped_count} 个已完成APK")
    
    timing_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_apk = {
            executor.submit(process_single_apk, apk_file, output_dir, decompiled_dir, enable_timing, include_size): apk_file 
            for apk_file in apk_files_to_process
        }
        
        if tqdm:
            iterator = tqdm(as_completed(future_to_apk), total=len(apk_files_to_process), desc="处理APK进度")
        else:
            iterator = as_completed(future_to_apk)
            
        for future in iterator:
            apk_file = future_to_apk[future]
            try:
                result = future.result()
                if enable_timing and isinstance(result, dict):
                    timing_results.append(result)
            except Exception as e:
                print(f"APK处理生成异常 {apk_file}: {e}")
                
    if enable_timing and timing_results:
        csv_file = os.path.join(output_dir, "timing_stats.csv")
        try:
            fieldnames = ['apk_name', 'decompile_time', 'filter_smali_time', 'icon_time', 'smaliopcode_matrix_time', 'apicall_graph_time', 'so_disasm_time', 'so_matrix_time', 'so_time', 'total_time']
            if include_size:
                fieldnames.insert(1, 'apk_size_mb')
                
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(timing_results)
            print(f"计时统计已保存至: {csv_file}")
        except Exception as e:
            print(f"保存计时统计失败: {e}")
        
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APK特征提取工具")
    parser.add_argument("input_dir", help="输入APK文件夹路径")
    parser.add_argument("output_dir", help="输出特征文件夹路径")
    parser.add_argument("--decom_dir", default="decom", help="反编译文件夹路径 (默认: decom)")
    parser.add_argument("--workers", type=int, default=4, help="最大线程数 (默认: 4)")
    parser.add_argument("--time", action="store_true", help="是否开启计时统计 (默认: True)")
    parser.add_argument("--size", action="store_true", help="是否统计APK大小 (默认: False, 需配合 --time 使用)")
    
    args = parser.parse_args()
    
    decom_dir = args.decom_dir
    if not os.path.isabs(decom_dir):
        decom_dir = os.path.join(args.output_dir, decom_dir)
        
    main(args.input_dir, args.output_dir, decom_dir, args.workers, enable_timing=True, include_size=args.size)
