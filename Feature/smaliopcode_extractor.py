import os
import pandas as pd
from collections import defaultdict
import numpy as np
import concurrent.futures
from feature_config import FILTER_LIBRARIES

class SmaliOpcodeProcess:
    def __init__(self, excel_path):
        """
        初始化 OMMProcess 处理器
        :param excel_path: Dalvik操作码Excel文件路径
        """
        self.excel_path = excel_path
        
        self.syntax_to_index = self._load_excel_config()
        self.third_party_libs = FILTER_LIBRARIES
        
    def _load_excel_config(self):
        """加载Excel配置并返回映射字典 (Syntax -> Index)"""
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"Excel文件未找到: {self.excel_path}")
            
        simplified_df = pd.read_excel(self.excel_path, converters={'Opcode': str})
        
        
        syntax_to_index = {}
        for _, row in simplified_df.iterrows():
            syntax = row['Simplified_Syntax']
            opcode_str = row['Opcode']
            
            try:
                opcode_int = int(opcode_str, 16)
                syntax_to_index[syntax] = opcode_int
            except ValueError:
                print(f"Warning: Could not parse opcode '{opcode_str}' for syntax '{syntax}'")
                
        return syntax_to_index

    def _should_filter(self, rel_path):
        """
        判断相对路径是否匹配第三方库前缀。
        会自动去除顶层的 smali* 目录前缀 (如 smali, smali_classes2)。
        """
        if not self.third_party_libs:
            return False
            
        norm = rel_path.replace('\\', '/')
        
        parts = norm.split('/')
        if parts and parts[0].startswith('smali'):
            if len(parts) > 1:
                norm = '/'.join(parts[1:])
            else:
                pass
        
        for lib in self.third_party_libs:
            lib_norm = lib.replace('\\', '/').strip('/')
            if norm.startswith(lib_norm):
                return True
        return False

    @staticmethod
    def _analyze_file(smali_file_path, syntax_to_index):
        """
        分析单个Smali文件，返回局部统计结果
        :param smali_file_path: smali 文件路径
        :param syntax_to_index: 语法到索引的映射表
        :return: (local_matrix, has_error)
                 local_matrix: np.ndarray (256x256)
        """
        try:
            with open(smali_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            tokens = []
            for line in content.splitlines():
                line = line.strip()
                if not line: continue
                token = line.split(None, 1)[0]
                if token == 'nop': continue
                tokens.append(token)
            
            if not tokens:
                return None, False

            indices = []
            for t in tokens:
                idx = syntax_to_index.get(t)
                if idx is not None:
                    indices.append(idx)
            
            if len(indices) < 2:
                return None, False
            
            arr = np.array(indices, dtype=np.int32)
            
            prev = arr[:-1]
            curr = arr[1:]
            
            flat_indices = prev * 256 + curr
            
            counts = np.bincount(flat_indices, minlength=65536)
            
            local_matrix = counts.reshape(256, 256)
            
            return local_matrix, False
            
        except Exception as e:
            print(f"Error analyzing file {smali_file_path}: {e}")
            return None, True

    def _calculate_transition_probabilities(self, transition_matrix):
        """计算转移概率矩阵"""
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        probabilities = np.divide(
            transition_matrix, 
            row_sums, 
            out=np.zeros_like(transition_matrix, dtype=float), 
            where=row_sums!=0
        )
        return probabilities

    def _save_results(self, transition_matrix, output_directory):
        """保存结果"""
        transition_probabilities = self._calculate_transition_probabilities(transition_matrix)
        npy_path = os.path.join(output_directory, 'dalvik.npy')
        np.save(npy_path, transition_probabilities)

    def process_directory(self, smali_directory, output_directory):
        """
        处理指定的 Smali 目录
        :param smali_directory: smali 源码目录
        :param output_directory: 结果输出目录
        """
        os.makedirs(output_directory, exist_ok=True)
        
        final_transition_matrix = np.zeros((256, 256), dtype=int)
        
        if not os.path.exists(smali_directory):
            print(f"Directory not found: {smali_directory}")
            self._save_results(final_transition_matrix, output_directory)
            return

        smali_files = []
        for root, _, files in os.walk(smali_directory):
            for file in files:
                if file.endswith('.smali'):
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, smali_directory)
                    
                    if self._should_filter(rel_path):
                        continue
                        
                    smali_files.append(abs_path)

        if not smali_files:
            print(f"No smali files found in {smali_directory} (after filtering)")
            self._save_results(final_transition_matrix, output_directory)
            return

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {
                executor.submit(self._analyze_file, file, self.syntax_to_index): file 
                for file in smali_files
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                local_matrix, has_error = future.result()
                if has_error or local_matrix is None:
                    continue
                
                final_transition_matrix += local_matrix
            
        self._save_results(final_transition_matrix, output_directory)


    if __name__ == "__main__":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        excel_path = os.path.join(script_dir, 'res', 'smaliopcode_opcode.xlsx')
        
        tpl_path = os.path.join(script_dir, 'res', 'smaliopcode_filter_tpl.txt')
        
        decom_base_directory = "/newdisk/liuzhuowu/analysis/data/decom"
        
        decom_base_directory = os.path.normpath(decom_base_directory)
    
        print(f"Configuring Analyzer...")
        print(f"Excel: {excel_path}")
        print(f"Filter List: {tpl_path}")
        print(f"Data Directory: {decom_base_directory}")
    
        try:
            analyzer = SmaliOpcodeProcess(excel_path, tpl_path)
            print(f"Loaded {len(analyzer.third_party_libs)} filter rules.")
    
            all_apk_paths = []
            if os.path.exists(decom_base_directory):
                for group_folder_name in os.listdir(decom_base_directory):
                    group_folder_path = os.path.join(decom_base_directory, group_folder_name)
                    if os.path.isdir(group_folder_path):
                        for apk_folder_name in os.listdir(group_folder_path):
                            apk_folder_path = os.path.join(group_folder_path, apk_folder_name)
                            if os.path.isdir(apk_folder_path):
                                all_apk_paths.append(apk_folder_path)
                all_apk_paths = sorted(all_apk_paths)
    
                print(f"Found {len(all_apk_paths)} APK folders to process")
    
                for apk_path in all_apk_paths:
                    try:
                        print(f"Processing: {os.path.basename(apk_path)}")
                        smali_directory = os.path.join(apk_path, 'all_smali')
                        output_directory = apk_path
                        
                        analyzer.process_directory(smali_directory, output_directory)
                        
                    except Exception as e:
                        print(f"Error processing APK folder {apk_path}: {e}")
            else:
                print(f"Base directory not found: {decom_base_directory}")
                print("Tip: If running on Windows with a different path, please update 'decom_base_directory'.")
    
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Initialization failed: {e}")
