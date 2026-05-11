import os
import r2pipe
import pandas as pd
from collections import defaultdict
import numpy as np
import concurrent.futures
from threading import Lock, Semaphore
import shutil
import threading
import time
import functools
import psutil


class TimeoutError(Exception):
    pass


def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            
            thread.join(seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Function timed out after {seconds} seconds")
            
            if exception[0] is not None:
                raise exception[0]
                
            return result[0]
        return wrapper
    return decorator


class SoSmaliOpcodeProcess:
    def __init__(self, excel_path, base_directory, max_workers=None):
        self.simplified_df = pd.read_excel(excel_path)

        self.opcode_dict = dict(zip(self.simplified_df['arm_opcode'], self.simplified_df['Opcode']))
        self.semantic_dict = dict(zip(self.simplified_df['Opcode'], self.simplified_df['arm_opcode']))
        self.suffixes = self.simplified_df['suffix'].dropna().unique()

        self.transition_matrix = np.zeros((94, 94), dtype=int)
        self.opcode_frequency = defaultdict(int)

        self.opcode_to_index = {format(i, '02x'): i for i in range(94)}

        self.base_directory = base_directory

        self.lock = Lock()
        
        self.max_workers = max_workers or max(1, os.cpu_count() // 2)
        self.semaphore = Semaphore(self.max_workers)
        
        print(f"设置最大工作线程数: {self.max_workers}")

    def find_opcode_with_suffix(self, opcode):
        """尝试匹配操作码和其后缀，如果没有找到，返回 None"""
        if opcode in self.opcode_dict:
            return opcode
        for suffix in self.suffixes:
            if opcode.endswith(suffix):
                full_opcode = opcode[:-len(suffix)]
                if full_opcode in self.opcode_dict:
                    return full_opcode
        return None

    def get_opcode_index(self, opcode):
        """根据操作码获取对应的索引"""
        return self.opcode_to_index.get(opcode, -1)

    def update_transition_matrix(self, previous_opcode, current_opcode):
        """更新转移矩阵，记录操作码之间的转换"""
        previous_index = self.get_opcode_index(previous_opcode)
        current_index = self.get_opcode_index(current_opcode)
        if previous_index != -1 and current_index != -1:
            with self.lock:
                self.transition_matrix[previous_index][current_index] += 1

    def analyze_file(self, txt_file_path):
        """分析单个 txt 文件，更新操作码频率和转移矩阵"""
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as file:
                txt_lines = file.readlines()

            previous_opcode = None
            for line in txt_lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        syntax = parts[2]
                        matched_syntax = self.find_opcode_with_suffix(syntax)
                        if matched_syntax:
                            opcode = self.opcode_dict[matched_syntax]
                            with self.lock:
                                self.opcode_frequency[opcode] += 1
                            if previous_opcode:
                                self.update_transition_matrix(previous_opcode, opcode)
                            previous_opcode = opcode
        except Exception as e:
            print(f"分析文件 {txt_file_path} 时出错: {e}")

    def analyze_directory(self, directory_path):
        """分析目录中的所有 txt 文件"""
        txt_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.analyze_file, file) for file in txt_files]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Analyze file failed: {e}")

    def calculate_transition_probabilities(self):
        """计算状态转移概率"""
        probabilities = np.zeros_like(self.transition_matrix, dtype=float)
        for i in range(94):
            row_sum = np.sum(self.transition_matrix[i])
            if row_sum > 0:
                probabilities[i] = self.transition_matrix[i] / row_sum
        return probabilities

    def save_transition_probabilities(self, probabilities, output_dir):
        """将转移概率保存为 npy 文件"""
        npy_path = os.path.join(output_dir, 'transition_probabilities.npy')
        np.save(npy_path, probabilities)

    @timeout(90)
    def extract_single_so_file(self, so_file_path, txt_output_folder):
        """处理单个SO文件（带90秒超时）"""
        r2 = None
        try:
            with self.semaphore:
                while psutil.cpu_percent(interval=1) > 80:
                    time.sleep(2)
                
                r2 = r2pipe.open(so_file_path, flags=['-2'])
                
                r2.cmd('e anal.bb.maxsize = 32768')
                r2.cmd('e anal.limits = false')
                
                r2.cmd('aa')
                
                functions = r2.cmdj('aflj')
                all_disasm = []
                
                max_functions = 1000
                processed_functions = 0
                
                for func in functions:
                    if processed_functions >= max_functions:
                        break
                        
                    if func.get('size', 0) > 0 and func.get('size', 0) < 10000:
                        disasm = r2.cmd(f'pD 50 @ {func["offset"]}')
                        all_disasm.append(disasm)
                        processed_functions += 1
                
                clean_lines = []
                for disasm_code in all_disasm:
                    for line in disasm_code.splitlines():
                        line = line.strip()
                        if not line:
                            continue

                        if '0x' in line:
                            idx = line.find('0x')
                            parts = line[idx:].split()
                            if len(parts) >= 3:
                                address = parts[0]
                                opcode = parts[1]
                                instruction = " ".join(parts[2:])
                                if ';' in instruction:
                                    instruction = instruction.split(';', 1)[0].strip()
                                clean_lines.append(f"{address}  {opcode}  {instruction}")

                cleaned_disasm = '\n'.join(clean_lines)

                so_file_name = os.path.basename(so_file_path)
                txt_file_path = os.path.join(txt_output_folder, f"{os.path.splitext(so_file_name)[0]}.txt")
                with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(cleaned_disasm)

                return True
                
        except TimeoutError:
            raise
        except Exception as e:
            raise Exception(f"处理文件 {so_file_path} 时出错: {str(e)}")
        finally:
            if r2 is not None:
                try:
                    r2.quit()
                except:
                    pass

    def extract_disassemblies_from_folder(self, so_folder_path, txt_output_folder):
        """提取文件夹中所有 SO 文件的反汇编，并保存为 txt 文件"""
        if not os.path.exists(txt_output_folder):
            os.makedirs(txt_output_folder)

        so_files = [f for f in os.listdir(so_folder_path) if f.endswith('.so')]
        print(f"  找到 {len(so_files)} 个SO文件")
        
        success_count = 0
        timeout_count = 0
        error_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for file_name in so_files:
                so_file_path = os.path.join(so_folder_path, file_name)
                future = executor.submit(self.process_single_so_wrapper, so_file_path, txt_output_folder)
                futures[future] = file_name
            
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                file_name = futures[future]
                print(f"  处理进度 ({i}/{len(so_files)}): {file_name}")
                
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                        print(f"    完成: {file_name}")
                    else:
                        error_count += 1
                        print(f"    处理失败: {file_name}")
                except TimeoutError:
                    timeout_count += 1
                    print(f"    超时跳过: {file_name} (超过90秒)")
                except Exception as e:
                    error_count += 1
                    print(f"    处理失败: {file_name} - {e}")
        
        print(f"  处理完成: 成功 {success_count}, 超时 {timeout_count}, 错误 {error_count}")
        return success_count > 0

    def process_single_so_wrapper(self, so_file_path, txt_output_folder):
        """包装单个SO文件处理，用于线程池"""
        try:
            return self.extract_single_so_file(so_file_path, txt_output_folder)
        except TimeoutError:
            file_name = os.path.basename(so_file_path)
            txt_file_name = f"{os.path.splitext(file_name)[0]}.txt"
            txt_file_path = os.path.join(txt_output_folder, txt_file_name)
            if os.path.exists(txt_file_path):
                try:
                    os.remove(txt_file_path)
                except:
                    pass
            raise
        except Exception as e:
            file_name = os.path.basename(so_file_path)
            txt_file_name = f"{os.path.splitext(file_name)[0]}.txt"
            txt_file_path = os.path.join(txt_output_folder, txt_file_name)
            if os.path.exists(txt_file_path):
                try:
                    os.remove(txt_file_path)
                except:
                    pass
            raise e

    def process_single_apk(self, apk_folder_path):
        """处理单个APK文件夹"""
        try:
            so_directory = os.path.join(apk_folder_path, "so_files")
            
            if not os.path.exists(so_directory) or not os.listdir(so_directory):
                print(f"  跳过: so_files文件夹不存在或为空")
                return False

            txt_directory = os.path.join(apk_folder_path, "so_txt")
            
            if os.path.exists(txt_directory):
                try:
                    shutil.rmtree(txt_directory)
                    print(f"  已删除现有的so_txt文件夹")
                except Exception as e:
                    print(f"  删除so_txt文件夹时出错: {e}")
                    return False
            
            os.makedirs(txt_directory, exist_ok=True)

            print(f"  开始提取反汇编...")
            success = self.extract_disassemblies_from_folder(so_directory, txt_directory)
            
            if not success:
                print(f"  没有成功处理任何SO文件")
                return False
                
            txt_files = [f for f in os.listdir(txt_directory) if f.endswith('.txt')]
            if not txt_files:
                print(f"  未生成任何反汇编txt文件")
                return False
                
            print(f"  生成了 {len(txt_files)} 个反汇编txt文件")

            self.transition_matrix = np.zeros((94, 94), dtype=int)
            self.opcode_frequency = defaultdict(int)

            print(f"  开始分析操作码...")
            self.analyze_directory(txt_directory)

            probabilities = self.calculate_transition_probabilities()
            self.save_transition_probabilities(probabilities, apk_folder_path)
            
            print(f"  成功生成转移概率矩阵")
            return True
            
        except Exception as e:
            print(f"  处理时出错: {e}")
            return False

    def batch_process_apks(self):
        """批量处理所有APK文件夹"""
        all_apk_paths = []
        
        for group_folder_name in os.listdir(self.base_directory):
            group_folder_path = os.path.join(self.base_directory, group_folder_name)
            if os.path.isdir(group_folder_path):
                for apk_folder_name in os.listdir(group_folder_path):
                    apk_folder_path = os.path.join(group_folder_path, apk_folder_name)
                    if os.path.isdir(apk_folder_path):
                        all_apk_paths.append(apk_folder_path)
        
        all_apk_paths = sorted(all_apk_paths)
        
        print(f"找到 {len(all_apk_paths)} 个APK文件夹")
        print("开始批量处理SO文件分析...\n")
        
        success_count = 0
        fail_count = 0
        
        for i, apk_folder_path in enumerate(all_apk_paths, 1):
            print(f"[{i}/{len(all_apk_paths)}] 处理: {os.path.basename(apk_folder_path)}")
            
            if self.process_single_apk(apk_folder_path):
                success_count += 1
            else:
                fail_count += 1
            
            print()
        
        print("=" * 50)
        print(f"批量处理完成!")
        print(f"成功: {success_count}")
        print(f"失败: {fail_count}")
        print(f"总计: {len(all_apk_paths)}")


if __name__ == "__main__":
    excel_path = r'./arm_opcode.xlsx'
    base_directory = r'/newdisk/liuzhuowu/analysis/data/decom'

    analyzer = SoSmaliOpcodeProcess(excel_path, base_directory, max_workers=2)
    analyzer.batch_process_apks()