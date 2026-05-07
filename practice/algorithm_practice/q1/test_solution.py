import io
import os
import random
import subprocess
import sys

# 设置标准输出编码为UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 暴力标准答案
def brute_force_standard(n, m, a):
    arr = a.copy()
    for _ in range(m):
        min_val = min(arr)
        min_idx = arr.index(min_val)
        arr.pop(min_idx)
    return arr

# 生成符合题目格式的批量测试数据
def generate_test_batch(max_T=10, max_n=200):
    T = random.randint(1, max_T)
    test_cases = []
    expected_output = []
    total_n = 0

    for _ in range(T):
        n = random.randint(1, max_n)
        m = random.randint(0, n-1)
        a = [random.randint(1, n) for _ in range(n)]
        test_cases.append((n, m, a))
        expected_output.append(' '.join(map(str, brute_force_standard(n, m, a))))
        total_n += n
        if total_n > 10000:  # 控制总数据量，避免超时
            break
    
    # 生成标准输入格式的字符串
    input_lines = [str(len(test_cases))]
    for n, m, a in test_cases:
        input_lines.append(f"{n} {m}")
        input_lines.append(' '.join(map(str, a)))
    return '\n'.join(input_lines), expected_output, test_cases

# 黑盒测试主逻辑
def run_black_box_test(num_batches=100):
    random.seed(42)
    passed = 0
    failed = 0

    for batch_idx in range(num_batches):
        # 生成测试数据
        input_str, expected, test_cases = generate_test_batch()
        # 运行算法程序，捕获输入输出
        try:
            result = subprocess.run(
                [sys.executable, os.path.join(SCRIPT_DIR, "solution.py")],
                input=input_str.encode('utf-8'),
                capture_output=True,
                timeout=5  # 超时时间，贴合笔试时间限制
            )
        except subprocess.TimeoutExpired:
            print(f"批次 {batch_idx+1}: 运行超时 - 程序执行时间超过5秒限制")
            failed += 1
            break
        except FileNotFoundError:
            print(f"批次 {batch_idx+1}: 语法错误 - 找不到 solution.py 文件，请检查文件是否存在")
            failed += 1
            break
        except Exception as e:
            print(f"批次 {batch_idx+1}: 运行错误 - 程序执行异常: {str(e)}")
            failed += 1
            break
        
        # 检查程序是否有标准错误输出（语法错误或运行时警告）
        if result.stderr:
            stderr_msg = result.stderr.decode('utf-8', errors='ignore').strip()
            if 'SyntaxError' in stderr_msg:
                print(f"批次 {batch_idx+1}: 语法错误 - solution.py 存在语法错误，无法运行")
                print(f"  错误信息: {stderr_msg[:200]}")
                failed += 1
                break
            elif 'Error' in stderr_msg or 'Exception' in stderr_msg:
                print(f"批次 {batch_idx+1}: 运行时错误 - 程序执行过程中出现异常")
                print(f"  错误信息: {stderr_msg[:200]}")
                failed += 1
                break
        
        # 检查程序是否正常退出
        if result.returncode != 0:
            print(f"批次 {batch_idx+1}: 执行失败 - 程序非正常退出（退出码: {result.returncode}）")
            if result.stderr:
                print(f"  错误信息: {result.stderr.decode('utf-8', errors='ignore')[:200]}")
            failed += 1
            break
        
        # 处理输出结果
        actual_output = result.stdout.decode('utf-8').strip()
        if not actual_output:
            print(f"批次 {batch_idx+1}: 结果错误 - 程序没有输出任何结果")
            failed += 1
            break
            
        actual = actual_output.split('\n')
        actual = [line.strip() for line in actual if line.strip()]

        # 对比结果
        if actual == expected:
            passed += 1
            if (batch_idx + 1) % 10 == 0:
                print(f"已完成 {batch_idx+1} 个批次测试，当前全部通过")
        else:
            print(f"批次 {batch_idx+1}: 结果错误 - 程序输出与预期结果不一致")
            # 定位出错的具体用例
            for i in range(min(len(actual), len(expected))):
                if actual[i] != expected[i]:
                    n, m, a = test_cases[i]
                    print(f"  出错用例详情:")
                    print(f"    输入: n={n}, m={m}, a={a}")
                    print(f"    预期输出: {expected[i]}")
                    print(f"    实际输出: {actual[i]}")
                    break
            
            # 如果输出行数不匹配
            if len(actual) != len(expected):
                print(f"  输出行数不匹配: 预期 {len(expected)} 行，实际 {len(actual)} 行")
            
            failed += 1
            break

    # 最终测试报告
    print("\n" + "="*50)
    print("测试报告")
    print("="*50)
    print(f"测试总批次: {passed + failed}")
    print(f"通过批次: {passed}")
    print(f"失败批次: {failed}")
    if failed == 0:
        print("测试结果: 通过")
        print("✅ 所有黑盒测试通过，程序符合笔试提交要求！")
    else:
        print("测试结果: 失败")
        print("❌ 存在测试失败，请检查程序的输入输出和逻辑！")
    print("="*50)

if __name__ == "__main__":
    run_black_box_test(num_batches=100)