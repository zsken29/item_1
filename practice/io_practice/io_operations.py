# 1. 标准输出
print("=== 标准输出示例 ===")
print("Hello, World!")
print("数字:", 123)
print("浮点数:", 3.14)
print("布尔值:", True)
print("多个值:", "a", "b", "c", sep="-")
print("不换行:", end="")
print("这是同一行")

# 2. 标准输入
print("\n=== 标准输入示例 ===")
name = input("请输入你的名字: ")
age = input("请输入你的年龄: ")
print(f"你好, {name}! 你今年{age}岁。")

# 3. 文件操作 - 写入文件
print("\n=== 文件操作 - 写入文件 ===")
# 获取当前文件所在目录
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# 写入模式 'w' - 会覆盖原有内容
with open(os.path.join(current_dir, "test.txt"), "w", encoding="utf-8") as f:
    f.write("这是第一行\n")
    f.write("这是第二行\n")
    f.write("这是第三行\n")
print("文件写入完成")

# 4. 文件操作 - 读取文件
print("\n=== 文件操作 - 读取文件 ===")
# 读取模式 'r'
with open(os.path.join(current_dir, "test.txt"), "r", encoding="utf-8") as f:
    # 读取全部内容
    content = f.read()
    print("读取全部内容:")
    print(content)

# 逐行读取
print("\n逐行读取:")
with open(os.path.join(current_dir, "test.txt"), "r", encoding="utf-8") as f:
    for line in f:
        print(line.strip())  # strip() 去除换行符

# 5. 文件操作 - 追加内容
print("\n=== 文件操作 - 追加内容 ===")
# 追加模式 'a'
with open(os.path.join(current_dir, "test.txt"), "a", encoding="utf-8") as f:
    f.write("这是追加的第四行\n")
    f.write("这是追加的第五行\n")
print("文件追加完成")

# 验证追加结果
print("\n验证追加结果:")
with open(os.path.join(current_dir, "test.txt"), "r", encoding="utf-8") as f:
    print(f.read())

# 6. 文件操作 - 二进制模式
print("\n=== 文件操作 - 二进制模式 ===")
# 写入二进制文件
with open(os.path.join(current_dir, "test.bin"), "wb") as f:
    f.write(b"Hello, Binary!")

# 读取二进制文件
with open(os.path.join(current_dir, "test.bin"), "rb") as f:
    binary_content = f.read()
    print("二进制内容:", binary_content)
    print("解码为字符串:", binary_content.decode('utf-8'))

# 7. 其他输入输出技巧
print("\n=== 其他输入输出技巧 ===")

# 格式化输出
name = "张三"
age = 20
print("格式化输出1: %s今年%d岁" % (name, age))
print("格式化输出2: {0}今年{1}岁".format(name, age))
print(f"格式化输出3: {name}今年{age}岁")

# 读取多行输入
print("\n读取多行输入（输入空行结束）:")
lines = []
while True:
    line = input()
    if not line:
        break
    lines.append(line)
print("你输入的内容:")
for line in lines:
    print(line)

print("\n所有示例执行完成！")