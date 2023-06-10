import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from skopt import gp_minimize
import joblib
import pandas as pd
import openai

openai.api_key = 'YOUR_API_KEY'

# 加载训练好的模型和数据
model = joblib.load('best_model.pkl')
saved_data = joblib.load('saved_data.pkl')
features_o = saved_data['features_o']
target_o = saved_data['target_o']

# 创建GUI窗口
window = tk.Tk()
window.title("BioCharGuard: Soil Cd Immobilization and Guidance Software")
window.geometry("400x400+200+200")
window.iconbitmap("favicon.ico")
window.configure(bg="white")

# 创建输入参数的标签和文本框
param_labels = ["pHsoil", "Ecsoil", "CECsoil", "OC", "ITCC", "SMC"]
entries = []

for i, label in enumerate(param_labels):
    ttk.Label(window, text=label + ":", font=("Arial", 12)).grid(row=i, column=0, padx=5, pady=5)
    entry = ttk.Entry(window, font=("Arial", 12))
    entry.grid(row=i, column=1, padx=5, pady=5)
    entries.append(entry)

# 创建用于显示结果的文本框
result_text = tk.Text(window, height=5, width=40)
result_text.grid(row=len(param_labels) + 1, column=0, columnspan=2, padx=10, pady=10)


# 数据归一化
def normalize_data(data, original_data):
    min_vals = np.min(original_data, axis=0)
    max_vals = np.max(original_data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data


def denormalize_data(normalized_data, original_data):
    min_vals = np.min(original_data)
    max_vals = np.max(original_data)
    denormalized_data = normalized_data * (max_vals - min_vals) + min_vals
    return denormalized_data


# 定义目标函数
def evaluate_model(x):
    # 检查C+H+O+N的限制
    if x[6] + x[7] + x[8] + x[9] <= 100:
        # 构建特征向量
        features = np.array(x).reshape(1, -1)
        features = normalize_data(features, features_o)
        # 预测CIE
        cie = model.predict(features)
        cie = denormalize_data(cie, target_o)
        print("combination of features:", x)
        print("CIE value:", -cie.item())  # 目标函数的负值，确保返回的是一个标量值

        # 保存结果到表格
        result_df = pd.DataFrame({'combination of features': [x], 'CIE value': [-cie.item()]})
        result_df.to_csv('results.csv', mode='a', header=False, index=False)

        return -cie.item()
    else:
        return np.inf


# 定义按钮的点击事件
def run_code():
    # 获取用户输入的参数
    params = [float(entry.get()) for entry in entries]

    # 定义搜索空间范围
    pbounds = [
        (params[0],),  # pHsoil
        (params[1],),  # Ecsoil
        (params[2],),  # CECsoil
        (params[3],),  # OC
        (200, 900),  # Ptemp
        (30, 95),  # C
        (1, 10),  # N
        (1, 10),  # H
        (5, 50),  # O
        (0, 2),  # H_C
        (0, 1),  # O_N_C
        (0, 14),  # pHbc
        (4, 14),  # Ecbc
        (10, 50),  # Ash
        (5, 1000),  # SSA
        (params[4],),  # ITCC
        (0, 2),  # BCC
        (params[5],),  # SMC
        (0, 20),  # BCAR
        (7, 365)  # IT
    ]

    # 进行贝叶斯优化搜索
    res = gp_minimize(evaluate_model, pbounds, n_calls=50, random_state=42)

    # 显示结果
    messagebox.showinfo("result", f"best combination of features：{res.x}\nMaximum CIE：{-res.fun}")

    # 借助openAI搜寻最佳特征的生物质
    temperature = res.x[4]
    carbon_content = res.x[5]
    nitrogen_content = res.x[6]
    hydrogen_content = res.x[7]
    oxygen_content = res.x[8]
    ph_bc = res.x[11]
    Ecbc = res.x[12]
    Ash = res.x[13]
    ssa = res.x[14]
    sentence = f"I want to prepare a biochar that satisfies:biochar carbon content {carbon_content}%, biochar nitrogen content {nitrogen_content}%, biochar hydrogen content {hydrogen_content}%, biochar oxygen content {oxygen_content}%, Biochar PH{ph_bc}, biochar Ec{Ecbc}ms/cm, biochar Ash{Ash}%，specific surface area {ssa}m2/g?Please help me write a detailed preparation scheme and analysis."
    print(sentence)
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=sentence,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7
    )

    response_text = response.choices[0].text.strip()
    print(response_text)
    result_text.delete(1.0, tk.END)  # 清空文本框
    result_text.insert(tk.END, response_text)


# 创建运行按钮
button = ttk.Button(window, text="Run", command=run_code, width=15)
button.grid(row=len(param_labels), column=0, columnspan=2, padx=10, pady=10)

# 运行GUI
window.mainloop()
