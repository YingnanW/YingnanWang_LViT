import matplotlib.pyplot as plt
import numpy as np

# 数据
models = ['VGG16', 'ResNet50', 'GoogleNet', 'MobileNetV2', 'DenseNet121', 'EfficientNetB0', 'DeiT-S/16', 'Swin-T', 'Twins-SVT-S', 'MaxViT-T', 'Our Model']

accuracy_scores = {
    'LCIGA': [0.8391, 0.8459, 0.8666, 0.8638, 0.8487, 0.8514, 0.8281, 0.8157, 0.8157, 0.8102, 0.8803],
    'WLIGA': [0.8691, 0.8584, 0.8605, 0.8466, 0.8476, 0.8863, 0.8283, 0.8144, 0.8305, 0.7944, 0.9040],
    'LCIIM': [0.9131, 0.8750, 0.9131, 0.9253, 0.9040, 0.9207, 0.8659, 0.8582, 0.8521, 0.9299, 0.9390],
    'WLIIM': [0.9234, 0.8935, 0.9342, 0.9043, 0.8935, 0.9246, 0.8493, 0.8696, 0.8864, 0.9390, 0.9402]
}

f1_scores = {
    'LCIGA': [0.8321, 0.8340, 0.8522, 0.8472, 0.8276, 0.8440, 0.8229, 0.8088, 0.7970, 0.7915, 0.8699],
    'WLIGA': [0.8140, 0.8024, 0.8060, 0.7837, 0.7835, 0.8450, 0.7568, 0.7250, 0.7476, 0.7179, 0.8669],
    'LCIIM': [0.8950, 0.8440, 0.8915, 0.9050, 0.8764, 0.9000, 0.8429, 0.8324, 0.8220, 0.9142, 0.9223],
    'WLIIM': [0.8661, 0.8202, 0.8852, 0.8312, 0.8078, 0.8732, 0.7439, 0.7636, 0.7826, 0.8953, 0.8980]
}
# 创建图表
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
# 颜色设置
# 颜色设置
colors = ['#1f77b4','#ff7f0e', '#2ca02c', '#ffbb78', '#2ca02c', '#98df8a']

# 创建图表
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
width = 0.5  # 柱状图宽度
# 遍历每个数据集
for i, (dataset, f1_data) in enumerate(f1_scores.items()):
    row = i // 2
    col = i % 2
    ax = axs[row, col]
    
    # 绘制柱状图（准确率）
    bars = []
    for j, model in enumerate(models):
        if j < 6:
            bars.append(ax.bar(j , accuracy_scores[dataset][j], width=0.7, color=colors[0], label='Acc (CNN)' if j == 0 else ""))
        elif j < 10:
            bars.append(ax.bar(j , accuracy_scores[dataset][j], width=0.7, color=colors[1], label='Acc (ViT)' if j == 6 else ""))
        else:
            bars.append(ax.bar(j , accuracy_scores[dataset][j], width=0.7, color=colors[2], label='Acc (our)' if j == 10 else ""))
    
    # 绘制折线图（F1分数）
    ax.plot(models, f1_data, 'o-', color='firebrick', label='F1 Score (Line)')
    ax.plot([10], [f1_data[-1]], '*k', label='Our Model F1 Score (Star)')
    
    ax.set_title(f'{dataset}')
    ax.set_ylabel('Score')
    ax.set_ylim(0.7, 1.0)  # 设置y轴范围
    if i == 0:
        ax.legend()
    ax.grid(axis='y', linestyle='--', linewidth=0.7)
    
    # 设置x轴标签

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
 
    
 
        
    # 在柱状图上方显示数值
    for bar in bars:
        # 获取柱状图容器中的每个Rectangle对象
        for rect in bar:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')        

# 调整布局
plt.tight_layout()

# 保存为PDF
plt.savefig('performance_comparison.pdf', format='pdf', bbox_inches='tight')

# 显示图表
plt.show()