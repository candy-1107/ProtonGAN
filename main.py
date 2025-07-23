import os
import random
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from Metric import calculate_shannon_entropy
from Discriminator import StyleGAN2Discriminator
from Generator import StyleGAN2Generator
from encode import one_hot_encode, decode_sequences


def set_seed(seed):
    random.seed(seed)  # 设置Python内置random模块的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch CPU随机数生成器的种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
    # 禁用CuDNN的非确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(26)


PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
train_size = 2000  # 训练集大小
df = pd.read_csv(r"train_sequences.csv", sep="\t")
sequences = df["sequence"].sample(train_size).tolist()
MAX_SEQ_LENGTH = 0
for seq in sequences:
    length = len(seq)
    if length > MAX_SEQ_LENGTH:
        MAX_SEQ_LENGTH = length
print(f"最大序列长度为：{MAX_SEQ_LENGTH}")

real_encoded_sequences = [one_hot_encode(seq, MAX_SEQ_LENGTH) for seq in sequences]
padded_sequences = pad_sequence(
    real_encoded_sequences, batch_first=True, padding_value=0.0
)  #


device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim = 512
w_dim = 512  # 风格空间维度

# 实例化StyleGAN2模型
generator = StyleGAN2Generator(
    z_dim=z_dim, w_dim=w_dim, max_seq_length=MAX_SEQ_LENGTH
).to(device)
discriminator = StyleGAN2Discriminator(
    max_seq_length=MAX_SEQ_LENGTH, sparse_rate=0.3
).to(device)

d_optim = torch.optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.9))
g_optim = torch.optim.Adam(generator.parameters(), lr=0.00005, betas=(0.5, 0.9))

# 划分数据集
dataset = TensorDataset(padded_sequences)
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义损失函数
criterion = nn.BCEWithLogitsLoss()

# 创建学习率调度器
d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    d_optim, mode="min", factor=0.5, patience=3
)
g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    g_optim, mode="min", factor=0.5, patience=3                                               
)

# 训练循环
num_epochs = 10
n = 5
clip_value = 0.02


def save_checkpoint(
    epoch,
    generator,
    discriminator,
    d_optim,
    g_optim,
    d_scheduler,
    g_scheduler,
    save_dir,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint = {
        "epoch": epoch,
        "generator_model_state": generator.state_dict(),
        "discriminator_model_state": discriminator.state_dict(),
        "d_optim_state": d_optim.state_dict(),
        "g_optim_state": g_optim.state_dict(),
        "d_scheduler_state": d_scheduler.state_dict(),
        "g_scheduler_state": g_scheduler.state_dict(),
    }
    checkpoint_path = os.path.join(save_dir, f"checkpoint_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"第{epoch}次模型保存完毕！！！")


def load_checkpoint(
    checkpoint_path,
    generator,
    discriminator,
    d_optim,
    g_optim,
    d_scheduler,
    g_scheduler,
):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    generator.load_state_dict(checkpoint["generator_model_state"])
    discriminator.load_state_dict(checkpoint["discriminator_model_state"])
    d_optim.load_state_dict(checkpoint["d_optim_state"])
    g_optim.load_state_dict(checkpoint["g_optim_state"])
    d_scheduler.load_state_dict(checkpoint["d_scheduler_state"])
    g_scheduler.load_state_dict(checkpoint["g_scheduler_state"])
    start_epoch = checkpoint["epoch"] + 1
    return start_epoch


save_dir = "checkpoints"
# 查找最大的检查点序号
max_epoch = -1
if os.path.exists(save_dir):
    for filename in os.listdir(save_dir):
        if filename.startswith("checkpoint_") and filename.endswith(".pth"):
            try:
                epoch = int(filename.split("_")[1].split(".")[0])
                if epoch > max_epoch:
                    max_epoch = epoch
            except ValueError:
                continue

if max_epoch >= 0:
    checkpoint_path = os.path.join(save_dir, f"checkpoint_{max_epoch}.pth")
    start_epoch = load_checkpoint(
        checkpoint_path,
        generator,
        discriminator,
        d_optim,
        g_optim,
        d_scheduler,
        g_scheduler,
    )
else:
    start_epoch = 0


loss_history_path = "loss_history.csv"
if os.path.exists(loss_history_path):
    loss_df = pd.read_csv(loss_history_path)
    loss_history = loss_df.values.tolist()
else:
    loss_history = []

for epoch in range(start_epoch, start_epoch + num_epochs):
    d_losses = []
    g_losses = []

    for i, real_batch in enumerate(dataloader):
        real_data = real_batch[0].to(device)
        batch_size = real_data.size(0)
        real_data_perm = real_data.permute(0, 2, 1)  # [B, C, L]

        # 判别器训练
        d_optim.zero_grad()
        real_output = discriminator(real_data_perm)
        d_real_loss = criterion(real_output, torch.ones(batch_size, 1).to(device))

        z = torch.randn(batch_size, z_dim).to(device)
        fake_data = generator(z)
        fake_output = discriminator(fake_data.detach())
        d_fake_loss = criterion(fake_output, torch.zeros(batch_size, 1).to(device))

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        # torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_value)
        d_optim.step()
        d_losses.append(d_loss.item())
    avg_d_loss = np.mean(d_losses)

    # 2. 连续训练 n 次生成器
    for _ in range(n):
        for i, real_batch in enumerate(dataloader):
            g_optim.zero_grad()
            z = torch.randn(batch_size, z_dim).to(device)   #
            fake_data = generator(z)
            fake_output = discriminator(fake_data)
            g_loss = criterion(fake_output, torch.ones(batch_size, 1).to(device))
            g_loss.backward()
            # torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_value)
            g_optim.step()
            g_losses.append(g_loss.item())
    avg_g_loss = np.mean(g_losses)

    # 判别器损失只在有训练时记录，否则为 None
    loss_history.append(
        [epoch, avg_g_loss, avg_d_loss]
    )


    save_checkpoint(epoch, generator, discriminator, d_optim, g_optim, d_scheduler, g_scheduler, save_dir)
    print(
        f"第[{epoch}]次---生成器损失：{avg_g_loss}， 辨别器损失：{d_loss.item() if d_loss is not None else 'None'}"
    )
    d_scheduler.step(d_loss.item() if d_loss is not None else 0)
    g_scheduler.step(avg_g_loss)


z = torch.randn(train_size, z_dim).to(device)
generated_sequences = generator(z)
generated_sequences = generated_sequences.permute(0, 2, 1)
protein_sequences = decode_sequences(generated_sequences)

# 打印生成的蛋白序列
for i, seq in enumerate(protein_sequences[0:20]):
    print(f"生成的蛋白质序列 {i + 1}: {seq}")

print(f"生成的序列长度为{len(protein_sequences[0])}")

# 保存损失历史
pd.DataFrame(loss_history, columns=["epoch", "g_loss", "d_loss"]).to_csv(
    loss_history_path, index=False
)
# 重新读取损失历史，确保loss_df已定义
loss_df = pd.read_csv(loss_history_path)

plt.figure(figsize=(10, 6))
plt.plot(loss_df["epoch"], loss_df["g_loss"], label="Generator Loss")
plt.plot(
    loss_df["epoch"][loss_df["d_loss"].notnull()],
    loss_df["d_loss"].dropna(),
    label="Discriminator Loss",
    marker="o",
    linestyle="None",
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_plot_dir = "loss"
if not os.path.exists(loss_plot_dir):
    os.makedirs(loss_plot_dir)
plt.savefig(os.path.join(loss_plot_dir, "loss_curve.png"))
plt.close()
print("损失曲线已保存!!!")


print("-----------性能评估指标如下----------")

# 香农熵
generated_entropy = calculate_shannon_entropy(
    protein_sequences, MAX_SEQ_LENGTH, PROTEIN_ALPHABET
)
print(f"生成序列的香农熵: {generated_entropy:.4f}")
real_entropy = calculate_shannon_entropy(sequences, MAX_SEQ_LENGTH, PROTEIN_ALPHABET)
print(f"真实序列的香农熵: {real_entropy:.4f}")


# boxplot
training_steps = [
    batch_size * i for i in range(num_epochs + 1)
]  # list(range(train_size // batch_size, (train_size // batch_size) * (num_epochs + 1), num_epochs))
identity_data = []


def sequence_recovery_rate(generated_seq, real_seq):
    matches = sum(1 for g, r in zip(generated_seq, real_seq) if g == r)
    return matches / len(real_seq) if len(real_seq) > 0 else 0


for step in training_steps:
    generated_identities = []
    for _ in range(10):
        z = torch.randn(1, z_dim).to(device)
        generated_sequence = generator(z)
        generated_sequence = decode_sequences(
            generated_sequence.permute(0, 2, 1)
        )  # 解码生成序列
        identity = sequence_recovery_rate(
            generated_sequence[0], random.choice(sequences)
        )  # 假设与第一个真实序列比较
        generated_identities.append(identity * 100)  # 转换为百分比
    identity_data.append(generated_identities)

plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=identity_data, color="orange")  # 设置颜色为橙色
ax.set_xticklabels(training_steps, rotation=45, ha="right")
plt.xlabel("Training Steps")
plt.ylabel("Generated sequence identity to natural ones (%)")
plt.title("Generated Sequence Identity Across Training Steps")

boxplot_dir = "boxplot"
if not os.path.exists(boxplot_dir):
    os.makedirs(boxplot_dir)
boxplot_path = os.path.join(boxplot_dir, f"boxplot_{int(time.time())}.png")
plt.savefig(boxplot_path)
plt.close()
print(f"boxplot图片已保存到: {boxplot_path}")
plt.show()


# t-SNE 可视化
real_encoded = np.array(
    [one_hot_encode(seq, MAX_SEQ_LENGTH).cpu().numpy().flatten() for seq in sequences]
)
generated_encoded = np.array(
    [
        one_hot_encode(seq, MAX_SEQ_LENGTH).cpu().numpy().flatten()
        for seq in protein_sequences
    ]
)
all_encoded = np.vstack([real_encoded, generated_encoded])
labels = ["Real"] * len(real_encoded) + ["Generated"] * len(generated_encoded)

tsne = TSNE(n_components=2, random_state=42)
all_encoded_tsne = tsne.fit_transform(all_encoded)

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=all_encoded_tsne[:, 0],
    y=all_encoded_tsne[:, 1],
    hue=labels,
    palette=["blue", "red"],
)
plt.title("t-SNE visualization of real and generated protein sequences")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

SEN_dir = "t-SEN"
if not os.path.exists(SEN_dir):
    os.makedirs(SEN_dir)
tsne_path = os.path.join(SEN_dir, f"tsne_{int(time.time())}.png")
plt.savefig(tsne_path)
plt.close()
print(f"t-SNE图片已保存到: {tsne_path}")

plt.show()
