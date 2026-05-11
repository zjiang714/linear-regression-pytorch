import os,time,torch
from datetime import datetime
from torch import nn,optim
from torch.nn import functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader


# RTX 5070Ti CUDA 极限优化
torch.backends.cudnn.benchmark=True
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
torch.set_float32_matmul_precision('high')
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"


# 功能开关
USE_AMP=True
USE_COMPILE=True
USE_CHANNELS_LAST=True

# 残差块 ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self,input_channels,num_channels,use_1x1conv=False,strides=1):
        super().__init__()
        self.conv1=nn.Conv2d(input_channels,num_channels,kernel_size=3,stride=strides,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(num_channels)
        self.conv2=nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(num_channels)
        self.conv3=nn.Conv2d(input_channels,num_channels,kernel_size=1,stride=strides,bias=False) if use_1x1conv else None

    def forward(self,X):
        Y=F.relu(self.bn1(self.conv1(X)),inplace=True)
        Y=self.bn2(self.conv2(Y))
        if self.conv3:X=self.conv3(X)
        Y+=X
        return F.relu(Y,inplace=True)


# 组装多个残差块
def make_resnet_layer(input_channels,num_channels,num_residuals,first_block=False):
    layers=[]
    for i in range(num_residuals):
        if i==0 and not first_block:
            layers.append(ResidualBlock(input_channels,num_channels,use_1x1conv=True,strides=2))
        elif i==0:
            layers.append(ResidualBlock(input_channels,num_channels))
        else:
            layers.append(ResidualBlock(num_channels,num_channels))
    return layers

# 创建 ResNet 模型
def create_model():
    return nn.Sequential(
        nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        *make_resnet_layer(64,64,2,first_block=True),
        *make_resnet_layer(64,128,2),
        *make_resnet_layer(128,256,2),
        *make_resnet_layer(256,512,2),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(512,10)
    )


# 主训练函数
def train():
    # 自动检测 GPU
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> Device: {device}")

    if device.type=="cuda":
        print(f">>> GPU: {torch.cuda.get_device_name(0)}")

    # CIFAR10 数据增强
    transform_train=transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])

    # 加载 CIFAR10
    train_set=datasets.CIFAR10(root='./data',train=True,download=False,transform=transform_train)

    # DataLoader
    train_loader=DataLoader(train_set,batch_size=1024,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,
        prefetch_factor=4,
        drop_last=True)

    # 创建模型
    model=create_model()
    # channels_last 优化 CNN
    if USE_CHANNELS_LAST:model=model.to(memory_format=torch.channels_last)
    # 模型搬运到 GPU
    model=model.to(device)
    # torch.compile 编译优化
    if USE_COMPILE:
        print("\n>>> torch.compile optimizing...")
        model=torch.compile(model,mode="max-autotune")
    # 损失函数
    criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
    # AdamW fused 优化器
    optimizer=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4,fused=True)
    # Cosine 学习率衰减
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20)

    # AMP 混合精度
    scaler=torch.amp.GradScaler('cuda',enabled=USE_AMP)

    # Epoch 数
    EPOCHS=10

    print("\n>>> Start Training")
    print(f">>> Start Time: {datetime.now()}")

    total_start=time.time()

    # 开始训练
    model.train()

    for epoch in range(EPOCHS):

        epoch_start=time.time()
        running_loss=0.0
        correct=0
        total=0

        for batch_idx,(inputs,labels) in enumerate(train_loader):
            # 数据搬运到 GPU
            inputs=inputs.to(device,non_blocking=True)
            labels=labels.to(device,non_blocking=True)

            # channels_last
            if USE_CHANNELS_LAST:inputs=inputs.to(memory_format=torch.channels_last)
            # 清空梯度
            optimizer.zero_grad(set_to_none=True)

            # AMP 前向传播
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=USE_AMP):
                outputs=model(inputs)
                loss=criterion(outputs,labels)

            # AMP 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 计算准确率
            _,predicted=outputs.max(1)
            total+=labels.size(0)
            correct+=predicted.eq(labels).sum().item()
            running_loss+=loss.item()

            # 打印训练日志
            if (batch_idx+1)%20==0:
                gpu_mem=torch.cuda.memory_allocated()/1024**3

                print(
                    f"[Epoch {epoch+1}/{EPOCHS}] "
                    f"[Batch {batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {running_loss/20:.4f} | "
                    f"Acc: {100.*correct/total:.2f}% | "
                    f"GPU: {gpu_mem:.2f} GB"
                )

                running_loss=0.0

        # 更新学习率
        scheduler.step()
        epoch_time=time.time()-epoch_start
        print(
            f"\n>>> Epoch {epoch+1} Finished "
            f"| Time: {epoch_time:.2f}s "
            f"| Acc: {100.*correct/total:.2f}% "
            f"| LR: {scheduler.get_last_lr()[0]:.6f}\n"
        )

    # 保存模型
    torch.save(model.state_dict(),"resnet5070ti_extreme.pth")

    total_time=time.time()-total_start

    print("\n>>> Training Finished")
    print(f">>> Total Time: {total_time:.2f}s")
    print(f">>> End Time: {datetime.now()}")
    print(">>> Model Saved: resnet5070ti_extreme.pth")


# 主程序入口
if __name__=='__main__':
    train()