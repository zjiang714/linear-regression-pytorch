# 目标检测和边界框
import os
import urllib.request
import torch
from d2l import torch as d2l
from PIL import Image, ImageFile

# 无头服务器环境下使用非交互式后端
import matplotlib
matplotlib.use('Agg')

# ==========================================
# 🌐 第一阶段：动态路径计算与【防截断】清华源下载引擎
# ==========================================

# 1. 动态获取当前运行的 .py 文件所在的绝对路径
current_file_dir = os.path.dirname(os.path.abspath(__file__))
print(f"📂 1. 当前代码文件所在的绝对路径: {current_file_dir}")

# 2. 计算上一级目录的 img 文件夹绝对路径
target_dir = os.path.abspath(os.path.join(current_file_dir, '../img'))
print(f"🎯 2. 计算出目标图片文件夹应该在: {target_dir}")

# 3. 如果这个 ../img 文件夹在物理上不存在，就自动创建它
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    print(f"📁 3. 物理磁盘上不存在该文件夹，已自动创建。")

# 4. 拼接出最终图片的绝对路径
img_path = os.path.join(target_dir, 'catdog.jpg')
print(f"📍 4. 最终期望的图片物理路径: {img_path}")

# 🔥 【硬核预检】：如果本地文件存在，强行解码验货
if os.path.exists(img_path):
    try:
        with Image.open(img_path) as test_img:
            test_img.load()  
        print(f"✅ 5. 检测到本地已存在完好无损的图片，准备直接读取。")
    except Exception:
        print("⚠️ 5. 检测到本地 catdog.jpg 数据损坏，正在强行清理并准备重新下载...")
        try:
            os.remove(img_path)
        except Exception:
            pass

# 5. 安全下载
if not os.path.exists(img_path):
    print("🌐 6. 正在从【GitHub Raw】安全下载经典的猫狗图片...")
    url = 'https://raw.githubusercontent.com/d2l-ai/d2l-en/master/img/catdog.jpg'
    
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                with open(img_path, 'wb') as f:
                    f.write(resp.read())
            with Image.open(img_path) as test_img:
                test_img.load()
            print("🎉 7. 图片下载并写入成功！数据完好无损！")
            break
        except Exception as e:
            if attempt < 2:
                print(f"🔄 7. 下载遭遇网络抖动 ({type(e).__name__})，正在尝试第 {attempt + 2} 次重试...")
            else:
                print("❌ 7. 连续 3 次下载均遭遇残缺。激活终极防御：允许 PIL 忽略截断错误。")
                ImageFile.LOAD_TRUNCATED_IMAGES = True

# 🚨 【硬核断言】：在 matplotlib 读图前，做最后的物理生存检查
print(f"🔍 8. 正在做读取前的终极物理检查...")
print(f"   - 该路径在操作系统中是否存在？ -> {os.path.exists(img_path)}")
if os.path.exists(img_path):
    print(f"   - 文件大小是多少字节？ -> {os.path.getsize(img_path)} 字节")
else:
    raise FileNotFoundError(f"❌ 关键错误：在路径 {img_path} 下根本没有找到 catdog.jpg 文件！请检查当前脚本的执行工作目录。")

# ==========================================
# 📊 第二阶段：图片读取与基础画布配置
# ==========================================
d2l.set_figsize()
img = d2l.plt.imread(img_path)
print("🖼️ 9. Matplotlib 成功将图片读入内存！")


# ==========================================
# 📐 第三阶段：边界框数学坐标转换核心算法
# ==========================================
#@save
def box_corner_to_center(boxes):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
boxes = torch.tensor((dog_bbox, cat_bbox))
validation = box_center_to_corner(box_corner_to_center(boxes)) == boxes
print(f"📐 10. 坐标转换函数互逆验证结果:\n{validation}")


# ==========================================
# 🎨 第四阶段：Matplotlib 画框工具函数与渲染
# ==========================================
#@save
def bbox_to_rect(bbox, color):
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))

output_path = os.path.join(current_file_dir, 'catdog_bbox.png')
d2l.plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"📸 11. 边界框可视化图片已成功保存至: {output_path}")