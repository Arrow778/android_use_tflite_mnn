import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os.path as path

# ======================
# 配置路径（请根据实际情况修改）
# ======================
TFLITE_MODEL_PATH = "models/model-mutil-12-01-19-19-08.tflite"  # 替换为你的 .tflite 文件
LABEL_FILE_PATH = "labels/label-mutil.txt"                      # 标签文件
TEST_DATA_DIR = "datasets/test"                                 # 测试集根目录
IMG_SIZE = (224, 224)                                           # 与训练时一致


def load_labels(label_file: str):
    """从文本文件加载类别标签"""
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    print(f"✅ 加载 {len(labels)} 个类别标签: {labels[:3]}{'...' if len(labels) > 3 else ''}")
    return labels


def load_test_dataset(test_dir: str, img_size: tuple, labels: list):
    """
    从 test_dir 加载所有图片和真实标签（按子目录名匹配）
    返回: images (np.ndarray), true_classes (list of int), filenames (list)
    """
    label_to_index = {name: idx for idx, name in enumerate(labels)}
    images = []
    true_labels = []
    filenames = []

    for class_name in sorted(os.listdir(test_dir)):
        class_path = path.join(test_dir, class_name)
        if not path.isdir(class_path):
            continue
        if class_name not in label_to_index:
            print(f"⚠️ 警告：测试集中的类别 '{class_name}' 不在标签文件中，跳过。")
            continue

        true_idx = label_to_index[class_name]
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = path.join(class_path, img_name)
            try:
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img)
                images.append(img_array)
                true_labels.append(true_idx)
                filenames.append(f"{class_name}/{img_name}")
            except Exception as e:
                print(f"❌ 加载失败 {img_path}: {e}")

    images = np.array(images, dtype=np.float32)
    print(f"✅ 加载 {len(images)} 张测试图片")
    return images, true_labels, filenames


def evaluate_tflite_model(tflite_path: str, images: np.ndarray):
    """使用 TFLite 模型对图像批量推理，返回预测类别索引"""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    batch_size = images.shape[0]
    predictions = []

    for i in range(batch_size):
        # 预处理：添加 batch 维度
        input_data = np.expand_dims(images[i], axis=0)
        # 如果模型需要归一化（如 MobileNetV2），确保输入是 float32 且范围正确
        # 此处假设 TFLite 模型内部已包含 preprocess_input（Lambda 层）
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        pred_class = np.argmax(output[0])
        predictions.append(pred_class)

    return predictions


def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path="confusion_matrix.png"):
    """绘制并保存混淆矩阵"""
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✅ 混淆矩阵已保存: {save_path}")


def main():
    # 1. 加载标签
    class_names = load_labels(LABEL_FILE_PATH)
    num_classes = len(class_names)

    # 2. 加载测试集
    images, true_labels, filenames = load_test_dataset(TEST_DATA_DIR, IMG_SIZE, class_names)

    if len(images) == 0:
        print("❌ 未加载到任何测试图片，请检查路径和格式。")
        return

    # 3. 推理
    print("🔍 正在使用 TFLite 模型进行推理...")
    pred_labels = evaluate_tflite_model(TFLITE_MODEL_PATH, images)

    # 4. 评估指标
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    accuracy = correct / len(true_labels)
    print(f"\n🎯 总体准确率: {accuracy:.4f} ({correct}/{len(true_labels)})")

    # 5. 分类报告
    print("\n📋 分类报告:")
    report = classification_report(true_labels, pred_labels, target_names=class_names, digits=4)
    print(report)

    # 6. 混淆矩阵
    plot_confusion_matrix(true_labels, pred_labels, class_names)

    # 7. （可选）打印部分错误样本
    print("\n🔍 部分错误预测示例:")
    errors = [(i, t, p) for i, (t, p) in enumerate(zip(true_labels, pred_labels)) if t != p]
    for i, (idx, t, p) in enumerate(errors[:5]):  # 打印前5个错误
        print(f"  {filenames[idx]} → 真实: {class_names[t]}, 预测: {class_names[p]}")


if __name__ == "__main__":
    main()