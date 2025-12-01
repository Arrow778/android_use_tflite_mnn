import os
import tensorflow as tf


def check_with_tensorflow(root_dir):
    total = 0
    invalid = []

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            total += 1
            file_path = os.path.join(class_path, filename)

            try:
                # 用 TensorFlow 实际训练时的解码方式测试
                raw = tf.io.read_file(file_path)
                _ = tf.io.decode_image(raw, channels=3)  # 必须指定 channels=3
            except tf.errors.InvalidArgumentError as e:
                print(f"❌ TF 无法解码: {file_path}")
                invalid.append(file_path)
            except Exception as e:
                print(f"⚠️ 其他错误: {file_path} | {e}")
                invalid.append(file_path)

    print(f"\n✅ 总文件数: {total}")
    print(f"❌ TensorFlow 无法读取: {len(invalid)}")

    if invalid:
        print("\n🔧 建议：删除这些文件，或用 PIL 转存为标准 RGB JPG")
    else:
        print("🎉 所有图片 TensorFlow 都能正常加载！")


# 替换为你的路径
check_with_tensorflow("datasets/train_1")