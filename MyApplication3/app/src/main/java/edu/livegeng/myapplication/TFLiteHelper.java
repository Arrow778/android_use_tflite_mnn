package edu.livegeng.myapplication;

import android.content.Context;
import android.graphics.Bitmap;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class TFLiteHelper {

    private final Context context;
    private Interpreter interpreter;
    private List<String> labels;
    private ImageProcessor imageProcessor;

    // 根据你的报错，输入尺寸确认是 224 (224*224*3*4 = 602112)
    private static final int INPUT_SIZE = 224;
    private static final String MODEL_PATH = "models/model-mutil-12-01-17-07-35.tflite";
    private static final String LABEL_PATH = "labels/label-mutil.txt";

    public TFLiteHelper(Context context) {
        this.context = context;
        init();
    }

    private void init() {
        try {
            MappedByteBuffer modelBuffer = FileUtil.loadMappedFile(context, MODEL_PATH);
            Interpreter.Options options = new Interpreter.Options();
            interpreter = new Interpreter(modelBuffer, options);
            labels = FileUtil.loadLabels(context, LABEL_PATH);

            // 修复：删除 NormalizeOp
            // 因为你的训练代码里包含了 layers.Lambda(preprocess_input)
            // 所以 TFLite 模型内部会自己处理归一化，这里只需要 Resize 即可
            imageProcessor = new ImageProcessor.Builder()
                    .add(new ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                    // .add(new NormalizeOp(0f, 255f))  <-- 删除或注释掉这一行！！！
                    .build();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String classify(Bitmap bitmap) {
        if (interpreter == null) return "Model Error";

        // ... (前面的代码保持不变: TensorImage加载, process处理, interpreter.run) ...
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(bitmap);
        tensorImage = imageProcessor.process(tensorImage);
        TensorBuffer probabilityBuffer = TensorBuffer.createFixedSize(new int[]{1, labels.size()}, DataType.FLOAT32);
        interpreter.run(tensorImage.getBuffer(), probabilityBuffer.getBuffer().rewind());

        // 1. 获取结果 Map
        Map<String, Float> labeledProbability = new TensorLabel(labels, probabilityBuffer).getMapWithFloatValue();

        // 2. 将 Map 转换为 List 进行排序
        List<Map.Entry<String, Float>> sortedList = new ArrayList<>(labeledProbability.entrySet());

        // 按值(Value)降序排序 (从大到小)
        Collections.sort(sortedList, (o1, o2) -> (o2.getValue()).compareTo(o1.getValue()));

        // 3. 构建字符串
        StringBuilder resultBuilder = new StringBuilder();
        for (Map.Entry<String, Float> entry : sortedList) {
            // 只显示概率大于 1% 的，或者你可以显示全部
            resultBuilder.append(entry.getKey())
                    .append(": ")
                    .append(String.format("%.1f%%", entry.getValue() * 100))
                    .append("\n");
        }

        return resultBuilder.toString();
    }
}