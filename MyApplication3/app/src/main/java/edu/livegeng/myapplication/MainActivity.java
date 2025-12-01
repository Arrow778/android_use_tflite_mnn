package edu.livegeng.myapplication;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import edu.livegeng.myapplication.databinding.ActivityMainBinding;

import android.graphics.Matrix;

public class MainActivity extends AppCompatActivity {

    private ActivityMainBinding binding;
    private TFLiteHelper tfLiteHelper;
    private ExecutorService cameraExecutor;

    // 权限请求启动器
    private final ActivityResultLauncher<String> requestPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    startCamera();
                } else {
                    Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // 设置 ViewBinding
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // 初始化模型帮助类
        tfLiteHelper = new TFLiteHelper(this);

        // 创建后台线程用于处理图片识别，防止卡顿主线程
        cameraExecutor = Executors.newSingleThreadExecutor();

        // 检查权限
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                // 1. 预览配置
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(binding.viewFinder.getSurfaceProvider());

                // 2. 图像分析配置 (核心部分)
                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        // 仅保留最新的一帧，防止分析速度跟不上导致积压
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888) // 直接输出 RGBA Bitmap 格式
                        .build();

                imageAnalysis.setAnalyzer(cameraExecutor, new ImageAnalysis.Analyzer() {
                    @Override
                    public void analyze(@NonNull ImageProxy image) {
                        // 获取 Bitmap
                        Bitmap bitmap = image.toBitmap();

                        // 这里的 bitmap 可能会根据设备旋转，如果识别很差，可能需要处理旋转
                        // 也就是读取 image.getImageInfo().getRotationDegrees()
                        int rotationDegrees = image.getImageInfo().getRotationDegrees();

                        if (bitmap != null) {
                            // 进行识别
                            Bitmap rotatedBitmap = rotateBitmap(bitmap, rotationDegrees);
                            String result = tfLiteHelper.classify(rotatedBitmap);

                            // 切回主线程更新 UI
                            runOnUiThread(() -> binding.tvResult.setText(result));
                        }

                        // 必须关闭 image，否则相机流会卡死
                        image.close();
                    }
                });

                // 3. 绑定到生命周期 (使用后置摄像头)
                CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;
                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);

            } catch (ExecutionException | InterruptedException e) {
                Log.e("CameraX", "Binding failed", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }
    }

    /**
     * 辅助函数，根据角度旋转Bitmap
     */
    private Bitmap rotateBitmap(Bitmap source, float angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }
}