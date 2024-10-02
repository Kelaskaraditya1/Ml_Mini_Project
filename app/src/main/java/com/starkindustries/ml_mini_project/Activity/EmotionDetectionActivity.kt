package com.starkindustries.ml_mini_project.Activity

import android.content.ContentResolver
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.Gravity
import android.widget.LinearLayout
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.databinding.DataBindingUtil
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.starkindustries.ml_mini_project.Keys.Keys
import com.starkindustries.ml_mini_project.R
import com.starkindustries.ml_mini_project.databinding.ActivityEmotionDetectionBinding
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class EmotionDetectionActivity : AppCompatActivity() {
    lateinit var binding: ActivityEmotionDetectionBinding
    private lateinit var tflite: Interpreter
    private val IMAGE_SIZE = 224

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_emotion_detection)
        binding = DataBindingUtil.setContentView(this, R.layout.activity_emotion_detection)

        try {
            val tfliteModel = loadModelFile("model.tflite")
            tflite = Interpreter(tfliteModel)
        } catch (e: Exception) {
            e.printStackTrace()
        }

        binding.GalleryButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK)
            intent.setData(MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent, Keys.GALLERY_CODE)
        }

        binding.realTimeButton.setOnClickListener {
            val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(cameraIntent, Keys.CAMERA_CODE)
        }

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            if (requestCode == Keys.GALLERY_CODE) {
                val uriImage: Uri? = data?.data
                val bitmapImage = getBitmapFromUri(uriImage, contentResolver)
                if (bitmapImage != null) {
                    detectFacesAndEmotions(bitmapImage)
                }
            }
            if (requestCode == Keys.CAMERA_CODE) {
                val bitmapImage = data?.extras?.get("data") as Bitmap
                detectFacesAndEmotions(bitmapImage)
            }
        }
    }

    fun getBitmapFromUri(uri: Uri?, contentResolver: ContentResolver): Bitmap? {
        val inputStream: InputStream? = uri?.let { contentResolver.openInputStream(it) }
        return BitmapFactory.decodeStream(inputStream)
    }

    private fun loadModelFile(modelFileName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelFileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // Detect faces and run emotion detection
    private fun detectFacesAndEmotions(bitmap: Bitmap) {
        val image = InputImage.fromBitmap(bitmap, 0)

        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .build()

        val detector = FaceDetection.getClient(options)
        detector.process(image)
            .addOnSuccessListener { faces ->
                val faceEmotionMap = mutableMapOf<Int, String>()
                var faceCount = 1

                for (face in faces) {
                    // Crop each face from the original bitmap
                    val faceBitmap = cropFaceFromBitmap(bitmap, face)

                    // Run emotion detection on the cropped face
                    val emotion = runEmotionInference(faceBitmap)

                    // Store face number and emotion in the map
                    faceEmotionMap[faceCount] = emotion

                    // Increment face count
                    faceCount++
                }

                // Draw contours and labels on the bitmap
                val bitmapWithFaces = drawFaceContoursAndLabels(bitmap, faces)

                // Set the modified bitmap (with contours) on the original ImageView
                binding.Image.setImageBitmap(bitmapWithFaces)

                // Display the face-emotion map on a TextView
                displayFaceEmotionMap(faceEmotionMap)

                showToast("Number of faces detected: ${faces.size}")
            }
            .addOnFailureListener { e ->
                Log.e("FaceDetection", "Face detection failed", e)
            }
    }

    // Function to crop face region from the original bitmap
    private fun cropFaceFromBitmap(bitmap: Bitmap, face: Face): Bitmap {
        val left = face.boundingBox.left
        val top = face.boundingBox.top
        val width = face.boundingBox.width()
        val height = face.boundingBox.height()

        return Bitmap.createBitmap(bitmap, left, top, width, height)
    }

    // Run emotion detection for each face
    private fun runEmotionInference(faceBitmap: Bitmap): String {
        val inputBuffer = processImageForInference(faceBitmap)
        val outputBuffer = Array(1) { FloatArray(7) } // Assuming 7 emotion classes

        tflite.run(inputBuffer, outputBuffer)

        val maxIndex = outputBuffer[0].indices.maxByOrNull { outputBuffer[0][it] } ?: -1
        return getEmotion(maxIndex)
    }

    // Get emotion label based on index
    private fun getEmotion(index: Int): String {
        return when (index) {
            0 -> "Anger"
            1 -> "Disgust"
            2 -> "Fear"
            3 -> "Happiness"
            4 -> "Sadness"
            5 -> "Surprise"
            6 -> "Neutral"
            else -> "Unknown"
        }
    }

    // Display face number and emotion on a TextView
    private fun displayFaceEmotionMap(faceEmotionMap: Map<Int, String>) {
        // Check if the map is empty
        if (faceEmotionMap.isEmpty()) {
            binding.report.text = "Please enter an image with faces"
            binding.report.gravity = Gravity.CENTER // Center horizontally
        } else {
            // Create a string representation of the face emotions
            val mapText = faceEmotionMap.entries.joinToString(separator = "\n") { (faceNo, emotion) ->
                "Face $faceNo: $emotion"
            }
            binding.report.text = mapText // Update the TextView with the map content
        }

        // Center the text within the TextView

    }



    // Draw contours and labels on the bitmap
    private fun drawFaceContoursAndLabels(bitmap: Bitmap, faces: List<Face>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint().apply {
            style = Paint.Style.STROKE
            color = Color.RED
            strokeWidth = 4f
        }

        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 50f
            style = Paint.Style.FILL
        }

        var faceNumber = 1
        for (face in faces) {
            val bounds = face.boundingBox
            canvas.drawRect(bounds, paint)
            canvas.drawText("Face $faceNumber", bounds.left.toFloat(), bounds.top.toFloat() - 10, textPaint)
            faceNumber++
        }
        return mutableBitmap
    }

    // Process image for inference (resize and normalize)
    private fun processImageForInference(image: Bitmap): ByteBuffer {
        val resizedImage = Bitmap.createScaledBitmap(image, IMAGE_SIZE, IMAGE_SIZE, true)
        val inputBuffer = ByteBuffer.allocateDirect(IMAGE_SIZE * IMAGE_SIZE * 3 * 4) // 4 bytes per float
        inputBuffer.order(ByteOrder.nativeOrder())

        for (y in 0 until IMAGE_SIZE) {
            for (x in 0 until IMAGE_SIZE) {
                val pixel = resizedImage.getPixel(x, y)
                // Normalize the RGB values to [0, 1] range and put into ByteBuffer as float
                inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // Red
                inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // Green
                inputBuffer.putFloat((pixel and 0xFF) / 255.0f)          // Blue
            }
        }

        return inputBuffer
    }

    private fun showToast(message: String) {
        runOnUiThread {
            Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        }
    }
}
