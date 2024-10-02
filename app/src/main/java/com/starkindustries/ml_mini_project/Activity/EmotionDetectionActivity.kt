package com.starkindustries.ml_mini_project.Activity

import android.content.ContentResolver
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.Gravity
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

        // Run inference
        tflite.run(inputBuffer, outputBuffer)

        // Log raw output for debugging
        Log.d("EmotionDetection", "Raw model output: ${outputBuffer[0].joinToString()}")

        // Apply softmax to outputBuffer to get probabilities
        val probabilities = softmax(outputBuffer[0])

        // Log probabilities for debugging
        Log.d("EmotionDetection", "Probabilities: ${probabilities.joinToString()}")

        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: -1

        Log.d("EmotionDetection", "Predicted emotion index: $maxIndex")
        return getEmotion(maxIndex)
    }

    // Softmax function to normalize logits
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val exps = logits.map { Math.exp((it - maxLogit).toDouble()).toFloat() }
        val sumExps = exps.sum()
        return exps.map { it / sumExps }.toFloatArray()
    }

    // Get emotion label based on index
    private fun getEmotion(index: Int): String {
        return when (index) {
            0 -> "Angry"
            1 -> "Disgust"
            2 -> "Fear"
            3 -> "Happy"
            4 -> "Sad"
            5 -> "Surprise"
            6 -> "Neutral"
            else -> "Unknown"
        }
    }

    // Display face number and emotion on a TextView
    private fun displayFaceEmotionMap(faceEmotionMap: Map<Int, String>) {
        // Check if the map is empty
        if (faceEmotionMap.isEmpty()) {
            binding.report.text = "Please select an image with minimum 1 face in it."
            binding.report.setTypeface(null, Typeface.BOLD)
            binding.report.gravity = Gravity.CENTER // Center horizontally
        } else {
            binding.report.setTypeface(null, Typeface.BOLD)
            // Create a string representation of the face emotions
            val mapText = faceEmotionMap.entries.joinToString(separator = "\n") { (faceNo, emotion) ->
                "Face $faceNo: $emotion"
            }
            binding.report.text = mapText // Update the TextView with the map content
        }
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
        val inputBuffer = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        resizedImage.getPixels(intValues, 0, resizedImage.width, 0, 0, resizedImage.width, resizedImage.height)

        for (pixel in intValues) {
            val r = (pixel shr 16 and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }

        return inputBuffer
    }

    // Show a toast message
    private fun showToast(message: String) {
        Toast.makeText(applicationContext, message, Toast.LENGTH_SHORT).show()
    }
}
