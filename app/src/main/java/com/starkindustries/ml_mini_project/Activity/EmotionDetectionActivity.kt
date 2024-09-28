package com.starkindustries.ml_mini_project.Activity
import android.content.ContentResolver
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Base64
import android.util.Log
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.databinding.DataBindingUtil
import com.starkindustries.ml_mini_project.Keys.Keys
import com.starkindustries.ml_mini_project.R
import com.starkindustries.ml_mini_project.databinding.ActivityEmotionDetectionBinding
import java.io.ByteArrayOutputStream
import java.io.InputStream
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import java.io.IOException
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class EmotionDetectionActivity : AppCompatActivity() {
    lateinit var binding:ActivityEmotionDetectionBinding
    val client = OkHttpClient()
    private lateinit var tflite: Interpreter
    private val IMAGE_SIZE = 22
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_emotion_detection)
        binding=DataBindingUtil.setContentView(this,R.layout.activity_emotion_detection)
        try {
            val tfliteModel = loadModelFile("model.tflite")
            tflite = Interpreter(tfliteModel)
        } catch (e: Exception) {
            e.printStackTrace()
        }
        binding.GalleryButton.setOnClickListener(){
            val intent = Intent(Intent.ACTION_PICK)
            intent.setData(MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent,Keys.GALLERY_CODE)
        }
        binding.realTimeButton.setOnClickListener(){
            val cameraIntet=Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(cameraIntet,Keys.CAMERA_CODE)
        }
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(resultCode== RESULT_OK){
            if(requestCode==Keys.GALLERY_CODE){
                var uriImage: Uri? = data?.getData()
                var bitmapImage = getBitmapFromUri(uriImage, contentResolver)
                if (bitmapImage != null) {
                    runInference(bitmapImage)
                }
                val processedImage = bitmapImage?.let { processImage(it) }
                binding.Image.setImageBitmap(bitmapImage)
            }
            if(requestCode==Keys.CAMERA_CODE){
                var bitmapImage:Bitmap=data?.extras?.get("data") as Bitmap
                binding.Image.setImageBitmap(bitmapImage)
            }
        }
    }
    fun getBitmapFromUri(uri: Uri?, contentResolver: ContentResolver): Bitmap? {
        val inputStream: InputStream? = uri?.let { contentResolver.openInputStream(it) }
        return BitmapFactory.decodeStream(inputStream)
    }



    //    fun processImage(image: Bitmap): Array<Array<Array<FloatArray>>> {
//        // Resize the image to 224x224
//        val resizedImage = Bitmap.createScaledBitmap(image, 224, 224, true)
//
//        // Create a float array to hold the image data
//        val input = Array(1) { Array(224) { Array(224) { FloatArray(3) } } }
//
//        // Normalize the pixel values (between 0 and 1)
//        for (x in 0 until 224) {
//            for (y in 0 until 224) {
//                val pixel = resizedImage.getPixel(x, y)
//                // Extract RGB values
//                input[0][x][y][0] = ((pixel shr 16) and 0xFF) / 255.0f  // Red
//                input[0][x][y][1] = ((pixel shr 8) and 0xFF) / 255.0f   // Green
//                input[0][x][y][2] = (pixel and 0xFF) / 255.0f           // Blue
//            }
//        }
//
//        return input  // The processed image is ready for the TFLite model
//    }
    private fun loadModelFile(modelFileName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelFileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    private fun processImage(image: Bitmap): FloatBuffer {
        // Resize the image to 224x224
        val resizedImage = Bitmap.createScaledBitmap(image, 224, 224, true)

        // Allocate a buffer for the input tensor
        val inputBuffer = ByteBuffer.allocateDirect(224 * 224 * 3 * 4) // 4 bytes per float
        inputBuffer.order(ByteOrder.nativeOrder())
        val floatBuffer = inputBuffer.asFloatBuffer()

        // Fill the buffer with normalized pixel values
        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = resizedImage.getPixel(x, y)
                floatBuffer.put(((pixel shr 16) and 0xFF) / 255.0f)  // Normalize Red
                floatBuffer.put(((pixel shr 8) and 0xFF) / 255.0f)   // Normalize Green
                floatBuffer.put((pixel and 0xFF) / 255.0f)           // Normalize Blue
            }
        }

        // Return the buffer for inference
        return floatBuffer
    }
    private fun handleResults(output: Array<FloatArray>) {
        // Example: show the result as a Toast
        val result = output[0].maxOrNull() // This depends on the model output structure
        Toast.makeText(this, "Inference Result: $result", Toast.LENGTH_LONG).show()
    }

    private fun runInference(image: Bitmap) {
        val inputBuffer = processImageForInference(image)

        // Create output buffer based on the model's output shape
        val outputBuffer = Array(1) { FloatArray(7) } // Assuming 7 classes

        // Run inference
        tflite.run(inputBuffer, outputBuffer)

        // Process predictions
        val predictions = outputBuffer[0] // This will have 7 elements

        // Find the class with the highest confidence
        val maxIndex = predictions.indices.maxByOrNull { predictions[it] } ?: -1
        val maxConfidence = predictions[maxIndex]

        // Show toast with the predicted class and its confidence
        val emotion = getEmotion(maxIndex)
        showToast("Predicted Emotion: $emotion with confidence: $maxConfidence")
    }

    private fun getEmotion(index: Int): String {
        return when (index) {
            0 -> "Anger"
            1 -> "Disgust"
            2 -> "Fear"
            3 -> "Happiness"
            4 -> "Sadness"
            5 -> "Surprise"
            6 -> "Neutral"
            else -> "Unknown Emotion"
        }
    }

    private fun showToast(message: String) {
        runOnUiThread {
            Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        }
    }
    private fun processImageForInference(image: Bitmap): Array<Array<Array<FloatArray>>> {
        // Resize image to model input size (224x224) and convert to FloatArray
        val resizedImage = Bitmap.createScaledBitmap(image, 224, 224, true)
        val inputArray = Array(1) { Array(224) { Array(224) { FloatArray(3) } } }

        for (x in 0 until 224) {
            for (y in 0 until 224) {
                val pixel = resizedImage.getPixel(x, y)
                inputArray[0][x][y][0] = ((pixel shr 16 and 0xFF) / 255.0f) // Red
                inputArray[0][x][y][1] = ((pixel shr 8 and 0xFF) / 255.0f)  // Green
                inputArray[0][x][y][2] = ((pixel and 0xFF) / 255.0f)         // Blue
            }
        }
        return inputArray
    }

    // Function to convert Bitmap to Base64
//    fun bitmapToBase64(bitmap: Bitmap?): String {
//        val byteArrayOutputStream = ByteArrayOutputStream()
//        if (bitmap != null) {
//            bitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream)
//        }
//        val byteArray = byteArrayOutputStream.toByteArray()
//        return Base64.encodeToString(byteArray, Base64.NO_WRAP)
//    }
//    fun detectEmotion(bitmap: Bitmap?) {
//        val base64Image = bitmapToBase64(bitmap)
//
//        // Define the API endpoint and request body
//        val url = "https://api.gemini-ai.com/emotion-detection" // Use actual API endpoint
//        val requestBody = RequestBody.create(
//            "application/json; charset=utf-8".toMediaTypeOrNull(),
//            """
//        {
//            "image": "$base64Image"
//        }
//        """.trimIndent()
//        )
//
//        // Create request
//        val request = Request.Builder()
//            .url(url)
//            .post(requestBody)
//            .addHeader("Authorization", Keys.API_KEY) // Replace with actual API key
//            .build()
//
//        // Make the request asynchronously
//        client.newCall(request).enqueue(object : Callback {
//            override fun onFailure(call: Call, e: IOException) {
//                e.printStackTrace()
//            }
//
//            override fun onResponse(call: Call, response: Response) {
//                if (response.isSuccessful) {
//                    val responseData = response.body?.string()
//                    // Parse and handle the response data
//                    Log.d("ValueListner",binding.report.toString())
//                } else {
//                    Log.d("errorListner",response.message)
//                }
//            }
//        })
//    }
}