package com.starkindustries.ml_mini_project.Activity
import android.content.Intent
import android.os.AsyncTask
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat.getContextForLanguage
import androidx.core.content.ContextCompat.startActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.databinding.DataBindingUtil
import com.starkindustries.ml_mini_project.R
import com.starkindustries.ml_mini_project.databinding.ActivityMainBinding
class MainActivity : AppCompatActivity() {
    lateinit var binding:ActivityMainBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        binding=DataBindingUtil.setContentView(this,R.layout.activity_main)
        Handler(Looper.getMainLooper()).postDelayed({
            var intent = Intent(this,EmotionDetectionActivity::class.java)
            startActivity(intent)
            finish()
        },3000)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
    }
    class LongOperation: AsyncTask<String?, Void?, String?>() {
        override fun doInBackground(vararg params: String?): String? {
            for(i in 1..3){
                try{
                    Thread.sleep(1000)
                }
                catch (e:Exception){
                    Thread.interrupted()
                }
            }
            return "Splash Screen"
        }
        override fun onPostExecute(result: String?) {
            super.onPostExecute(result)
        }
    }
}