package com.example.eyedtrack

import android.content.Intent
import android.os.Bundle
import android.view.WindowManager
import android.widget.ImageButton
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity

// Activity for the "Sounds" screen.
class SoundsActivity : AppCompatActivity() {

    // Called when the activity is created.
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Enable fullscreen mode by hiding the status bar.
        window.setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN
        )

        setContentView(R.layout.sounds_page) // Set the layout resource for this activity.

        // Initialize navigation buttons.
        val backButton = findViewById<ImageView>(R.id.back_button)
        val btnGoToSettings = findViewById<ImageButton>(R.id.settings_icon)
        val btnGoToProfile = findViewById<ImageButton>(R.id.profile_icon)
        val btnGoToHomePage = findViewById<ImageButton>(R.id.home_icon)

        // Close the activity when the back button is clicked.
        backButton.setOnClickListener {
            finish()
        }

        // Navigation to other activities.
        btnGoToProfile.setOnClickListener {
            startActivity(Intent(this, ProfileActivity::class.java))
        }

        btnGoToSettings.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        btnGoToHomePage.setOnClickListener {
            startActivity(Intent(this, HomePageActivity::class.java))
        }
    }
}