package com.example.eyedtrack.utils

import android.content.Context
import android.os.Environment
import android.util.Log
import com.example.eyedtrack.model.AlertHistoryItem
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.BufferedReader
import java.io.File
import java.io.FileInputStream
import java.io.InputStreamReader
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Utility class to load and parse alert logs from driver monitoring logs
 */
class AlertLogLoader(private val context: Context) {
    private val TAG = "AlertLogLoader"
    private val ALERT_LOG_DIR = "driver_monitoring_logs"
    private val ALERT_LOG_FILE = "driver_monitoring.json"

    /**
     * Find the most recent log file in the driver_monitoring_logs directory
     * @return The most recent log file, or null if none found
     */
    private fun getMostRecentLogFile(): File? {
        try {
            // List of possible paths to check
            val possiblePaths = listOf(
                // App's internal storage
                File(context.filesDir, ALERT_LOG_FILE),
                File(context.filesDir, "$ALERT_LOG_DIR/$ALERT_LOG_FILE"),

                // App's external storage
                File(context.getExternalFilesDir(null), ALERT_LOG_FILE),
                File(context.getExternalFilesDir(ALERT_LOG_DIR), ALERT_LOG_FILE),

                // External storage root
                File(Environment.getExternalStorageDirectory(), "EyeDTrack/$ALERT_LOG_FILE"),
                File(Environment.getExternalStorageDirectory(), "EyeDTrack/$ALERT_LOG_DIR/$ALERT_LOG_FILE"),

                // Backend directory if accessible
                File(context.filesDir.parentFile?.parentFile, "BACKEND-AGAIN/$ALERT_LOG_DIR/$ALERT_LOG_FILE"),
                File("/storage/emulated/0/BACKEND-AGAIN/$ALERT_LOG_DIR/$ALERT_LOG_FILE")
            )

            // Check each path and log the results
            for (path in possiblePaths) {
                Log.d(TAG, "Checking path: ${path.absolutePath}")
                if (path.exists()) {
                    Log.d(TAG, "File exists at: ${path.absolutePath}")
                    if (path.canRead()) {
                        Log.d(TAG, "File is readable at: ${path.absolutePath}")
                        val lastModified = path.lastModified()
                        val formattedDate = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date(lastModified))
                        Log.d(TAG, "File size: ${path.length()} bytes, Last modified: $formattedDate")
                        return path
                    } else {
                        Log.w(TAG, "File exists but is not readable at: ${path.absolutePath}")
                    }
                }
            }

            // If no file found, create directory in app's files directory
            val internalDir = File(context.filesDir, ALERT_LOG_DIR)
            if (!internalDir.exists()) {
                internalDir.mkdirs()
                Log.d(TAG, "Created directory at: ${internalDir.absolutePath}")
            }

            // Create an empty log file if it doesn't exist
            val internalFile = File(internalDir, ALERT_LOG_FILE)
            if (!internalFile.exists()) {
                internalFile.createNewFile()
                Log.d(TAG, "Created empty log file at: ${internalFile.absolutePath}")
                return internalFile
            }

            Log.e(TAG, "No readable log file found in any location")
            return null

        } catch (e: Exception) {
            Log.e(TAG, "Error finding log file: ${e.message}")
            e.printStackTrace()
            return null
        }
    }

    /**
     * Parse a log file and add alerts to the provided list
     */
    private fun parseLogFile(file: File, alertItems: MutableList<AlertHistoryItem>) {
        try {
            val content = file.readText()
            if (content.isBlank()) {
                Log.w(TAG, "Log file is empty: ${file.absolutePath}")
                return
            }

            // Split by newlines as each line should be a JSON object
            content.split("\n").forEach { line ->
                if (line.isNotBlank()) {
                    try {
                        val json = JSONObject(line)
                        val timestamp = json.getString("timestamp")
                        val dateTime = timestamp.split("T")
                        val date = dateTime[0]
                        val time = dateTime[1].split(".")[0]

                        val behaviorCategory = json.getJSONObject("behavior_category")
                        val isDrowsy = behaviorCategory.optBoolean("is_drowsy", false)
                        val isYawning = behaviorCategory.optBoolean("is_yawning", false)
                        val isDistracted = behaviorCategory.optBoolean("is_distracted", false)

                        val alertType = when {
                            isDrowsy -> "Drowsiness"
                            isYawning -> "Yawning"
                            isDistracted -> "Distraction"
                            else -> "Normal"
                        }

                        val reason = when {
                            isDrowsy -> "The driver shows signs of drowsiness"
                            isYawning -> "The driver is yawning"
                            isDistracted -> "The driver is distracted"
                            else -> "Normal behavior"
                        }

                        val confidence = json.optDouble("behavior_confidence", 0.0)
                        val confidencePercentage = (confidence * 100).toInt()

                        alertItems.add(AlertHistoryItem(
                            date = date,
                            time = time,
                            alertType = alertType,
                            confidence = confidencePercentage,
                            reason = reason
                        ))
                    } catch (e: Exception) {
                        Log.e(TAG, "Error parsing JSON line: ${e.message}")
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error reading log file: ${e.message}")
        }
    }

    /**
     * Load alert logs from the driver monitoring logs directory
     */
    suspend fun loadAlertLogs(maxLogs: Int = 10): List<AlertHistoryItem> = withContext(Dispatchers.IO) {
        val alertItems = mutableListOf<AlertHistoryItem>()

        try {
            val logFile = getMostRecentLogFile()
            if (logFile != null) {
                parseLogFile(logFile, alertItems)
            }

            if (alertItems.isEmpty()) {
                Log.w(TAG, "No alerts found in log file, using sample data")
                alertItems.addAll(createSampleAlerts())
            }

            // Sort by date and time (most recent first)
            alertItems.sortByDescending { "${it.date}T${it.time}" }

            return@withContext alertItems.take(maxLogs)
        } catch (e: Exception) {
            Log.e(TAG, "Error loading alert logs: ${e.message}")
            return@withContext createSampleAlerts().take(maxLogs)
        }
    }

    /**
     * Create sample alerts for testing or when no logs are available
     */
    private fun createSampleAlerts(): List<AlertHistoryItem> {
        val now = System.currentTimeMillis()
        val dateFormat = SimpleDateFormat("yyyy-MM-dd", Locale.US)
        val timeFormat = SimpleDateFormat("HH:mm:ss", Locale.US)

        return listOf(
            AlertHistoryItem(
                date = dateFormat.format(now),
                time = timeFormat.format(now),
                alertType = "Drowsiness",
                confidence = 85,
                reason = "The driver shows signs of drowsiness"
            ),
            AlertHistoryItem(
                date = dateFormat.format(now - 300000), // 5 minutes ago
                time = timeFormat.format(now - 300000),
                alertType = "Yawning",
                confidence = 90,
                reason = "The driver is yawning"
            ),
            AlertHistoryItem(
                date = dateFormat.format(now - 600000), // 10 minutes ago
                time = timeFormat.format(now - 600000),
                alertType = "Distraction",
                confidence = 75,
                reason = "The driver is distracted"
            )
        )
    }

    /**
     * Reads the latest behavior flags (isDrowsy, isYawning, isDistracted) from the backend API.
     * @return Triple<Boolean, Boolean, Boolean> representing (isDrowsy, isYawning, isDistracted)
     */
    fun readLatestBehaviorFlags(): Triple<Boolean, Boolean, Boolean> {
        try {
            Log.d(TAG, "Attempting to connect to REAL BEHAVIOR API...")
            // REAL: Use driver monitoring logs endpoint for actual behavior detection
            val url = java.net.URL("http://172.20.10.3/api/latest_behavior")
            val connection = url.openConnection() as java.net.HttpURLConnection

            // Configure connection
            connection.requestMethod = "GET"
            connection.connectTimeout = 5000  // 5 second timeout
            connection.readTimeout = 5000
            connection.setRequestProperty("Accept", "application/json")
            connection.setRequestProperty("User-Agent", "EyeDTrack-Android/1.0")

            // Log connection attempt
            Log.d(TAG, "Connecting to REAL BEHAVIOR endpoint: ${url}")

            // Connect and get response
            connection.connect()
            val responseCode = connection.responseCode
            Log.d(TAG, "Response code: $responseCode")

            if (responseCode == 200) {
                val response = connection.inputStream.bufferedReader().use { it.readText() }
                Log.d(TAG, "Raw behavior API response: $response")

                val json = JSONObject(response)

                // Check if response is successful
                val success = json.optBoolean("success", false)
                Log.d(TAG, "Behavior API success status: $success")

                if (success) {
                    // Extract behavior flags from the response
                    val behaviorCategory = json.getJSONObject("behavior_category")
                    val isDrowsy = behaviorCategory.optBoolean("is_drowsy", false)
                    val isYawning = behaviorCategory.optBoolean("is_yawning", false)
                    val isDistracted = behaviorCategory.optBoolean("is_distracted", false)

                    Log.i(TAG, "✅ REAL BEHAVIOR API: isDrowsy=$isDrowsy, isYawning=$isYawning, isDistracted=$isDistracted")

                    // Only log if any behavior is detected to reduce noise
                    if (isDrowsy || isYawning || isDistracted) {
                        Log.w(TAG, "⚠️ RISKY BEHAVIOR DETECTED! This should trigger voice alert!")
                    }

                    connection.disconnect()
                    return Triple(isDrowsy, isYawning, isDistracted)
                } else {
                    val error = json.optString("error", "Unknown error")
                    Log.w(TAG, "API returned success=false: $error")
                }
            } else {
                Log.w(TAG, "API returned error code: $responseCode")
                try {
                    val errorResponse = connection.errorStream?.bufferedReader()?.use { it.readText() }
                    Log.w(TAG, "Error response: $errorResponse")
                } catch (e: Exception) {
                    Log.w(TAG, "Could not read error response: ${e.message}")
                }
            }

            connection.disconnect()

        } catch (e: java.net.SocketTimeoutException) {
            Log.e(TAG, "✗ API connection timed out: ${e.message}")
        } catch (e: java.net.ConnectException) {
            Log.e(TAG, "✗ Could not connect to API: ${e.message}")
        } catch (e: java.net.UnknownHostException) {
            Log.e(TAG, "✗ Unknown host (check IP address): ${e.message}")
        } catch (e: Exception) {
            Log.e(TAG, "✗ Error calling API: ${e.message}")
            e.printStackTrace()
        }

        // If API fails, don't fall back to file reading since Android can't access backend files
        Log.w(TAG, "API call failed, returning false for all behaviors")
        return Triple(false, false, false)
    }


}