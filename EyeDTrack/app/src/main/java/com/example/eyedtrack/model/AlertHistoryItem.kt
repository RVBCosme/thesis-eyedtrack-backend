package com.example.eyedtrack.model

/**
 * Data class representing an alert history item
 */
data class AlertHistoryItem(
    val date: String,
    val time: String,
    val alertType: String,
    val confidence: Int,
    val behaviorOutput: String = "RISKY BEHAVIOR DETECTED",
    val reason: String = "" // Kept for backward compatibility
) 