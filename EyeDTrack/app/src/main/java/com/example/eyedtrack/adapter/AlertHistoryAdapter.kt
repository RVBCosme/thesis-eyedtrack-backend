package com.example.eyedtrack.adapter

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.eyedtrack.R
import com.example.eyedtrack.model.AlertHistoryItem

/**
 * Adapter for displaying alert history items in a RecyclerView
 */
class AlertHistoryAdapter(private var alertItems: List<AlertHistoryItem>) : 
    RecyclerView.Adapter<AlertHistoryAdapter.AlertViewHolder>() {

    class AlertViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val dateText: TextView = itemView.findViewById(R.id.alert_date)
        val timeText: TextView = itemView.findViewById(R.id.alert_time)
        val reasonText: TextView = itemView.findViewById(R.id.alert_reason)
        val behaviorText: TextView = itemView.findViewById(R.id.alert_behavior)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): AlertViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.alert_history_item, parent, false)
        return AlertViewHolder(view)
    }

    override fun onBindViewHolder(holder: AlertViewHolder, position: Int) {
        val currentItem = alertItems[position]
        
        holder.dateText.text = currentItem.date
        holder.timeText.text = currentItem.time
        holder.reasonText.text = currentItem.reason
        holder.behaviorText.text = currentItem.behaviorOutput
    }

    override fun getItemCount(): Int = alertItems.size

    fun updateAlerts(newAlerts: List<AlertHistoryItem>) {
        alertItems = newAlerts
        notifyDataSetChanged()
    }
} 