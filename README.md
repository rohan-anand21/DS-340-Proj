---
title: Hotel Sentiment Analysis
emoji: 🏨
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "latest"
app_file: Dockerfile
pinned: false
---

## Multi-Model Hotel Sentiment Analysis

This Space hosts a FastAPI demo comparing three sentiment models on hotel reviews: a simple LSTM, a stacked LSTM, and a fine-tuned BERT classifier. Enter a single review or upload a CSV with `reviews.text` and `reviews.rating` columns to see how each model scores the sentiment. Inside the repository you’ll find the training scripts, preprocessing utilities, and saved artifacts that back the interactive UI.
