# 📊 Data Insights AI Agent (LangGraph-Powered)

An intelligent, tool-augmented AI agent built with **LangGraph** that can analyze and visualize tabular data, such as CSV files or pandas DataFrames. It leverages LLM capabilities and Python tools to assist in data exploration, reporting, and visualization.

---

## 🚀 Features

* 🔍 **Descriptive Statistics**

  * Calculate mean, median, min, max, and standard deviation.
  * Optionally group statistics by one or more columns.

* 📊 **Data Visualizations**

  * Supports bar plots, line plots, scatter plots, histograms, and box plots.
  * Uses `matplotlib` + `seaborn` under the hood.

* 🧠 **LLM-Powered Interaction**

  * Automatically decides when to compute or plot based on user instructions.
  * Intelligently asks clarifying questions if user input is incomplete.

---

## 🧰 Tools Implemented

### 1. `calculate_statistics`

* Computes descriptive statistics for specified columns.
* Supports optional grouping (e.g., "Sales by City").

### 2. `generate_plot`

* Visualizes relationships between numeric columns.
* Intelligently selects plotting strategy based on user input.

---

## 💡 Example Use Cases

* "Show me the average sales by city"
* "Plot sales against footfall as a scatter plot"
* "Give me the summary stats for basket size"
* "Generate a histogram of weekly footfall"

---

## ⚙️ Project Structure

```bash
📁 data-insights-agent/
├── agent.py              # Main LangGraph implementation
├── retail_data.csv       # Sample dataset
├── README.md
├── requirements.txt
```

---

## 📌 Why Use a Multi-Tool AI Agent?

Unlike traditional agents that rely solely on language models:

✅ **Tool-augmented agents** can:

* Execute actual statistical code (e.g., pandas groupby)
* Visualize data dynamically with Python plots
* Avoid hallucinating facts not in the dataset
* Ask for clarification before proceeding

This hybrid approach ensures **accuracy**, **flexibility**, and **explainability**—perfect for analysts, data scientists, and educators.

---

### 🧠 How It Helps

- Combines **language understanding** (LLM) with **functional execution** (tools)
- Can **route intelligently** between different tools based on user intent
- Ensures **precision** in logic-heavy tasks like conversions, which pure LLMs are not designed for
- Easily extensible — add more tools like date conversion, BMI calculators, tax estimators, etc.

---

> 🤖 This is not just an LLM — it's an **intelligent agent** that knows when to "think" and when to "act".

---

## 📦 Getting Started

```bash
pip install -r requirements.txt
python agent.py
```

Make sure to add your CSV or use the provided `retail_data.csv`. The app will prompt you for natural-language queries.

---

## 📌 Example Queries

Here are some sample prompts you can use to interact with the Data Insights Agent:

- **"Show me the average sales by city."**
- **"What is the median footfall for each week?"**
- **"Generate a scatter plot of Sales vs. Footfall."**
- **"Give me summary statistics for 'Avg Basket Size'."**
- **"Plot a bar chart comparing average sales across different cities."**
- **"Show the trend of footfall over time using a line plot."**
- **"Which week had the highest average basket size?"**
- **"Visualize the distribution of footfall using a histogram."**
- **"Compare the sales performance grouped by week and city."**

You can mix and match chart types, metrics, and grouping variables to uncover deeper insights.

---

### 🧑‍💻 Author
Jayesh Suryawanshi
- 🧠 Python Developer | 💡 AI Tools Builder | 🌍 Data & Engineering Enthusiast
- 📫 [LinkedIn](https://www.linkedin.com/in/jayesh-suryawanshi-858bb21aa/)

### 📄 License
MIT License

