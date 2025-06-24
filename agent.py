from typing import TypedDict, List, Union, Annotated, Sequence, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Explicitly set the backend before importing pyplot
import matplotlib.pyplot as plt
import re
import seaborn as sns
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    df: pd.DataFrame  # Holds the dataset throughout the graph

llm = ChatOpenAI(
    model='gpt-4o'
)

@tool
def calculate_statistics(df, col_name: Union[str, list[str]], group_by: Union[None, str, list[str]] = None) -> dict:
    """This function calculates statistics on a column denoted by col_name of a dataset denoted by df"""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    col_name = [col_name] if type(col_name) == str else col_name
    if group_by:
        if type(group_by) == str:
            grouped = df.groupby(group_by)[col_name]
            stats = {
            'mean': grouped.mean().round(5).to_dict(),
            'median': grouped.median().round(5).to_dict(),
            'max': grouped.max().round(5).to_dict(),
            'min': grouped.min().round(5).to_dict(),
            'stddev': grouped.std().round(5).to_dict()}
        
        elif type(group_by) == list:
            grouped = df.groupby(group_by)[col_name]
            stats = {
            'mean': grouped.mean().round(5).reset_index().to_dict(orient='records'),
            'median': grouped.median().round(5).reset_index().to_dict(orient='records'),
            'max': grouped.max().round(5).reset_index().to_dict(orient='records'),
            'min': grouped.min().round(5).reset_index().to_dict(orient='records'),
            'stddev': grouped.std().round(5).reset_index().to_dict(orient='records')}

    else:  
        stats = {
            'mean': df[col_name].mean().round(5).to_dict(),
            'median': df[col_name].median().round(5).to_dict(),
            'max': df[col_name].max().round(5).to_dict(),
            'min': df[col_name].min().round(5).to_dict(),
            'stddev': df[col_name].std().round(5).to_dict()}
    
    return stats

@tool
def generate_plot(
    df: dict,
    x_column: str,
    y_column: str,
    plot: str,
    title: str,
    group_by: Optional[str] = None
) -> str:
    """This function plots the data using pandas .plot() function and columns provided as parameters using the plot type deemed most logical for the scenarios by the LLM"""
    try:
        df = pd.DataFrame(df)

        # Validate x_column
        if x_column not in df.columns:
            return f"❌ x_column '{x_column}' not found in the dataset."

        # Validate y_column
        if isinstance(y_column, str):
            if y_column not in df.columns:
                return f"❌ y_column '{y_column}' not found in the dataset."
        elif isinstance(y_column, list):
            invalid_cols = [col for col in y_column if col not in df.columns]
            if invalid_cols:
                return f"❌ y_column(s) {invalid_cols} not found in the dataset."
        else:
            return "❌ y_column must be a string or list of strings."

        # Normalize plot type
        plot_type_mapping = {
            "line": "lineplot", "lineplot": "lineplot", "linplot": "lineplot", "lin": "lineplot",
            "bar": "barplot", "barplot": "barplot", "barchart": "barplot",
            "scatter": "scatterplot", "scatterplot": "scatterplot",
            "hist": "histplot", "histogram": "histplot", "histo": "histplot",
            "box": "boxplot", "boxplot": "boxplot", "boxchart": "boxplot",
        }

        plot = plot.lower().strip()
        if plot not in plot_type_mapping:
            return "❌ The required plot type is not supported yet."

        plot_func = plot_type_mapping[plot]

        # Group-by limitations
        unsupported_grouped_plots = ["barplot", "scatterplot", "histplot", "boxplot"]
        if group_by and plot_func in unsupported_grouped_plots:
            return f"⚠️ The plot type '{plot_func}' is not supported with 'group_by'. Please remove grouping or use a line plot."

        # Start plotting
        plt.figure(figsize=(8, 5))

        if not group_by:
            if plot_func == "barplot":
                sns.barplot(x=x_column, y=y_column, data=df)
            elif plot_func == "lineplot":
                sns.lineplot(x=x_column, y=y_column, data=df)
            elif plot_func == "scatterplot":
                sns.scatterplot(x=x_column, y=y_column, data=df)
            elif plot_func == "histplot":
                sns.histplot(data=df[y_column])
            elif plot_func == "boxplot":
                sns.boxplot(x=x_column, y=y_column, data=df)
            else:
                return f"❌ Plot type '{plot_func}' is not implemented yet."
        else:
            if plot_func == "lineplot":
                sns.lineplot(x=x_column, y=y_column, hue=group_by, data=df)
            else:
                return f"⚠️ Grouped plotting for '{plot_func}' is not yet supported."

        plt.title(title)
        plt.xticks(rotation=30)
        plt.tight_layout()

        # Sanitize filename
        filename_safe = re.sub(r'[^a-zA-Z0-9_]', '_', title.strip().lower()) + '.png'
        full_path = f'folder_path/{title}.png'
        plt.savefig(full_path)
        plt.close()
        return f"✅ Plot titled '{filename_safe}' saved to: {full_path}"
    except Exception as e:
        return f"❌ Plot generation failed: {str(e)}"


tools = [calculate_statistics, generate_plot]

llm = llm.bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content='''You are a Data Insights Agent designed to analyze tabular data, typically in the form of a CSV or DataFrame.

        You have access to two tools:
        1. `calculate_statistics`: Use this to compute summary statistics (mean, median, min, max, stddev), with optional grouping.
        2. `generate_plot`: Use this to create visualizations like bar plots, line plots, scatter plots, histograms, and box plots.

        📌 Guidelines for effective use:
        - Select the most appropriate tool based on user intent, inferring details when not explicitly provided.
        - Validate column names and types before tool usage. Politely ask clarifying questions only when absolutely necessary.
        - You are allowed to make reasoned assumptions or attempt defaults (e.g., default groupings, sensible column selection) where appropriate.
        - If the user asks for insights (e.g., “best region” or “top performer”), use `calculate_statistics` to guide your answer.
        - You may run tools multiple times for multiple columns or comparative views.
        - When a plot is requested without specific type or axis, choose a suitable visualization that fits the column types.
        - Keep explanations clear, concise, and aligned with the dataset.

        🚫 Do not attempt analysis without data or fabricate insights beyond the dataset.
        ✅ You are precise, insightful, and proactive — like a skilled data analyst who speaks Python and thinks visually.
        '''
    )

    # Add a preview of the dataset (first 5 rows only) as a SystemMessage
    df_preview = SystemMessage(content="📊 Data Preview:\n" + state["df"].head().to_markdown())
    print("Available columns:", state['df'].columns)
    # Combine everything and pass to the LLM
    response = llm.invoke([system_prompt, df_preview] + state["messages"])
    return {'messages': [response], 'df': state['df']}


def should_continue(state:AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)
graph.add_node('Model',model_call)
graph.add_node('Tools',ToolNode(tools))
graph.add_conditional_edges(
    'Model',
    should_continue,
    {
        'continue':'Tools',
        'end': END
    }
)
graph.add_edge('Tools','Model')
graph.add_edge(START,'Model')
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

def print_conversation(conversation):
    for msg in conversation["messages"]:
        # if isinstance(msg, HumanMessage):
        #     print(f"\n🧑 Human: {msg.content}")
        if isinstance(msg, AIMessage) and msg.content.strip():  # Only print non-empty AI responses
            print(f"\n🤖 AI: {msg.content}\n")

user_input = input("🧑 Human: ")
df = pd.read_csv(r'retail_data.csv')

while user_input != "exit":
    # response = app.invoke({"messages": [HumanMessage(content=user_input)], 'df': df})
    # print_conversation(response)
    print_stream(app.stream({"messages": [HumanMessage(content=user_input)], 'df':df}, stream_mode="values"))
    user_input = input("🧑 Human: ")

plt.close('all')
