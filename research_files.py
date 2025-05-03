from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_ollama import OllamaLLM  # ✅ Use official Ollama integration

# 🧠 LLM setup
llm = OllamaLLM(model="llama3")  # Make sure `ollama` is running locally

# 🔍 Tools
search_tool = SerperDevTool()

# 🧠 Agents
researcher = Agent(
    role="Senior Research Analyst",
    goal="Track AI breakthroughs",
    backstory="An expert on emerging AI trends.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm,
    memory=None
)

writer = Agent(
    role="Content Writer",
    goal="Create engaging AI content",
    backstory="Transforms insights into compelling articles.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm,
    memory=None
)

# 🧩 Tasks
research_task = Task(
    description="Research and summarize the latest developments in AI.",
    agent=researcher,
    expected_output="3 concise bullet points of major developments."
)

writing_task = Task(
    description="Write a 500-word blog post based on the research.",
    agent=writer,
    expected_output="A detailed blog post about the top 3 AI breakthroughs."
)

# 🤖 Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True,
)

# 🚀 Run
result = crew.kickoff()
print("\n✅ Final Output:\n", result)
