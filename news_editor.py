from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_ollama import OllamaLLM

# ✅ Use Ollama for local LLM
llm = OllamaLLM(model="llama3")

# ✅ Tool for agents
search_tool = SerperDevTool()

# 🔍 Researcher Agent
researcher = Agent(
    role="Senior Research Analyst",
    goal="Track cutting-edge AI developments",
    backstory="An expert in researching AI trends and technologies.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm,
    memory=None
)

# ✍️ Writer Agent
writer = Agent(
    role="Content Writer",
    goal="Create engaging newsletter content based on AI research",
    backstory="A skilled writer who turns complex research into accessible and compelling content.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm,
    memory=None
)

# 🪄 Editor Agent
editor = Agent(
    role="Newsletter Editor",
    goal="Polish content into a clear, concise, and professional newsletter",
    backstory="An experienced editor with a sharp eye for clarity, grammar, and audience tone.",
    verbose=True,
    allow_delegation=False,
    tools=[],
    llm=llm,
    memory=None
)

# 📌 Task 1: Research
research_task = Task(
    description="Research the latest developments in AI and summarize the top 3 breakthroughs.",
    agent=researcher,
    expected_output="A bullet point list summarizing 3 significant recent AI developments."
)

# 📝 Task 2: Write Draft
writing_task = Task(
    description="Write a 500-word newsletter article based on the research findings.",
    agent=writer,
    expected_output="A draft newsletter article in a clear and engaging tone suitable for a general audience."
)

# ✨ Task 3: Edit
editing_task = Task(
    description="Edit the newsletter draft for clarity, grammar, and flow. Ensure it's polished and publication-ready.",
    agent=editor,
    expected_output="A professionally edited and formatted newsletter article."
)

# 🧠 Crew Setup
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,
    verbose=True,
)

# 🚀 Run the Crew
result = crew.kickoff()
print("\n✅ Final Newsletter Output:\n", result)
