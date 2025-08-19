from typing import List
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from .tools.sec_tools import SEC10KTool, SEC10QTool
from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

# Use GPT-4o-mini for good performance at reasonable cost
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
)

@CrewBase
class StockAnalysisCrew:
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, stock_symbol: str = "AMZN"):
        super().__init__()  # Call CrewBase.__init__()
        self.stock_symbol = stock_symbol.upper()

    @agent
    def financial_analyst_agent(self) -> Agent:
        # IMPORTANT: Put SEC tools first (deterministic, cached) then web tools.
        return Agent(
            config=self.agents_config['financial_analyst'],
            verbose=True,
            llm=llm,
            tools=[
                # Filings first (these are initialized with the ticker and are cached):
                SEC10QTool(ticker=self.stock_symbol),
                SEC10KTool(ticker=self.stock_symbol),
                # Then web tools as fallback:
                WebsiteSearchTool(),
                ScrapeWebsiteTool(),
            ],
        )

    @agent
    def research_analyst_agent(self) -> Agent:
        # Same tool order to encourage filings → web pattern
        return Agent(
            config=self.agents_config['research_analyst'],
            verbose=True,
            llm=llm,
            tools=[
                SEC10QTool(ticker=self.stock_symbol),
                SEC10KTool(ticker=self.stock_symbol),
                WebsiteSearchTool(),
                ScrapeWebsiteTool(),
            ],
        )

    @agent
    def investment_advisor_agent(self) -> Agent:
        # Advisor synthesizes; no filings tools needed here
        return Agent(
            config=self.agents_config['investment_advisor'],
            verbose=True,
            llm=llm,
            tools=[
                WebsiteSearchTool(),
                ScrapeWebsiteTool(),
            ],
        )

    @task
    def financial_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['financial_analysis'],
            agent=self.financial_analyst_agent(),
        )

    @task
    def research(self) -> Task:
        return Task(
            config=self.tasks_config['research'],
            agent=self.research_analyst_agent(),
        )

    @task
    def filings_analysis(self) -> Task:
        # If this duplicates financial_analysis in your YAML, consider removing it
        return Task(
            config=self.tasks_config['filings_analysis'],
            agent=self.financial_analyst_agent(),
        )

    @task
    def recommend(self) -> Task:
        return Task(
            config=self.tasks_config['recommend'],
            agent=self.investment_advisor_agent(),
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Stock Analysis crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )