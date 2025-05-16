import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch, TavilyExtract
from crewai import Agent, Task, Crew
from crewai import LLM
from crewai_tools import DirectoryReadTool, FileReadTool
from crewai.tools import BaseTool
from pydantic import Field
from pathlib import Path

load_dotenv()

api_key = os.environ.get("API_KEY")
model = os.environ.get("MODEL_NAME")
base_url = os.environ.get("BASE_URL")
# os.environ["TAVILY_API_KEY"] = "" # api key musi znajdnować się w pliku .env lub zmiennych systemowych

llm = LLM(
    model=f"hosted_vllm/{model}",
    temperature=0.5,
    base_url=base_url,
    api_key=api_key,
)

Path("bielik_output/atrakcje.md").touch()
docs_tool = DirectoryReadTool(directory="./bielik_output")
file_tool = FileReadTool()

search = TavilySearch(
    max_results=5,
    topic="general",
    include_answer=True,
    include_raw_content=True,
)

extract = TavilyExtract(
    extract_depth="basic",
    include_images=False,
)


class SearchTool(BaseTool):
    name: str = "Search"
    description: str = (
        "Useful for search-based queries. Use this to find information in Internet"
    )
    search: TavilySearch = Field(default_factory=TavilySearch)

    def _run(self, query: str) -> str:
        try:
            return self.search.invoke({"query": query})
        except Exception as e:
            return f"Error performing search: {str(e)}"


class WebExtractTool(BaseTool):
    name: str = "Extract"
    description: str = "Usfull for extracting information from webpage based on url."
    extract: TavilyExtract = Field(default_factory=TavilyExtract)

    def _run(self, url: str) -> str:
        try:
            return self.extract.invoke({"urls": [url]})
        except Exception as e:
            return f"Error performing extract: {str(e)}"


researcher = Agent(
    role="researcher",
    goal="Znaleźć informacje o ciekawych miejscach w Warszawie oraz pogodzie w stopniach Celciusza",
    backstory="Jesteś przewodnikiem który zna Warszawę jak własną kieszeń",
    verbose=True,
    allow_delegation=False,
    tools=[SearchTool(), WebExtractTool()],
    llm=llm,
)

writer = Agent(
    role="Redaktor czasopisma",
    goal="Wygenerować profesjonalny artykuł o Warszawie wraz z informacjami prognozie pogody na najbliższe dni",
    backstory="Jesteś redaktorem czasopisma podróżniczego który pisze o Warszawie i dodaje do artykułu prognozę pogody",
    verbose=True,
    allow_delegation=False,
    tools=[docs_tool, file_tool],
    llm=llm,
)

research = Task(
    description="Zebrać informacje o ciekawych miejscach w Warszawie",
    agent=researcher,
    expected_output="Lista z krótkim opisem miejsc w Warszawie",
)

weather = Task(
    description="Zebrać informacje o prognozie pogody w Warszawie",
    agent=researcher,
    expected_output="Prognoza pogody w najbliższych dniach w Warszawie",
)

write = Task(
    description="redagowanie artykułu na temat miejsc w Warszawie, wraz z prognozą pogody w stopniach Celciusza",
    agent=writer,
    expected_output="""
    Ekscytujący artykuł o miejscach w Warszawie wraz z prognozą pogody w formacie markdown. 
    """,
    output_file="bielik_output/atrakcje.md",
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[weather, research, write],
    verbose=True,
)

crew.kickoff()
