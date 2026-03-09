from enums.project_type import ProjectType
from startup import Startup

startup = Startup(
    project_type=ProjectType.GROUP_RESEARCH_PROJECT
)
startup.run_application()
