import os
import re

class Script:

    def __init__(self, jid:int, body:str = None, application:str = None, path:str = None, job_directives_keyword:str = "SBATCH"):
        self.jid = jid
        self.body = body
        self.application = application
        self.path = path
        self.job_directives_keyword = job_directives_keyword

    @property
    def script_commands(self) -> str:
        # Return only the script commands without comments
        return self.__parse_commands()

    def __parse_commands(self):
        out = ""
        lines = self.body.split("\n")

        # remove heading for job id
        if len(lines) >= 3:
            if (
                lines[0] == "---------------------------------------------------" and
                lines[1].startswith("Jobscript for jobid:") and
                lines[2] == "---------------------------------------------------"
            ):
                lines = lines[3:]

        for line in lines:
            e = line.strip()

            if e.startswith("#") and (not e.startswith(f"#{self.job_directives_keyword} ")):
                # Remove comment line starting with "#" but not "#PBS "
                continue
            elif e:
                out += (e + "\n")

        # truncate to fit in context length limit
        out = out[:200000]

        return out



