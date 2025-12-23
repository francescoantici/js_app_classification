import os

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
        for e in lines:
            modified_line = ""
            # Remove comments
            if "#" in e and not(e.startswith(f"#{self.job_directives_keyword}")):
                idx = e.index("#")
                if idx > 2:
                    modified_line = e[:idx].replace("\n", "").strip() + "\n"
            # Remove directive keywors and keep the requested resources
            elif e.startswith(f"#{self.job_directives_keyword}"):
                modified_line = e.replace(f"#{self.job_directives_keyword}", "").replace("\n", "").strip() + "\n"
            # Keep the whole line if it's a command
            else:
                modified_line = e.replace("\n", "").strip() + "\n"
            # Remove empty lines
            if (modified_line == "\n") or (modified_line == " ") or (modified_line == ""):
                continue
            out += modified_line
        return out


        