#!c:\users\alanm\desktop\projects\cardiovascular-analysis\cardiovascular-analysis\scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'jupyter-nbextensions-configurator==0.4.1','console_scripts','jupyter-nbextensions_configurator'
__requires__ = 'jupyter-nbextensions-configurator==0.4.1'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('jupyter-nbextensions-configurator==0.4.1', 'console_scripts', 'jupyter-nbextensions_configurator')()
    )
