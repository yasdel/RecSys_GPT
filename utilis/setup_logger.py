import logging
logging.basicConfig(
    level=logging.INFO,
    format='\033[1;36m %(funcName)s - %(asctime)s - %(levelname)s - %(message)s \033[0m', 
    datefmt='[%X]'
)
logger = logging.getLogger('myLogger')