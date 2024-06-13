from app.lib.utils.utils import load_environment_variables, verify_environment_variables

env_vars = load_environment_variables()
VECTOR_INDEX_NAME = "vector"

# Verify the environment variables
if not verify_environment_variables(env_vars):
    raise ValueError("Some environment variables are missing!")

OPEN_AI_SECRET_KEY=env_vars["OPEN_AI_SECRET_KEY"]

NEO4J_URI=env_vars["NEO4J_URI"]
NEO4J_USERNAME=env_vars["NEO4J_USERNAME"]
NEO4J_PASSWORD=env_vars["NEO4J_PASSWORD"]
# LOGGER_NAME=env_vars["LOGGER_NAME"]