from app.lib.utils.utils import load_environment_variables, verify_environment_variables
import app

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Load environment variables using the utility
        
        # logger.info(f"Running RAG service")
        links = [
            "https://data-plane-engine.genai.sc.eng.hitachivantara.com/demo/HCP_Shredding_service.pdf?response-content-disposition=attachment&AWSAccessKeyId=DPNACCESSKEYID&Signature=xR4ahzWhaFhZ0tnh6OLVRUjmquw%3D&Expires=1818094359&x-dpn-namespace=genai-ns"
        ]
        app.run(links)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()